"""Клиент OpenAI-совместимого API платформы: чат, эмбеддинги, реранкер."""

from __future__ import annotations

import os
import re
import sys
import time

import httpx


def env(name: str, default: str = "") -> str:
    value = os.getenv(name, "").strip()
    return default if not value or value == name else value


def env_num(name: str, default, lower, upper, kind=int):
    try:
        return max(lower, min(upper, kind(env(name))))
    except ValueError:
        return default


class DeadlineExceeded(RuntimeError):
    """Общий лимит времени прогона исчерпан — новых запросов не делаем."""


THINK = re.compile(r"<think>.*?</think>", re.S)


def strip_thinking(text: str) -> str:
    text = THINK.sub("", text)
    if "<think>" in text:  # размышление не закрыто — ответа нет
        text = text.split("<think>")[0]
    return text.replace("</think>", "").strip()


class Api:
    def __init__(self):
        self.base_url = (env("LLM_BASE_URL") or env("BASE_URL") or env("OPENAI_BASE_URL")).rstrip("/")
        self.llm_key = env("LLM_API_KEY") or env("OPENAI_API_KEY")
        self.emb_key = env("EMBEDDER_API_KEY") or self.llm_key
        self.rerank_key = env("RERANKER_API_KEY") or self.llm_key
        self.emb_url = (env("EMBEDDER_BASE_URL") or self.base_url).rstrip("/")
        self.rerank_url = (env("RERANKER_BASE_URL") or self.base_url).rstrip("/")
        self.llm_model = env("LLM_MODEL") or env("GEN_MODEL") or env("OPENAI_MODEL", "qwen/qwen3.8-27b")
        self.emb_model = env("EMB_MODEL", "baai/bge-m3")
        self.rerank_model = env("RERANKER_MODEL", "qwen/qwen3-reranker-8b")
        # Свои имена (EB_*): платформа передаёт TOP_K, MAX_TOKENS, TEMPERATURE под бейзлайн,
        # и они не должны молча перебивать настройки нашего решения.
        # Демо-замеры: с размышлениями судья стабильно выше (7.3 → 7.55 на 18 демо жюри,
        # 4 прогона). Пустые ответы от съеденного лимита ловит повтор с EB_THINKING_BUDGET.
        self.max_tokens = env_num("EB_MAX_TOKENS", 2500, 128, 8192)
        self.temperature = env_num("EB_TEMPERATURE", 0.0, 0.0, 2.0, float)
        self.timeout = env_num("EB_REQUEST_TIMEOUT", 60.0, 5.0, 600.0, float)
        self.embed_batch = env_num("EB_EMBED_BATCH", 16, 1, 256)
        self.use_embeddings = env("EB_USE_EMBEDDINGS", "1") != "0"
        self.can_rerank = env("EB_USE_RERANKER", "1") != "0"
        # Qwen3 — гибридная модель; режим размышлений отключаем ради скорости и чистого ответа.
        self.disable_thinking = env("EB_DISABLE_THINKING", "1") != "0"
        # chat_template_kwargs — vLLM (вероятно, прокси платформы); reasoning — OpenRouter;
        # enable_thinking — DashScope; none — ничего не слать.
        # none (по умолчанию) — размышления НЕ отключаем. chat_template_kwargs — vLLM (платформа);
        # reasoning — OpenRouter-подобные прокси; enable_thinking — DashScope.
        # Конфигурация бенча 8.75/10 на демо была с выключенным размышлением: на платформе
        # это «chat_template_kwargs», но дефолт оставляем «none» — по вечерним замерам ON ≥ OFF.
        self.thinking_param = env("EB_THINKING_PARAM", "chat_template_kwargs")
        self.thinking_budget = env_num("EB_THINKING_BUDGET", 3000, 500, 16000)
        self.thinking_overflows = 0
        self._thinking_param_ok = True
        self.deadline = None  # абсолютное time.perf_counter(), после которого запросы не делаем
        self.client = httpx.Client(timeout=self.timeout)

    def _remaining(self):
        return None if self.deadline is None else self.deadline - time.perf_counter()

    @property
    def configured(self) -> bool:
        return bool(self.base_url and self.llm_key)

    def _post(self, url: str, key: str, body: dict, attempts: int = 3) -> dict:
        last = None
        for attempt in range(attempts):
            remaining = self._remaining()
            if remaining is not None and remaining < 3:
                raise DeadlineExceeded("исчерпан общий лимит времени прогона")
            timeout = self.timeout if remaining is None else min(self.timeout, remaining - 1)
            try:
                r = self.client.post(url, headers={"Authorization": f"Bearer {key}"}, json=body, timeout=timeout)
                if r.status_code in (429, 500, 502, 503, 504):
                    raise httpx.HTTPStatusError(f"HTTP {r.status_code}", request=r.request, response=r)
                r.raise_for_status()
                return r.json()
            except httpx.HTTPStatusError as exc:
                last = exc
                if exc.response is not None and exc.response.status_code in (400, 401, 403, 404, 422):
                    raise
            except (httpx.TimeoutException, httpx.TransportError, ValueError) as exc:
                last = exc
            time.sleep(0.6 * (2 ** attempt))
        raise last

    # ---------- эмбеддинги ----------
    def embed(self, texts: list, batch: int | None = None) -> list:
        """Эмбеддинги пачками. Лимит размера пачки у провайдеров разный (бывает 20, 10…):
        при ответе 400 пачка уменьшается вдвое, и дальше используется уже меньший размер."""
        batch = batch or self.embed_batch
        out = []
        i = 0
        while i < len(texts):
            part = texts[i:i + batch]
            try:
                data = self._post(self.emb_url + "/embeddings", self.emb_key, {"model": self.emb_model, "input": part})
            except httpx.HTTPStatusError as exc:
                if exc.response is not None and exc.response.status_code == 400 and batch > 1:
                    batch = self.embed_batch = max(1, batch // 2)
                    continue
                raise
            items = sorted(data["data"], key=lambda d: d.get("index", 0))
            if len(items) != len(part):
                raise ValueError("Получены не все эмбеддинги")
            out.extend([float(x) for x in it["embedding"]] for it in items)
            i += len(part)
        return out

    # ---------- реранкер ----------
    def rerank(self, query: str, documents: list) -> list:
        data = self._post(self.rerank_url + "/rerank", self.rerank_key, {
            "model": self.rerank_model, "query": query, "documents": documents, "top_n": len(documents),
        }, attempts=2)
        scores = [float("-inf")] * len(documents)
        for item in data.get("results", data.get("data", [])):
            idx = int(item["index"])
            if 0 <= idx < len(documents):
                scores[idx] = float(item.get("relevance_score", item.get("score", 0.0)))
        return scores

    # ---------- генерация ----------
    def _thinking_fields(self) -> dict:
        """Как отключить размышления: у разных серверов разный параметр (EB_THINKING_PARAM)."""
        if not self.disable_thinking or not self._thinking_param_ok:
            return {}
        return {
            "chat_template_kwargs": {"chat_template_kwargs": {"enable_thinking": False}},  # vLLM / SGLang
            # OpenRouter: только провайдеры, которые поддерживают параметр, — иначе часть из них его
            # игнорирует или отклоняет, и модель думает (медленно, иногда пустой ответ).
            "reasoning": {"reasoning": {"enabled": False}, "provider": {"require_parameters": True}},
            "enable_thinking": {"enable_thinking": False},  # Alibaba DashScope
        }.get(self.thinking_param, {})

    def chat(self, messages: list, max_tokens: int | None = None, temperature: float | None = None, final_reasoning: bool = False, observe_final: bool = False) -> str:
        body = {
            "model": self.llm_model,
            "messages": messages,
            "max_tokens": max_tokens or self.max_tokens,
            "temperature": self.temperature if temperature is None else temperature,
            **self._thinking_fields(),
        }
        if final_reasoning:
            body.pop("chat_template_kwargs", None)
            body["reasoning"] = {"enabled": True}
            body["provider"] = {"require_parameters": True}
        try:
            data = self._post(self.base_url + "/chat/completions", self.llm_key, body)
        except httpx.HTTPStatusError as exc:
            # Сервер не знает параметр отключения размышлений — больше его не шлём.
            extra = set(self._thinking_fields())
            if not final_reasoning and extra and exc.response is not None and exc.response.status_code in (400, 422):
                print(f"[llm] сервер отклонил {sorted(extra)}, отправляем без него", file=sys.stderr)
                self._thinking_param_ok = False
                for key in extra:
                    body.pop(key, None)
                data = self._post(self.base_url + "/chat/completions", self.llm_key, body)
            else:
                raise
        choice = data["choices"][0]
        if final_reasoning or observe_final:
            usage = data.get("usage") or {}
            details = usage.get("completion_tokens_details") or {}
            message = choice.get("message") or {}
            print(f"[final-thinking] reasoning_tokens={details.get('reasoning_tokens', 'unknown')} reasoning_field={bool(message.get('reasoning') or message.get('reasoning_content') or message.get('reasoning_details'))} finish={choice.get('finish_reason')} completion_tokens={usage.get('completion_tokens', 'unknown')}", file=sys.stderr)
        text = strip_thinking(choice["message"].get("content") or "")
        truncated = choice.get("finish_reason") == "length"
        if (not text or truncated) and self._ran_out_thinking(choice, data):
            # Модель потратила весь лимит на размышления: ответ пуст или обрезан посреди слова.
            # Один повтор с запасом токенов: медленнее, но без пустого/обрезанного ответа.
            self.thinking_overflows += 1
            body["max_tokens"] = max(self.thinking_budget, 3 * body["max_tokens"])
            data = self._post(self.base_url + "/chat/completions", self.llm_key, body)
            text = strip_thinking(data["choices"][0]["message"].get("content") or "")
        return text

    @staticmethod
    def _ran_out_thinking(choice: dict, data: dict) -> bool:
        message = choice.get("message") or {}
        details = (data.get("usage") or {}).get("completion_tokens_details") or {}
        return (choice.get("finish_reason") == "length"
                or bool(message.get("reasoning") or message.get("reasoning_content"))
                or bool(details.get("reasoning_tokens"))
                or "<think>" in (message.get("content") or ""))
