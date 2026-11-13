"""Обязательные детерминированные расчёты EnergyTool без арифметики LLM."""

from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP

OPERATIONS = {
    "threshold_check", "solar_indicative_cost", "platform_indicative_cost",
    "energy_savings", "bill_compare", "simple_payback", "solar_maintenance",
}
ISOLATED = {"камчат", "магадан", "сахалин", "чукот"}
FIELD_QUESTIONS = {
    "power_value": "Какая у вас присоединённая мощность и в каких единицах она указана?",
    "station_type": "Какая станция нужна: сетевая для снижения платежа или гибридная с накопителями?",
    "placement": "Где планируется размещение: на скатной или плоской кровле, на земле или на фасаде?",
    "device_count": "Сколько приборов учёта нужно включить в сценарий?",
    "replaced_kwh": "Какой объём потребления в кВт·ч будет замещён за выбранный период?",
    "rate_rub_per_kwh": "Какую ставку в ₽/кВт·ч использовать: из вашего счёта или заданную для сценария?",
    "period": "За какой период выполняем расчёт: месяц, год или другой?",
    "before_total": "Какая сумма была в счёте за первый сопоставимый период?",
    "after_total": "Какая сумма получилась за второй сопоставимый период?",
    "investment_rub": "Какие инвестиции в рублях использовать для условного сценария?",
    "annual_net_effect_rub": "Какой чистый положительный эффект в рублях в год использовать для сценария?",
    "panel_cost_rub": "Какова стоимость именно панелей в рублях, без стоимости остального проекта?",
    "electrical_connection": "Есть ли электрическая связность между площадками хотя бы по 0,4 кВ?",
}

AMOUNT = r"(-?\d+(?:[ \u202f\u00a0]\d{3})*(?:[.,]\d+)?)"
RUBLES = r"\s*(?:₽|руб(?:\.|ль|ля|лей)?)(?!\w)"
PARAMETERS_BY_OPERATION = {
    "threshold_check": {"power_value", "power_unit", "region", "sites", "electrically_connected"},
    "solar_indicative_cost": {"power_value", "power_unit", "station_type", "placement"},
    "platform_indicative_cost": {"device_count", "explicit_scenario"},
    "energy_savings": {"replaced_kwh", "rate_rub_per_kwh", "period"},
    "bill_compare": {"before_total", "after_total"},
    "simple_payback": {"investment_rub", "annual_net_effect_rub", "explicit_scenario"},
    "solar_maintenance": {"panel_cost_rub"},
}


def infer_operation(text: str) -> str:
    q = text.lower()
    requested = re.search(r"посчит|рассчит|подсчит|вычисл|сколько|во сколько|на сколько|насколько|какая разниц", q)
    if re.search(r"окуп|payback", q) and requested \
            and re.search(r"сценар|условн|допустим|пусть|инвестиц|вложен", q):
        return "simple_payback"
    if requested and re.search(r"сэконом|экономи|замест|замен|снижен", q) and re.search(r"квт|ставк|тариф|₽|руб", q):
        return "energy_savings"
    if (requested or re.search(r"\bсравни(?:те)?\b", q)) \
            and re.search(r"сравни|разниц|измен|вырос|подорож|сниз|было[^\n]*стало", q) \
            and re.search(r"плат[её]ж|сч[её]т(?:[ауе]|ов|ами)?\b|было[^\n]*стало", q):
        return "bill_compare"
    if re.search(r"\bопт\b|оптов", q) and re.search(r"можно|можем|допуск|пройд|проходим|хватит|достаточ|подход", q):
        return "threshold_check"
    if requested and re.search(r"обслуж|очистк", q) and re.search(r"панел|станц", q):
        return "solar_maintenance"
    if re.search(r"панел|станц|солнеч", q) and re.search(r"стоим|стоит|цен|ценник", q) \
            and re.search(r"\d+\s*(?:кВт|МВт)\b", text, re.I):
        return "solar_indicative_cost"
    if re.search(r"прибор|сч[её]тчик|платформ", q) and re.search(r"сколько|стоим|цен", q) \
            and re.search(r"сценар|условн|допустим|пусть|по справочному ориентиру|по ориентиру", q):
        return "platform_indicative_cost"
    return ""


def _number(pattern: str, text: str):
    values = {"".join(match.group(1).split()).replace(",", ".") for match in re.finditer(pattern, text, re.I)}
    return next(iter(values)) if len(values) == 1 else None


def extract_parameters(operation: str, text: str) -> dict:
    """Осторожное извлечение явно названных значений для fallback-ветки."""
    params = {}
    powers = set(re.findall(
        AMOUNT + r"\s*(кВт|МВт|киловатт|мегаватт)(?![·*\-]?\s*ч)",
        text, re.I,
    ))
    shared_unit = re.search(
        r"\d+(?:[.,]\d+)?\s+(?:и|или)\s+\d+(?:[.,]\d+)?\s*"
        r"(?:кВт|МВт|киловатт|мегаватт)(?![·*\-]?\s*ч)",
        text, re.I,
    )
    per_site = re.search(
        r"\bпо\s+\d+(?:[.,]\d+)?\s*(?:кВт|МВт|киловатт|мегаватт)(?![·*\-]?\s*ч)",
        text, re.I,
    )
    power = next(iter(powers)) if len(powers) == 1 and not shared_unit and not per_site else None
    if power:
        params["power_value"] = "".join(power[0].split()).replace(",", ".")
        params["power_unit"] = "MW" if power[1].lower().startswith("м") else "kW"
    energy = _number(
        r"(?:замещ\w*|замен\w*|снижен\w*\s+на)\s+" + AMOUNT + r"\s*кВт[·*\-]?ч", text,
    )
    if energy:
        params["replaced_kwh"] = energy
    rate = _number(AMOUNT + RUBLES + r"\s*(?:/|за)\s*кВт[·*\-]?ч", text)
    if rate:
        params["rate_rub_per_kwh"] = rate
    period = re.search(r"\b(?:за|в)\s+(месяц|год|сутки|неделю)\b", text, re.I)
    if period:
        params["period"] = period.group(1).lower()
    devices = _number(r"(\d+)\s*(?:прибор|сч[её]тчик)", text)
    if devices:
        params["device_count"] = devices
    before = _number(r"\bбыл[оаи]?\s+" + AMOUNT + RUBLES, text)
    after = _number(r"\bстал[оаи]?\s+" + AMOUNT + RUBLES, text)
    if before is not None:
        params["before_total"] = before
    if after is not None:
        params["after_total"] = after
    if operation == "threshold_check":
        region = ""
        for match in re.finditer(r"\b(?:камчат|магадан|сахалин|чукот)\w*", text, re.I):
            prefix = text[max(0, match.start() - 12):match.start()].lower()
            if not re.search(r"\bне\s+(?:на\s+|в\s+)?$", prefix):
                region = match.group()
                break
        if region:
            params["region"] = region
        sites = _number(r"(\d+)\s*(?:площадк\w*|объект\w*|завод\w*)", text)
        if sites:
            params["sites"] = sites
        elif re.search(r"(?:суммир|объедин|сложи)\w*[^.!?\n]*"
                       r"(?:площадк\w*|объект\w*|завод\w*)|"
                       r"(?:площадок|площадки|объектов|объекты|заводов|заводы)"
                       r"[^.!?\n]*(?:суммир|объедин|сложи)", text, re.I):
            params["sites"] = 2
        if re.search(r"(?:нет\s+электрическ\w*\s+связ|электрическ\w*\s+не\s+связан)", text, re.I):
            params["electrically_connected"] = False
        elif re.search(r"электрическ\w*\s+связан", text, re.I):
            voltage = _number(r"электрическ\w*\s+связан\w*\s+(?:по|на)\s*"
                              r"(\d+(?:[.,]\d+)?)\s*кВ\b", text)
            if voltage is not None:
                params["electrically_connected"] = Decimal(voltage) >= Decimal("0.4")
    if operation == "solar_indicative_cost":
        if re.search(r"гибрид|накопител", text, re.I):
            params["station_type"] = "hybrid"
        elif re.search(r"сетев", text, re.I):
            params["station_type"] = "grid"
        if re.search(r"скатн", text, re.I):
            params["placement"] = "pitched_roof"
        elif re.search(r"плоск.*кровл", text, re.I):
            params["placement"] = "flat_roof"
        elif re.search(r"земл", text, re.I):
            params["placement"] = "ground"
        elif re.search(r"фасад|вертикаль", text, re.I):
            params["placement"] = "facade"
        elif re.search(r"кровл", text, re.I):
            params["placement"] = "rooftop"
    if operation in ("simple_payback", "platform_indicative_cost"):
        params["explicit_scenario"] = bool(re.search(
            r"сценар|условн|допустим|пусть|по справочному ориентиру|по ориентиру", text, re.I))
    if operation == "simple_payback":
        params["investment_rub"] = _number(r"(?:инвестици\w*|вложени\w*)\s*[—:–]?\s*" + AMOUNT + RUBLES, text)
        params["annual_net_effect_rub"] = _number(
            r"(?:чист\w*\s+эффект|чист\w*\s+экономи\w*)\s*[—:–]?\s*" + AMOUNT + RUBLES + r"\s+в год",
            text,
        )
    if operation == "solar_maintenance":
        params["panel_cost_rub"] = _number(r"панели\s+сто[ия]т\s+" + AMOUNT + RUBLES, text)
    return {key: value for key, value in params.items() if key in PARAMETERS_BY_OPERATION.get(operation, ())}


def decimal_value(value) -> Decimal:
    if isinstance(value, bool) or value is None:
        raise ValueError("not a number")
    text = str(value).strip().replace(" ", "").replace(",", ".")
    try:
        result = Decimal(text)
    except InvalidOperation as exc:
        raise ValueError("not a number") from exc
    if not result.is_finite():
        raise ValueError("not finite")
    return result


def _power(params: dict) -> Decimal:
    if params.get("power_kw") is not None:
        return decimal_value(params["power_kw"])
    value = decimal_value(params.get("power_value"))
    unit = str(params.get("power_unit", "kW")).lower()
    if unit in ("kw", "квт", "киловатт"):
        return value
    if unit in ("mw", "мвт", "мегаватт"):
        return value * 1000
    raise ValueError("power_unit")


def _energy(params: dict) -> Decimal:
    return decimal_value(params.get("replaced_kwh"))


def _result(operation: str, value, unit: str, formatted: str, **extra) -> dict:
    return {"status": "ok", "operation": operation, "value": str(value), "unit": unit,
            "formatted": formatted, "missing": [], **extra}


def _needs(operation: str, fields: list, reason: str = "") -> dict:
    return {"status": "needs_input", "operation": operation, "missing": fields,
            "questions": [FIELD_QUESTIONS[f] for f in fields if f in FIELD_QUESTIONS],
            "reason": reason, "formatted": ""}


def _invalid(operation: str, reason: str, fields: list | None = None) -> dict:
    fields = fields or []
    return {"status": "invalid", "operation": operation, "missing": fields,
            "questions": [FIELD_QUESTIONS[f] for f in fields if f in FIELD_QUESTIONS],
            "reason": reason, "formatted": ""}


def _unsupported(operation: str, reason: str) -> dict:
    return {"status": "unsupported", "operation": operation, "missing": [], "reason": reason, "formatted": ""}


def _money(value: Decimal) -> str:
    rounded = value.quantize(Decimal("1"), rounding=ROUND_HALF_UP)
    return f"{int(rounded):,}".replace(",", " ") + " ₽"


def _number_text(value: Decimal) -> str:
    text = format(value.normalize(), "f")
    return text.rstrip("0").rstrip(".") if "." in text else text


def calculate(request: dict) -> dict:
    """Выполняет операцию или возвращает точные недостающие параметры."""
    if not isinstance(request, dict):
        return _unsupported("", "Некорректное описание расчёта.")
    operation = str(request.get("operation", ""))
    if operation not in OPERATIONS:
        return _unsupported(operation, "Такая операция не поддерживается локальным калькулятором.")
    params = dict(request.get("parameters") or {})
    for key in ("object", "period", "provenance"):
        if key in request:
            params[key] = request[key]
    try:
        if operation == "threshold_check":
            region = str(params.get("region", "")).lower()
            if any(name in region for name in ISOLATED):
                return _result(operation, False, "eligible", "В этой изолированной энергосистеме оптовый рынок не функционирует, поэтому выход на него недоступен.")
            sites = params.get("sites", 1)
            if int(sites) > 1 and params.get("electrically_connected") is not True:
                if params.get("electrically_connected") is False:
                    return _result(operation, False, "eligible", "Без электрической связности начиная с 0,4 кВ нельзя суммировать мощность площадок: проверка выполняется отдельно для каждой.")
                return _needs(operation, ["electrical_connection"])
            if params.get("power_kw") is None and params.get("power_value") is None:
                return _needs(operation, ["power_value"])
            try:
                kw = _power(params)
            except ValueError:
                return _invalid(operation, "Для порога нужна присоединённая мощность в кВт или МВт, а не потребление в кВт·ч.", ["power_value"])
            if kw < 0:
                return _invalid(operation, "Мощность не может быть отрицательной.", ["power_value"])
            if kw < 750:
                mode = "ниже порога 750 кВт: выход на оптовый рынок невозможен"
            elif kw < 20000:
                mode = "проход по мощности: участие возможно через энергосбытовую компанию-посредника, но не прямое"
            else:
                mode = "проход по мощности: возможно прямое участие в оптовом рынке"
            conditional = not region
            return _result(operation, kw >= 750, "eligible",
                           f"Присоединённая мощность {_number_text(kw)} кВт — {mode}."
                           + (" Вывод предварительный: регион не указан, географические исключения не проверены." if conditional else ""),
                           conditional=conditional, hourly_metering_required=kw >= 670)

        if operation == "solar_indicative_cost":
            missing = [f for f in ("station_type", "placement") if not params.get(f)]
            if params.get("power_kw") is None and params.get("power_value") is None:
                missing.append("power_value")
            if missing:
                return _needs(operation, missing)
            station = str(params["station_type"]).lower()
            placement = str(params["placement"]).lower()
            if station not in ("grid", "сетевая", "сетевой", "network") or placement not in ("rooftop", "roof", "кровля"):
                return _unsupported(operation, "Ориентир 70 ₽/Вт применим только к типовой сетевой станции на кровле; гибрид, земля и фасад требуют проектного расчёта.")
            kw = _power(params)
            if kw < 15 or kw > 500:
                return _unsupported(operation, "Индикативная формула применяется только в типовом диапазоне 15–500 кВт.")
            cost = kw * 1000 * 70
            return _result(operation, cost, "RUB", f"Индикативная стоимость типовой сетевой станции на кровле мощностью {_number_text(kw)} кВт — около {_money(cost)}. Это не итоговая цена проекта: она определяется после обследования и зависит от конфигурации.", formula="power_w*70_rub_per_w")

        if operation == "platform_indicative_cost":
            if params.get("explicit_scenario") is not True:
                return _unsupported(operation, "Справочное умножение доступно только для явного сценария. Стоимость реального проекта определяется отдельно по количеству приборов, конфигурации и составу работ.")
            if params.get("device_count") is None:
                return _needs(operation, ["device_count"])
            count = decimal_value(params["device_count"])
            if count <= 0 or count != count.to_integral_value():
                return _invalid(operation, "Количество приборов должно быть целым положительным числом.", ["device_count"])
            if params.get("volume_project"):
                return _unsupported(operation, "Объёмные внедрения рассчитываются отдельно, справочное умножение здесь не применяется.")
            cost = count * 25000
            return _result(operation, cost, "RUB/year", f"По справочному ориентиру для {_number_text(count)} приборов получается {_money(cost)} в год. Это арифметика сценария, а не цена проекта, монтажа или обещание скидки.", formula="devices*25000_rub_per_year")

        if operation == "energy_savings":
            missing = [f for f in ("replaced_kwh", "rate_rub_per_kwh") if params.get(f) is None]
            if not params.get("period"):
                missing.append("period")
            if missing:
                return _needs(operation, missing, "Для расчёта нужны объём замещения и применимая ставка за тот же период.")
            kwh = _energy(params)
            rate = decimal_value(params["rate_rub_per_kwh"])
            if kwh < 0 or rate < 0:
                return _invalid(operation, "Объём и ставка не могут быть отрицательными.")
            if str(params.get("rate_source", "")).lower() in ("kb_example", "example", "region_guess"):
                return _unsupported(operation, "Ставку из примера другого объекта или предположение по региону нельзя переносить на ваш расчёт.")
            savings = kwh * rate
            return _result(operation, savings, "RUB", f"По заданному сценарию снижение переменной части затрат за {params['period']} составит {_money(savings)}: {_number_text(kwh)} кВт·ч × {_number_text(rate)} ₽/кВт·ч. Это не прогноз выработки и не перерасчёт всего счёта: плата за мощность и другие компоненты не изменены.", formula="replaced_kwh*rate")

        if operation == "bill_compare":
            missing = [f for f in ("before_total", "after_total") if params.get(f) is None]
            if missing:
                return _needs(operation, missing, "Для сравнения нужны суммы двух сопоставимых периодов одного объекта.")
            before = decimal_value(params["before_total"])
            after = decimal_value(params["after_total"])
            if before < 0 or after < 0:
                return _invalid(operation, "Суммы счетов не могут быть отрицательными.")
            delta = after - before
            percent = None if before == 0 else delta / before * 100
            text = f"Изменение счёта: {_money(delta)}"
            text += "." if percent is None else f" ({_number_text(percent.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP))}%)."
            text += " По одним итоговым суммам причину изменения определить нельзя: нужны объём, стоимость энергии и плата за мощность по каждому периоду."
            return _result(operation, delta, "RUB", text, percent=percent, analysis_scope="total_only")

        if operation == "simple_payback":
            missing = [f for f in ("investment_rub", "annual_net_effect_rub") if params.get(f) is None]
            if missing:
                return _needs(operation, missing)
            if params.get("explicit_scenario") is not True:
                return _unsupported(operation, "Простой срок окупаемости вычисляется только как явно заданный условный сценарий, не как прогноз проекта.")
            investment = decimal_value(params["investment_rub"])
            effect = decimal_value(params["annual_net_effect_rub"])
            if investment < 0 or effect <= 0:
                return _invalid(operation, "Для простого срока нужны неотрицательные инвестиции и положительный годовой эффект.")
            years = investment / effect
            return _result(operation, years, "years", f"Условный простой срок окупаемости — {_number_text(years.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP))} года при заданных инвестициях и неизменном годовом эффекте. Это не прогноз проекта.", formula="investment/annual_net_effect")

        if operation == "solar_maintenance":
            if params.get("panel_cost_rub") is None:
                return _needs(operation, ["panel_cost_rub"])
            cost = decimal_value(params["panel_cost_rub"])
            if cost < 0:
                return _invalid(operation, "Стоимость панелей не может быть отрицательной.")
            annual = cost * Decimal("0.005")
            return _result(operation, annual, "RUB/year", f"Ориентир обслуживания — около {_money(annual)} в год, то есть 0,5% стоимости именно панелей. Накопители гибридной станции обслуживаются отдельно.", formula="panel_cost*0.005")
    except (ValueError, InvalidOperation, TypeError) as exc:
        return _invalid(operation, f"Не удалось проверить параметры расчёта: {exc}.")
    return _unsupported(operation, "Операция пока не реализована.")


def render_calculation(result: dict) -> str:
    """Формирует ответ только из фактического результата инструмента."""
    status = result.get("status")
    if status == "ok":
        return result.get("formatted", "")
    if status == "needs_input":
        questions = result.get("questions", [])
        return "Чтобы выполнить расчёт, уточните: " + " ".join(questions) if questions else result.get("reason", "Не хватает входных данных.")
    if status == "invalid":
        suffix = " ".join(result.get("questions", []))
        return (result.get("reason", "Проверьте параметры расчёта.") + " " + suffix).strip()
    return (result.get("reason", "Локальный расчёт для этого запроса не поддерживается.") +
            " Здесь нужен проектный расчёт по данным объекта; его лучше передать специалисту.")
