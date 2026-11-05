"""Qt-free chart projections over saved aggregates; no source IO or calculations."""
from dataclasses import dataclass
from decimal import Decimal

from Application.statistics_purchase_policy import STRICT_COMPLETED, EXTENDED_COMPLETED
from Application.statistics_rfm_presentation import SEGMENT_LABELS, WINDOW_LABELS
from Application.statistics_cohorts_core import HORIZONS


@dataclass(frozen=True)
class ChartData:
    title: str
    kind: str
    labels: tuple[str, ...]
    series: tuple[tuple[str, tuple[int | float | Decimal | None, ...]], ...]
    tooltips: tuple[tuple[str, ...], ...]
    inner_title: bool = False
    ranked: bool = False
    money: bool = False
    secondary: bool = False
    zero_is_data: bool = False
    percentage: bool = False
    value_axis_title: str = ""
    ranges: tuple[tuple[Decimal, Decimal], ...] = ()
    bases: tuple[tuple[int, int], ...] = ()

    @property
    def display_labels(self):
        return tuple(f"{i+1}. {label or ''}" if self.ranked else label or ""
                     for i,label in enumerate(self.labels))

    @property
    def empty(self):
        return not self.labels or not any(
            value is not None if self.zero_is_data else bool(value)
            for _, values in self.series for value in values)


def exact_number(value):
    number = Decimal(str(value))
    text = format(number, "f")
    whole, _, fraction = text.partition(".")
    grouped = f"{int(whole):,}".replace(",", " ")
    fraction = fraction.rstrip("0")
    return grouped + ("," + fraction if fraction else "")


def _policy(result, field, policy=STRICT_COMPLETED):
    saved = result.get(field)
    return (saved.get("policies", {}).get(policy) or {}) if saved else {}


def _empty(title, kind="bar"):
    return ChartData(title, kind, (), (), ())


def buyer_frequency_chart_data(result):
    policy = _policy(result, "buyer_statistics")
    rows = policy.get("frequency") or ()
    title = "Частота подтверждённых покупок"
    return ChartData(title, "bar", tuple({"1": "1 заказ", "2": "2 заказа", "3–5": "3–5 заказов",
               "6–10": "6–10 заказов", "11+": "11+ заказов"}.get(row[0], row[0]) for row in rows),
        (("Покупателей", tuple(row[1] for row in rows)),),
        (tuple(f"{row[0]}\nПокупателей: {exact_number(row[1])}" for row in rows),),
        value_axis_title="Покупателей")


def purchase_policy_comparison_chart_data(result):
    strict = _policy(result, "buyer_statistics")
    extended = _policy(result, "buyer_statistics", EXTENDED_COMPLETED)
    title = "Сравнение политик учёта покупок"
    if not strict or not extended:
        return _empty(title)
    keys = ("buyers", "repeat_buyers", "orders")
    labels = ("Покупатели", "Повторные покупатели", "Заказы")
    series = tuple((name, tuple(policy[key] for key in keys))
                   for name, policy in (("Подтверждённые покупки", strict),
                                        ("Расширенный учёт покупок", extended)))
    return ChartData(title, "bar", labels, series,
        tuple(tuple(f"{name}\n{label}: {exact_number(value)}" for label, value in zip(labels, values))
              for name, values in series), value_axis_title="Количество")


def buyer_turnover_chart_data(result, currency):
    if currency not in ("RUB", "KZT"):
        raise ValueError("Unsupported chart currency")
    rows = _policy(result, "buyer_statistics").get("money", {}).get(currency, {}).get("quintiles") or ()
    ranges = tuple((Decimal(row[0]), Decimal(row[1])) for row in rows)
    values = tuple(Decimal(row[4]) for row in rows)
    return ChartData(f"Полный оборот покупателей по группам — {currency}", "bar",
        tuple(f"Группа {i + 1}" for i in range(len(rows))), ((currency, values),),
        (tuple(f"Группа {i + 1}\nДиапазон: {exact_number(low)} — {exact_number(high)} {currency}\n"
               f"Покупателей: {exact_number(row[2])}\nПолный оборот: {exact_number(value)} {currency}"
               for i, (row, (low, high), value) in enumerate(zip(rows, ranges, values))),),
        money=True, secondary=currency == "KZT", zero_is_data=True,
        value_axis_title=f"Полный оборот, {currency}", ranges=ranges)


RFM_WINDOWS = ("LAST_12M", "LAST_24M", "LAST_48M")


def rfm_segments_chart_data(result, window="LAST_12M", currency="RUB"):
    if window not in RFM_WINDOWS or currency not in ("RUB", "KZT"):
        raise ValueError("Unsupported RFM chart selection")
    title = f"RFM-сегменты — {WINDOW_LABELS[window]}, {currency}"
    saved = _policy(result, "rfm_statistics").get("windows", {}).get(window, {}).get("currencies", {}).get(currency)
    if saved is None:
        return _empty(title, "horizontal")
    counts = {code: count for code, count, _ in saved["segments"]}
    labels = tuple(label for label, _ in SEGMENT_LABELS.values())
    values = tuple(counts.get(code) for code in SEGMENT_LABELS)
    return ChartData(title, "horizontal", labels, (("Покупателей", values),),
        (tuple(f"{WINDOW_LABELS[window]} · {currency}\n{label}\nПокупателей: "
               + ("Нет данных" if value is None else exact_number(value))
               for label, value in zip(labels, values)),), zero_is_data=True, value_axis_title="Покупателей")


def cohort_repeat_chart_data(result):
    horizons = _policy(result, "cohort_statistics").get("summary", {}).get("horizons")
    title = "Доля повторных покупок по горизонтам"
    if horizons is None:
        return _empty(title)
    rows = [horizons[str(day)] for day in HORIZONS]
    values = tuple(row["repeat_rate"] if row["eligible"] else None for row in rows)
    return ChartData(title, "bar", tuple(f"{day} дней" for day in HORIZONS),
        (("Повторная доля, %", values),),
        (tuple(f"{day} дней\nПокупатели с полным наблюдением: {exact_number(row['eligible'])}\n"
               f"С повторной покупкой: {exact_number(row['repeated'])}\nДоля: "
               + ("Нет данных" if value is None else exact_number(value) + " %")
               for day, row, value in zip(HORIZONS, rows, values)),),
        zero_is_data=True, percentage=True, value_axis_title="Повторная доля, %",
        bases=tuple((row["eligible"], row["repeated"]) for row in rows))


@dataclass(frozen=True)
class RetentionCell:
    value: float | Decimal | None
    status: str
    eligible: int | None
    active: int | None
    censored: int | None
    coverage_incomplete: int | None


@dataclass(frozen=True)
class RetentionRow:
    month: str
    buyers: int
    cells: tuple[RetentionCell, ...]


@dataclass(frozen=True)
class RetentionHeatmapData:
    rows: tuple[RetentionRow, ...]
    columns: tuple[str, ...] = tuple(f"M{i}" for i in range(1, 13))


def cohort_retention_heatmap_data(result):
    cohorts = _policy(result, "cohort_statistics").get("cohorts") or {}
    rows = []
    for month, row in sorted(cohorts.items()):
        saved = {cell["month_offset"]: cell for cell in row["retention"]}
        cells = []
        for offset in range(1, 13):
            cell = saved.get(offset, {})
            cells.append(RetentionCell(cell.get("retention_rate"), cell.get("status", "UNAVAILABLE"),
                cell.get("eligible"), cell.get("active"), cell.get("censored"), cell.get("coverage_incomplete")))
        rows.append(RetentionRow(month, row["cohort_size"], tuple(cells)))
    return RetentionHeatmapData(tuple(rows))


MONTHS = ('янв.', 'февр.', 'март', 'апр.', 'май', 'июнь', 'июль', 'авг.', 'сент.', 'окт.', 'нояб.', 'дек.')
MONTHS_FULL = ('Январь', 'Февраль', 'Март', 'Апрель', 'Май', 'Июнь', 'Июль', 'Август', 'Сентябрь', 'Октябрь', 'Ноябрь', 'Декабрь')

def month_label(month, full=False):
    year, number = month.split('-')
    return f'{(MONTHS_FULL if full else MONTHS)[int(number) - 1]} {year}'

def monthly_data(result, kind):
    fields = {'views': ('Динамика просмотров', 'action_monthly_dynamics', (('Просмотры', 1),)),
              'favorites': ('Динамика избранного', 'action_monthly_dynamics', (('Избранное', 3),))}
    title, field, columns = fields[kind]
    rows = result.get(field) or ()
    return ChartData(title, 'line', tuple(month_label(row[0]) for row in rows),
                     tuple((name, tuple(row[index] for row in rows)) for name, index in columns),
                     tuple(tuple(f'{month_label(row[0], full=True)}\n{name}: {exact_number(row[index])}'
                                 for row in rows) for name, index in columns), inner_title=True, secondary=kind == 'favorites')

def purchase_count_data(result, currency):
    index = {'RUB': 1, 'KZT': 3}[currency]
    rows = result.get('order_monthly_dynamics') or ()
    return ChartData(f'Динамика количества покупок — {currency}', 'line', tuple(month_label(row[0]) for row in rows),
                     ((currency, tuple(row[index] for row in rows)),),
                     (tuple(f'{month_label(row[0], full=True)}\nПокупок: {exact_number(row[index])}\nВалюта: {currency}'
                            for row in rows),), secondary=currency == 'KZT')

def interaction_frequency_data(result):
    rows = result.get('interaction_activity_distribution') or ()
    return ChartData('Распределение клиентов по количеству взаимодействий', 'bar', tuple(row[0] for row in rows),
                     (('Клиентов', tuple(row[1] for row in rows)),),
                     (tuple(f'Количество взаимодействий: {label}\nКлиентов: {exact_number(count)}\n'
                            f'Доля: {format(rate, ".2f").replace(".", ",")} %' for label, count, rate in rows),))

def distribution_data(result, kind):
    title, field, unit = {'basket': ('Состав корзины', 'purchase_basket_distribution', 'Заказов'),
                         'age': ('Возрастная структура клиентов', 'age_distribution', 'Клиентов')}[kind]
    rows = result.get(field) or ()
    return ChartData(title, 'bar', tuple(row[0] for row in rows), ((unit, tuple(row[1] for row in rows)),),
                     (tuple(f'{label}\n{unit}: {exact_number(count)}\nДоля: {rate:.2f} %'.replace('.', ',')
                            for label, count, rate in rows),))

def purchased_products_data(result):
    rows = (result.get('top_purchased_products') or ())[:10]
    return ChartData('Популярные товары по покупкам — Top-10', 'horizontal', tuple(row[1] for row in rows),
                     (('Покупки', tuple(row[2] for row in rows)),),
                     (tuple(f'Код: {code}\n{name}\nПокупки: {exact_number(count)}\nПокупатели: {exact_number(users)}\n'
                            f'Продано единиц: {exact_number(quantity)}' for code, name, count, users, quantity, *_ in rows),), ranked=True)

def activity_distribution_data(result, kind):
    field, noun = {'views': ('view_user_activity_distribution', 'просмотров'),
                   'favorites': ('favorite_user_activity_distribution', 'избранного')}[kind]
    rows = result.get(field) or ()
    return ChartData(f'Распределение клиентов по количеству {noun}', 'bar', tuple(row[0] for row in rows),
                     (('Клиентов', tuple(row[1] for row in rows)),),
                     (tuple(f'Количество {noun}: {label}\nКлиентов: {exact_number(count)}\n'
                            f'Доля: {format(rate, ".2f").replace(".", ",")} %' for label, count, rate in rows),),
                     secondary=kind == 'favorites')

def revenue_chart_data(result, currency):
    index = {'RUB': 2, 'KZT': 4}[currency]
    rows = result.get('order_monthly_dynamics') or ()
    values = tuple(Decimal(row[index]) for row in rows)
    return ChartData(f'Динамика суммы покупок — {currency}', 'line', tuple(month_label(row[0]) for row in rows),
                     ((currency, values),), (tuple(f'{month_label(row[0], full=True)}\nСумма покупок: '
                     f'{exact_number(value)} {currency}' for row, value in zip(rows, values)),), money=True, secondary=currency == 'KZT')

def payment_chart_data(result):
    rows = result.get('payment_type_distribution') or ()
    if len(rows) > 10:
        rows = sorted(rows, key=lambda row: (-row[1], row[0]))[:10]
    return ChartData('Оплата', 'horizontal', tuple(row[0] for row in rows), (('Заказов', tuple(row[1] for row in rows)),),
                     (tuple(f'{name}\nЗаказов с этим способом: {exact_number(count)}\nДоля заказов: {format(rate, ".2f").replace(".", ",")} %' for name, count, rate in rows),))

def product_ranking_data(result, kind):
    field, title, unit, user_unit, extra = {
        'views': ('top_viewed_products', 'Популярные товары по просмотрам', 'Просмотры', 'Клиенты с просмотрами', 'Добавления в избранное'),
        'favorites': ('top_favorited_products', 'Популярные товары по избранному', 'Добавления в избранное', 'Клиенты с избранным', 'Просмотры'),
    }[kind]
    rows = (result.get(field) or ())[:10]
    return ChartData(title, 'horizontal', tuple(row[1] for row in rows), ((unit, tuple(row[2] for row in rows)),),
                     (tuple(f'Код: {code}\n{name}\n{unit}: {exact_number(count)}\n{user_unit}: {exact_number(users)}\n'
                            f'{extra}: {exact_number(other)}\nПокупки: {exact_number(purchases)}'
                            for code, name, count, users, other, purchases in rows),), ranked=True)

def season_chart_data(result):
    # This snapshot field is already aggregated by Коллекция, never СезонНоски.
    rows = result.get('product_season_statistics') or ()
    return ChartData('Спрос по сезону', 'bar' if len(rows) <= 6 else 'horizontal', tuple(row[0] for row in rows),
                     (('Покупки', tuple(row[4] for row in rows)),),
                     (tuple(f'{name}\nТоваров: {exact_number(items)}\nПросмотры: {exact_number(views)}\n'
                            f'Добавления в избранное: {exact_number(favorites)}\nПокупки: {exact_number(purchases)}\n'
                            f'Продано единиц: {exact_number(quantity)}' for name, items, views, favorites, purchases, quantity in rows),))

def gender_chart_data(result):
    rows = result.get('gender_distribution') or ()
    return ChartData('Распределение по полу', 'donut', tuple(row[0] for row in rows),
                     (('Клиентов', tuple(row[1] for row in rows)),),
                     (tuple(f'{label}\nКлиентов: {exact_number(count)}\nДоля: {rate:.2f} %'.replace('.', ',')
                            for label, count, rate in rows),))
