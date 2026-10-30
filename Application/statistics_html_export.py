"""Standalone HTML over the same aggregate presentation as the agreed XLSX. No Qt/raw IO."""

from datetime import date
from decimal import Decimal, ROUND_HALF_UP, localcontext
from html import escape
import json
from pathlib import Path
import re

from Application import statistics_export as tables
from Application.analysis_filter import AnalysisFilter
from Application.statistics_presentation import _calculation_message, _date_time, ITEM_CLIENTS_LABEL


SECTIONS = ('Обзор', *tables.SHEET_NAMES[1:6])  # RFM is a subsection of Clients in HTML.
ASSETS = Path(__file__).with_name('statistics_html_assets')
TEXT_TRANSLATION = str.maketrans({'ё': 'е', 'Ё': 'Е', '—': '-', '–': '-'})
IDENTIFIER_HEADERS = {'Код', 'Код категории', 'Артикул', 'Системное название', 'Система идентификаторов', 'Статус'}


def display_text(value):
    return str(value).translate(TEXT_TRANSLATION)


def period_text(result):
    selection = AnalysisFilter.from_dict(result['analysis_filter'])
    if selection.start_date is not None:
        return ' - '.join(date.fromisoformat(value).strftime('%d.%m.%Y')
                          for value in (selection.start_date, selection.end_date))
    message = _calculation_message(result)
    for prefix in ('Статистика рассчитана за периоды: ', 'Статистика рассчитана за период: '):
        if message.startswith(prefix):
            return display_text(message[len(prefix):].removesuffix('.'))
    return 'Не указан'


def exact_number(value):
    """Round display only, with enough local precision for large Decimal values."""
    if value is None:
        return 'Нет данных'
    number = Decimal(str(value))
    with localcontext() as context:
        context.prec = max(len(number.as_tuple().digits), number.adjusted() + 1, 1) + 2
        rounded = number.quantize(Decimal('.01'), rounding=ROUND_HALF_UP)
    if rounded.is_zero():
        return '0'
    whole, _, fraction = format(rounded, 'f').partition('.')
    fraction = fraction.rstrip('0')
    sign = '-' if whole.startswith('-') else ''
    digits = whole.lstrip('-')
    grouped = ' '.join(reversed([digits[max(0, end - 3):end] for end in range(len(digits), 0, -3)]))
    return sign + grouped + (',' + fraction if fraction else '')


def _cell(value, *, identifier=False):
    fmt = None
    if isinstance(value, tables._Numeric):
        value, fmt = value.value, value.number_format
    if value is None:
        return {'text': 'Нет данных', 'sort': '', 'kind': 'empty'}
    if isinstance(value, date):
        return {'text': value.strftime('%d.%m.%Y'), 'sort': value.isoformat(), 'kind': 'date'}
    if isinstance(value, (int, float, Decimal)):
        text = exact_number(value)
        if fmt == tables.PERCENT_FORMAT:
            text += ' %'
        return {'text': text, 'sort': format(Decimal(str(value)), 'f'), 'kind': 'number'}
    return {'text': str(value) if identifier else display_text(value), 'sort': str(value), 'kind': 'text'}


class _Section:
    """A small sink for the existing pure table projections, with no workbook creation."""

    def __init__(self):
        self.blocks = []

    def heading(self, text):
        self.blocks.append({'kind': 'heading', 'text': display_text(text)})

    def table(self, headers, rows):
        rows = [[_cell(value, identifier=header in IDENTIFIER_HEADERS) for header, value in zip(headers, row)] for row in rows]
        self.blocks.append({'kind': 'table', 'headers': [display_text(header) for header in headers], 'rows': rows})

    def cards(self, values):
        self.table(('Показатель', 'Значение'), [
            (label, tables._Numeric(Decimal(str(value)), tables.PERCENT_FORMAT)
             if label.startswith('Доля') and value is not None and not isinstance(value, str) else value)
            for label, value in values])

    def buyer_cards(self, values):
        self.cards(values)
        # Same whitelist cells as the adjacent metric table, no DTO serialization.
        self.blocks.insert(len(self.blocks)-1, {'kind': 'buyer_cards', 'rows': self.blocks[-1]['rows']})


def _trace(name, values, labels, details=None, *, horizontal=False, line=False, pie=False):
    # Coordinates and raw sort values stay exact; only human-readable numbers are rounded.
    name = display_text(name)
    raw = [str(value) for value in values]
    hover = [escape(str(text)) for text in (details or [f'{label}\n{name}: {exact_number(value)}'
                                                      for label, value in zip(labels, values)])]
    hover = [text.replace('\n', '<br>') for text in hover]
    trace = {'name': name, 'hovertext': hover, 'exact': raw,
             'hovertemplate': '%{hovertext}<extra>%{fullData.name}</extra>'}
    if pie:
        trace.update(type='pie', labels=[escape(str(label)) for label in labels], values=raw, hole=.55,
                     textinfo='none', sort=False)
    elif horizontal:
        trace.update(type='bar', orientation='h', x=raw, y=[str(i) for i in range(len(labels))])
    else:
        trace.update(type='scatter' if line else 'bar', x=[escape(str(label)) for label in labels], y=raw)
        if line:
            trace.update(mode='lines+markers')
    return trace


def chart_specs(result):
    """Presentation of existing aggregate rows only; no identities, sources or new statistics."""
    charts = []

    def add(section, title, traces, *, horizontal=False, labels=None, monthly=False, money=False):
        if not any(Decimal(value) for trace in traces for value in trace['exact']):
            traces = []
        charts.append({'section': section, 'title': display_text(title), 'data': traces, 'horizontal': horizontal,
                       'labels': [escape(str(label)) for label in (labels or [])], 'monthly': monthly, 'money': money})

    months = result['action_monthly_dynamics']
    labels = [row[0] + '-01' for row in months]
    views = _trace('Просмотры', [row[1] for row in months], labels, line=True)
    favorites = _trace('Избранное', [row[3] for row in months], labels, line=True)
    add(0, 'Динамика просмотров', [views], monthly=True)
    add(0, 'Динамика избранного', [favorites], monthly=True)
    add(1, 'Динамика просмотров', [views], monthly=True)
    add(1, 'Динамика избранного', [favorites], monthly=True)
    orders = result['order_monthly_dynamics']
    labels = [row[0] + '-01' for row in orders]
    for section in (0, 2):
        for currency, index in (('RUB', 1), ('KZT', 3)):
            add(section, f'Количество покупок - {currency}',
                [_trace(currency, [row[index] for row in orders], labels, line=True)], monthly=True)
        for currency, index in (('RUB', 2), ('KZT', 4)):
            add(section, f'Сумма покупок — {currency}',
                [_trace(currency, [row[index] for row in orders], labels, line=True)], monthly=True, money=True)
    for section, field, title in (
        (1, 'view_user_activity_distribution', 'Клиенты по количеству просмотров'),
        (1, 'favorite_user_activity_distribution', 'Клиенты по количеству добавлений в избранное'),
        (2, 'purchase_basket_distribution', 'Количество позиций в корзине'),
        (4, 'gender_distribution', 'Пол клиентов'),
        (4, 'age_distribution', 'Возрастные группы'),
        (4, 'interaction_activity_distribution', 'Активность клиентов'),
        (4, 'purchase_order_distribution', 'Покупатели по числу заказов'),
    ):
        rows = result[field]
        labels = [display_text(row[0]) for row in rows]
        details = [f'{display_text(row[0])}\nКоличество: {exact_number(row[1])}\nДоля: {exact_number(row[2])} %' for row in rows]
        horizontal = field == 'age_distribution'
        traces = [_trace('Заказов' if section == 2 else 'Клиентов', [row[1] for row in rows], labels, details,
                         horizontal=horizontal, pie=field == 'gender_distribution')]
        if result['customers'] is None and field in ('gender_distribution', 'age_distribution'):
            traces = []
        add(section, title, traces, horizontal=horizontal, labels=labels if horizontal else None)
    rows = result['payment_type_distribution']
    rows = sorted(rows, key=lambda row: (-row[1], row[0]))[:10] if len(rows) > 10 else rows
    labels = [display_text(row[0]) for row in rows]
    add(2, 'Способы оплаты — до 10 категорий', [_trace('Заказов', [row[1] for row in rows], labels,
        [f'{display_text(label)}\nЗаказов: {exact_number(count)}\nДоля: {exact_number(rate)} %' for label, count, rate in rows], horizontal=True)],
        horizontal=True, labels=labels)
    for field, noun in (('top_viewed_products', 'Просмотры'), ('top_favorited_products', 'Избранное'),
                        ('top_purchased_products', 'Покупки')):
        rows = result[field][:10]
        labels = [f'{row[0]} · {display_text(row[1])[:25]}' for row in rows]
        # Include every available product measure, with the same column semantics as its full table.
        headers = {
            'top_purchased_products': ('Код', 'Название', 'Покупки', 'Покупатели', 'Продано единиц', 'Просмотры', 'Избранное'),
            'top_viewed_products': ('Код', 'Название', 'Просмотры', 'Клиенты с просмотрами', 'Избранное', 'Покупки'),
            'top_favorited_products': ('Код', 'Название', 'Избранное', 'Клиенты с избранным', 'Просмотры', 'Покупки'),
        }[field]
        details = ['\n'.join(f'{key}: {value if i == 0 else display_text(value) if i == 1 else exact_number(value)}'
                             for i, (key, value) in enumerate(zip(headers, row))) for row in rows]
        add(3, noun + ' товаров — Top-10', [_trace(noun, [row[2] for row in rows], labels, details,
            horizontal=True)], horizontal=True, labels=labels)
    rows = sorted(result['product_season_statistics'], key=lambda row: (-row[4], row[0]))[:10]
    labels = [display_text(row[0])[:32] for row in rows]
    details = [f'{display_text(name)}\nТоваров: {exact_number(items)}\nПросмотры: {exact_number(views)}\n'
               f'Избранное: {exact_number(fav)}\nПокупки: {exact_number(buys)}\nПродано единиц: {exact_number(quantity)}'
               for name, items, views, fav, buys, quantity in rows]
    add(3, 'Коллекции по покупкам — Top-10', [_trace('Покупки', [row[4] for row in rows], labels, details,
        horizontal=True)], horizontal=True, labels=labels)
    return charts


def presentation(result):
    sections = [_Section() for _ in SECTIONS]
    tables._summary(sections[0], result)
    sections[0].blocks[0]['rows'][0][1] = _cell(period_text(result))
    tables._actions_page(sections[1], result)
    tables._orders_page(sections[2], result)
    tables._products_page(sections[3], result)
    tables._clients_technical(sections[4], sections[5], result)
    tables._rfm_page(sections[4], result)
    return sections


def _table_html(block, identifier, *, unavailable=False):
    headers, rows = block['headers'], block['rows']
    if unavailable:
        # Missing Customers uses zero placeholders in the DTO. Preserve those raw sort values,
        # but never present unavailable demographic counts/shares as observed zeroes.
        rows = [[dict(cell, text='Нет данных', kind='empty') if column else cell
                 for column, cell in enumerate(row)] for row in rows]
    controls = (f'<label class="search">Поиск по таблице <input type="search" data-table="{identifier}" '
                f'aria-label="Поиск по таблице {escape(headers[0], quote=True)}"></label>') if len(rows) > 10 else ''
    head = ''.join(f'<th scope="col" aria-sort="none"><button type="button" data-column="{i}">{escape(header)}'
                   '<span aria-hidden="true"> ↕</span></button></th>' for i, header in enumerate(headers))
    body = ''.join(f'<tr data-order="{i}">' + ''.join(
        f'<td data-sort="{escape(cell["sort"], quote=True)}" data-kind="{cell["kind"]}" '
        f'class="{cell["kind"]}">{escape(cell["text"])}</td>' for cell in row) + '</tr>' for i, row in enumerate(rows))
    if not rows:
        body = f'<tr><td colspan="{len(headers)}">Нет данных для таблицы.</td></tr>'
    return (f'<div class="table-card">{controls}<div class="table-scroll" tabindex="0" aria-label="Таблица {escape(headers[0], quote=True)}">'
            f'<table id="{identifier}"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div></div>')


def build_statistics_html(result):
    """Render an offline whitelist projection, never serialize the whole incoming snapshot."""
    from plotly.offline import get_plotlyjs

    sections = presentation(result)
    charts = chart_specs(result)
    navigation = ''.join(f'<button role="tab" id="tab-{i}" aria-controls="section-{i}" '
                         f'aria-selected="{str(i == 0).lower()}" tabindex="{0 if i == 0 else -1}" data-section="{i}">{display_text(title)}</button>'
                         for i, title in enumerate(SECTIONS))
    overview = (
        ('Взаимодействия', 'total_interactions'), ('Просмотры', 'view_interactions'),
        ('Добавления в избранное', 'favorite_interactions'), ('Заказы', 'orders'), ('Покупатели', 'purchase_users'),
        ('Повторные покупатели', 'repeat_buyers'), (ITEM_CLIENTS_LABEL, 'interaction_users'),
        ('Доля повторных покупателей', 'repeat_buyer_rate'))
    kpi = ''.join(f'<article class="kpi"><span>{display_text(label)}</span><strong>{escape(_cell(result[field])["text"])}'
                  f'{" %" if field == "repeat_buyer_rate" else ""}</strong></article>' for label, field in overview)
    panels = []
    for index, (title, section) in enumerate(zip(SECTIONS, sections)):
        content = [f'<section id="section-{index}" role="tabpanel" aria-labelledby="tab-{index}"'
                   f'{" hidden" if index else ""}><h2>{display_text(title)}</h2>']
        if index == 0:
            content.append('<div class="kpis">' + kpi + '</div>')
        content.append('<div class="charts">')
        for chart_index, chart in enumerate(charts):
            if chart['section'] == index:
                chart_class = 'chart donut' if chart['data'] and chart['data'][0]['type'] == 'pie' else 'chart'
                content.append(f'<article class="chart-card"><h3>{escape(chart["title"])}</h3>'
                    f'<div id="chart-{chart_index}" class="{chart_class}" data-chart="{chart_index}" aria-label="{escape(chart["title"], quote=True)}">'
                    f'{"" if chart["data"] else "Нет данных для диаграммы."}</div></article>')
        content.append('</div>')
        for block_index, block in enumerate(section.blocks):
            if block['kind'] == 'heading':
                content.append(f'<h3 class="section-title">{escape(block["text"])}</h3>')
            elif block['kind'] == 'buyer_cards':
                content.append('<div class="kpis buyer-kpis">' + ''.join(
                    '<article class="kpi"><span>' + escape(row[0]['text']) + '</span><strong>' +
                    escape(row[1]['text']) + '</strong></article>' for row in block['rows']) + '</div>')
            else:
                unavailable = index == 4 and result['customers'] is None and block['headers'][0] in ('Пол', 'Возрастная группа')
                content.append(_table_html(block, f'table-{index}-{block_index}', unavailable=unavailable))
        content.append('</section>')
        panels.append(''.join(content))
    selection = AnalysisFilter.from_dict(result['analysis_filter'])
    metadata = '<dl>' + ''.join(f'<div><dt>{display_text(label)}</dt><dd>{escape(display_text(value))}</dd></div>' for label, value in (
        ('Период статистики', period_text(result)),
        ('Вид номенклатуры', tables._selection_values(selection.nomenclature_types)),
        ('Коллекции', tables._selection_values(selection.collections)),
        ('Дата расчета', _date_time(result['calculated_at'])))) + '</dl>'
    payload = json.dumps(charts, ensure_ascii=False, allow_nan=False).replace('&', r'\u0026').replace('<', r'\u003c').replace('>', r'\u003e')
    payload = payload.replace('\u2028', r'\u2028').replace('\u2029', r'\u2029')
    library = re.sub(r'</script', r'<\/script', get_plotlyjs(), flags=re.IGNORECASE)
    css = (ASSETS / 'report.css').read_text(encoding='utf-8')
    js = (ASSETS / 'report.js').read_text(encoding='utf-8')
    return ('<!doctype html><html lang="ru"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
        '<meta http-equiv="Content-Security-Policy" content="default-src \'none\'; script-src \'unsafe-inline\'; '
        'style-src \'unsafe-inline\'; img-src data: blob:; font-src data:; connect-src \'none\'; object-src \'none\'; '
        'base-uri \'none\'; form-action \'none\'">'
        '<title>KANZLER AI - Аналитический отчет</title><style>' + css + '</style></head><body>'
        '<header><div class="brand">KANZLER AI</div><h1>Аналитический отчет</h1>' + metadata + '</header>'
        '<nav role="tablist" aria-label="Разделы отчета">' + navigation + '</nav><main>' + ''.join(panels) + '</main>'
        '<noscript>Для интерактивных диаграмм и переключения разделов включите JavaScript.</noscript>'
        '<script>window.PlotlyConfig={MathJaxConfig:"local"};</script><script id="plotly-library">' + library + '</script>'
        '<script type="application/json" id="chart-data">' + payload + '</script><script>' + js + '</script></body></html>')


def export_statistics_html(result, path):
    Path(path).write_text(build_statistics_html(result), encoding='utf-8')
