"""Offline HTML rendering of the same Qt-free report used by GUI and XLSX."""
from datetime import date
from decimal import Decimal, ROUND_HALF_UP, localcontext
from html import escape
import json
from pathlib import Path
import re
from types import SimpleNamespace

from Application import statistics_report_presentation as report
from Application.analysis_filter import AnalysisFilter

SECTIONS = report.PAGE_LABELS
ASSETS = Path(__file__).with_name("statistics_html_assets")
IDENTIFIER_HEADERS = {"Код", "Код категории", "Артикул", "Системное название", "Система идентификаторов", "Статус"}


def display_text(value):
    return str(value)


def period_text(result):
    selection = AnalysisFilter.from_dict(result["analysis_filter"])
    if selection.start_date is not None:
        return "–".join(date.fromisoformat(v).strftime("%d.%m.%Y") for v in (selection.start_date, selection.end_date))
    from Application.statistics_presentation import _calculation_message
    message = _calculation_message(result)
    for prefix in ("Статистика рассчитана за периоды: ", "Статистика рассчитана за период: "):
        if message.startswith(prefix):
            return message[len(prefix):].removesuffix(".")
    return "Не указан"


def exact_number(value):
    if value is None:
        return "Нет данных"
    number = Decimal(str(value))
    with localcontext() as context:
        context.prec = max(len(number.as_tuple().digits), number.adjusted()+1, 1)+2
        rounded = number.quantize(Decimal(".01"), rounding=ROUND_HALF_UP)
    if rounded.is_zero():
        return "0"
    text = format(rounded, ",f").replace(",", " ")
    return text.rstrip("0").rstrip(".").replace(".", ",") if "." in text else text


def _cell(value, *, identifier=False, percentage=False):
    if isinstance(value, report.ReportCell):
        cell = value
    else:
        # Compatibility with the existing private numerical wrapper.
        from Application.statistics_export import _Numeric
        if isinstance(value, _Numeric):
            percentage = percentage or "%" in value.number_format
            value = value.value
        cell = report.cell_presentation(value, percentage=percentage)
    sort = (format(Decimal(str(cell.value)), "f") if cell.kind == "number" else
            cell.value.isoformat() if cell.kind == "date" else "" if cell.value is None else str(cell.value))
    return dict(text=cell.text, sort=sort, kind=cell.kind)


def _project_block(block):
    if block.kind == "heading" or block.kind == "text":
        return dict(kind=block.kind, text=block.title)
    if block.kind == "table":
        return dict(kind="table", headers=list(block.headers),
                    rows=[[_cell(c) for c in row] for row in report.table_cells(block)])
    if block.kind == "cards":
        return dict(kind="buyer_cards", rows=[[_cell(label), _cell(cell)] for label, cell in report.kpi_cells(block)])
    return dict(kind=block.kind, data=block.data, choices=block.choices)


def presentation(result):
    """Compatibility table inspection facade, backed entirely by the shared model."""
    model = report.report_presentation(result)
    sections = []
    for page in model.pages:
        blocks = []
        for group in page.groups:
            if page.key == "customers":
                blocks.append(dict(kind="heading", text=group.title))
            blocks.extend(_project_block(b) for b in group.blocks)
        sections.append(SimpleNamespace(key=page.key, title=page.title, blocks=blocks, groups=page.groups))
    return tuple(sections)


def _chart_spec(data, *, section, group, block):
    traces = []
    if not data.empty:
        for number, (name, values) in enumerate(data.series):
            trace = dict(name=name, exact=[None if value is None else str(value) for value in values],
                         hovertext=[escape(t).replace("\n", "<br>") for t in data.tooltips[number]],
                         hovertemplate="%{hovertext}<extra>%{fullData.name}</extra>")
            if data.kind == "donut":
                trace.update(type="pie", labels=[escape(label) for label in data.labels], values=trace["exact"],
                             hole=.5, textinfo="none", sort=False)
            elif data.kind == "horizontal":
                trace.update(type="bar", orientation="h", x=trace["exact"], y=[str(i) for i in range(len(data.labels))])
            else:
                trace.update(type="scatter" if data.kind == "line" else "bar",
                             x=[escape(label) for label in data.labels], y=trace["exact"])
                if data.kind == "line":
                    trace.update(mode="lines+markers")
            traces.append(trace)
    return dict(section=section, group=group, block=block, title=data.title, data=traces,
                labels=[escape(label) for label in data.display_labels], horizontal=data.kind == "horizontal",
                money=data.money, percentage=data.percentage, secondary=data.secondary,
                zero_is_data=data.zero_is_data, monthly=False)


def _specs(model):
    specs = []
    for section, page in enumerate(model.pages):
        for group in page.groups:
            for index, block in enumerate(group.blocks):
                if block.kind == "chart":
                    specs.append(_chart_spec(block.data, section=section, group=group.key, block=index))
                elif block.kind == "rfm":
                    choices = {window+"|"+currency: _chart_spec(data, section=section, group=group.key, block=index)
                               for window, currency, data in block.choices}
                    selected = dict(choices["LAST_12M|RUB"])
                    selected["variants"] = choices
                    specs.append(selected)
    return specs


def chart_specs(result):
    return _specs(report.report_presentation(result))


def _table_html(block, identifier, *, unavailable=False):
    headers, rows = block["headers"], block["rows"]
    controls = (f'<label class="search">Поиск по таблице <input type="search" data-table="{identifier}" '
                f'aria-label="Поиск по таблице {escape(headers[0], quote=True)}"></label>') if len(rows)>10 else ""
    head = "".join(f'<th scope="col" aria-sort="none"><button type="button" data-column="{i}">{escape(str(h))}'
                   '<span aria-hidden="true"> ↕</span></button></th>' for i,h in enumerate(headers))
    body = "".join(f'<tr data-order="{i}">' + "".join(
        f'<td data-sort="{escape(c["sort"],quote=True)}" data-kind="{c["kind"]}" class="{c["kind"]}">'
        f'{escape(c["text"])}</td>' for c in row) + "</tr>" for i,row in enumerate(rows))
    if not rows:
        body = f'<tr><td colspan="{len(headers)}">Нет данных для таблицы.</td></tr>'
    return (f'<div class="table-card">{controls}<div class="table-scroll" tabindex="0">'
            f'<table id="{identifier}"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div></div>')


def heatmap_style(cell):
    surface=(245,245,245)
    if cell.value is None:
        rgb=surface
    else:
        accent=(166,92,242)
        intensity=.10+.80*min(1.,max(0.,float(cell.value)/100.))
        rgb=tuple(round(a*(1-intensity)+b*intensity) for a,b in zip(surface,accent))
    background="rgb("+",".join(map(str,rgb))+")"
    if cell.value is not None and cell.status!="FULL":
        background="repeating-linear-gradient(45deg,"+background+" 0 3px,#f5f5f5 3px 5px)"
    return "background:"+background+";color:#202124;text-align:center;white-space:nowrap"


def _heatmap_html(data):

    head = "".join("<th>"+escape(label)+"</th>" for label in ("Когорта","Покупателей",*data.columns))
    rows = []
    for row in data.rows:
        cells = ["<td>"+escape(row.month)+"</td>", "<td>"+str(row.buyers)+"</td>"]
        for cell in row.cells:
            text = "—" if cell.value is None else report.cell_presentation(cell.value,percentage=True).text
            if cell.value is not None and cell.status != "FULL":
                text += " (частично)"
            cells.append('<td style="'+heatmap_style(cell)+'">'+escape(text)+"</td>")
        rows.append("<tr>"+"".join(cells)+"</tr>")
    if not rows:
        rows.append('<tr><td colspan="14">Нет данных для отображения.</td></tr>')
    return '<div class="table-card"><div class="table-scroll heatmap"><table><thead><tr>'+head+'</tr></thead><tbody>'+"".join(rows)+"</tbody></table></div></div>"


def _cards_html(values):
    return '<div class="kpis" data-count="'+str(len(values))+'">'+"".join(
        '<article class="kpi"><span>'+escape(label)+'</span><strong>'+escape(cell.text)+"</strong></article>"
        for label,cell in values)+"</div>"


def build_statistics_html(result, *, snapshot_diagnostics=None, presentation=None, check=lambda: None):
    """Whitelist only the shared aggregate model; no individual rows or source IO."""
    from plotly.offline import get_plotlyjs
    check()
    model = presentation if presentation is not None else report.report_presentation(result,snapshot_diagnostics=snapshot_diagnostics)
    specs = _specs(model)
    chart_indices = {(s["section"],s["group"],s["block"]):i for i,s in enumerate(specs)}
    def blocks_html(page_index,group):
        content = []
        for index,block in enumerate(group.blocks):
            check()
            identifier=f"table-{page_index}-{group.key}-{index}"
            if block.kind == "heading":
                content.append('<h3 class="section-title">'+escape(block.title)+"</h3>")
            elif block.kind == "text":
                content.append('<p class="report-note">'+escape(block.title)+"</p>")
            elif block.kind == "cards":
                content.append(_cards_html(report.kpi_cells(block)))
            elif block.kind == "table":
                content.append(_table_html(_project_block(block),identifier))
            elif block.kind in ("chart","rfm"):
                n=chart_indices[page_index,group.key,index]
                controls=""
                if block.kind == "rfm":
                    controls=('<div class="rfm-controls"><label>Период <select data-rfm-window="'+str(n)+'">'
                              '<option value="LAST_12M">12 месяцев</option><option value="LAST_24M">24 месяца</option>'
                              '<option value="LAST_48M">48 месяцев</option></select></label>'
                              '<label>Валюта <select data-rfm-currency="'+str(n)+'"><option>RUB</option><option>KZT</option></select></label></div>')
                inner_title = ('<h3>'+escape(block.data.title)+'</h3>') if block.kind=="chart" and block.data.inner_title else ""
                content.append('<article class="chart-card">'+inner_title+controls+f'<div id="chart-{n}" class="chart" data-chart="{n}" '
                               f'aria-label="{escape(specs[n]["title"],quote=True)}"></div></article>')
            elif block.kind == "heatmap":
                content.append(_heatmap_html(block.data))
        return "".join(content)
    navigation = "".join(f'<button role="tab" id="tab-{i}" aria-controls="section-{i}" '
                         f'aria-selected="{str(i==0).lower()}" tabindex="{0 if i==0 else -1}">'
                         +escape(page.title)+"</button>" for i,page in enumerate(model.pages))
    panels = []
    for i,page in enumerate(model.pages):
        content=[f'<section id="section-{i}" role="tabpanel" aria-labelledby="tab-{i}"'+(" hidden" if i else "")+">"]
        if page.key=="customers":
            content.append('<nav class="subnav" role="tablist" aria-label="Подразделы клиентов">')
            content.extend(f'<button role="tab" id="customer-tab-{g}" aria-controls="customer-section-{g}" '
                           f'aria-selected="{str(g==0).lower()}" tabindex="{0 if g==0 else -1}">'
                           +escape(group.title)+"</button>" for g,group in enumerate(page.groups))
            content.append("</nav>")
            for g,group in enumerate(page.groups):
                content.append(f'<section id="customer-section-{g}" role="tabpanel" aria-labelledby="customer-tab-{g}"'
                               +(" hidden" if g else "")+">"+blocks_html(i,group)+"</section>")
        else:
            content.append(blocks_html(i,page.groups[0]))
        content.append("</section>");panels.append("".join(content))
    metadata="<dl>"+"".join("<div><dt>"+escape(label)+"</dt><dd>"+escape(report.cell_presentation(value).text)+"</dd></div>"
                           for label,value in model.header)+"</dl>"
    payload=json.dumps(specs,ensure_ascii=False,allow_nan=False).replace("&",r"\u0026").replace("<",r"\u003c").replace(">",r"\u003e")
    payload=payload.replace("\u2028",r"\u2028").replace("\u2029",r"\u2029")
    library=re.sub(r"</script",r"<\/script",get_plotlyjs(),flags=re.IGNORECASE)
    css=(ASSETS/"report.css").read_text(encoding="utf-8")
    js=(ASSETS/"report.js").read_text(encoding="utf-8")
    global_cards=report.kpi_cells(report.ReportBlock("cards",values=model.global_cards))
    return ('<!doctype html><html lang="ru"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
            '<meta http-equiv="Content-Security-Policy" content="default-src \'none\'; script-src \'unsafe-inline\'; '
            'style-src \'unsafe-inline\'; img-src data: blob:; font-src data:; connect-src \'none\'; object-src \'none\'; '
            'base-uri \'none\'; form-action \'none\'"><title>KANZLER AI — Статистика</title><style>'+css+"</style></head><body>"
            '<header><div class="brand">KANZLER AI</div><h1>Статистика</h1>'+metadata+_cards_html(global_cards)+"</header>"
            '<nav role="tablist" aria-label="Разделы отчёта">'+navigation+"</nav><main>"+"".join(panels)+"</main>"
            '<noscript>Для переключения вкладок и графиков включите JavaScript.</noscript>'
            '<script>window.PlotlyConfig={MathJaxConfig:"local"};</script><script id="plotly-library">'+library+"</script>"
            '<script type="application/json" id="chart-data">'+payload+"</script><script>"+js+"</script></body></html>")


def export_statistics_html(result,path):
    Path(path).write_text(build_statistics_html(result),encoding="utf-8")
