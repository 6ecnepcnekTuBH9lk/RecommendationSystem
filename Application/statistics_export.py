"""XLSX projection of an already validated snapshot; no Qt, raw reads or recalculation."""

from dataclasses import dataclass
from datetime import date
from decimal import Decimal
import math
import os
from pathlib import Path
import tempfile

from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.table import Table, TableStyleInfo

from Application.statistics_presentation import (
    CATEGORY_HEADERS,
)
from Application import statistics_report_presentation as report
from Application.statistics_cache import parse_timestamp


SHEET_NAMES = report.PAGE_LABELS
INK = "242424"
HEADER_FILL = "D6E5F2"
SECTION_FILL = "EAF1F7"
BORDER_COLOR = "BDCCDA"
PERCENT_FORMAT = '#,##0.00" %"'  # Snapshot percentages are already on the 0..100 scale.
MAX_ROW_HEIGHT = 96
ROW_HEIGHT = 21


@dataclass(frozen=True)
class _Numeric:
    value: Decimal
    number_format: str


def _decimal(value, *, money=True):
    number = Decimal(value)
    places = len(format(number, "f").partition(".")[2].rstrip("0"))
    quantity_format = "#,##0" + ("." + "0" * places if places else "")
    return _Numeric(number, "#,##0.00" if money else quantity_format)


class _Sheet:
    """Plain metric/data tables; no charts, navigation or presentation tiles."""

    def __init__(self, worksheet, check=lambda: None):
        self.ws = worksheet
        self.check = check
        self.row = 1
        self.max_columns = 2
        self.ws.sheet_view.showGridLines = True
        self.ws.sheet_view.zoomScale = 90
        self.ws.sheet_properties.pageSetUpPr.fitToPage = True
        self.ws.sheet_format.defaultRowHeight = ROW_HEIGHT
        widths = (34, 48, 18, 20, 20, 20, 22)
        if worksheet.title == "Товары":
            widths = (20, 48, 72, 18, 20, 20, 20, 22)
        elif worksheet.title == "Клиенты":
            widths = (*widths, 24, 28)
        elif worksheet.title == "RFM-анализ":
            widths = (34, 64, 36, 26, 26, 30, 26, 26, 26)
        elif worksheet.title == "Когортный анализ":
            widths = (34, 64, *(24 for _ in range(12)))
        for column, width in enumerate(widths, 1):
            self.ws.column_dimensions[get_column_letter(column)].width = width
        self.write(["KANZLER AI"], section=True)
        self.write([worksheet.title], section=True)
        self.row += 1


    def _cell(self, row, column, value, *, header=False, fmt=None):
        cell = self.ws.cell(row, column)
        if isinstance(value, report.ReportCell):
            value, fmt = value.value if value.kind != "empty" else value.text, value.number_format
        elif isinstance(value, report.ReportNumber):
            cell_data = report.cell_presentation(value)
            value, fmt = cell_data.value, cell_data.number_format
        elif isinstance(value, _Numeric):
            value, fmt = value.value, value.number_format
        cell.value = value
        if isinstance(value, str):
            if len(value) > 32767:
                raise ValueError("Значение превышает лимит ячейки XLSX (32767 символов).")
            cell.data_type = "s"  # Also protects =, +, -, @ labels and leading-zero SKUs.
            fmt = "@"
        cell.number_format = fmt or ("dd.mm.yyyy" if isinstance(value, date) else
                                    "#,##0.00" if isinstance(value, float) else "#,##0")
        cell.font = Font(name="Arial", size=10, bold=header, color=INK)
        cell.alignment = Alignment(vertical="center", wrap_text=True,
                                   horizontal="center" if header else "left" if isinstance(value, str) else "right")
        if header:
            cell.fill = PatternFill("solid", fgColor=HEADER_FILL)
        side = Side(style="thin", color=BORDER_COLOR)
        cell.border = Border(left=side, right=side, top=side, bottom=side)
        return cell

    def write(self, values, *, header=False, section=False, formats=None):
        self.check()
        height = ROW_HEIGHT
        for column, value in enumerate(values, 1):
            fmt = (formats or {}).get(column)
            if not header and column == 2 and isinstance(values[0], str) and "%" in values[0] and not isinstance(value, report.ReportCell):
                fmt = PERCENT_FORMAT
            cell = self._cell(self.row, column, value, header=header, fmt=fmt)
            if section:
                cell.font = Font(name="Arial", size=11, bold=True, color=INK)
                cell.fill = PatternFill("solid", fgColor=SECTION_FILL)
            value = cell.value
            width = self.ws.column_dimensions[get_column_letter(column)].width
            if value is not None:
                height = max(height, 13.5 * sum(max(1, math.ceil(len(line) / (width - 3)))
                                             for line in str(value).split("\n")) + 6)
        self.ws.row_dimensions[self.row].height = min(48 if header else MAX_ROW_HEIGHT, height)
        self.row += 1

    def heading(self, title):
        self.write([title], section=True)
        self.row += 1

    def table(self, headers, rows):
        header = self.row
        self.max_columns = max(self.max_columns, len(headers))
        self.write(headers, header=True)
        formats = {i: PERCENT_FORMAT for i, label in enumerate(headers, 1) if "%" in label}
        for values in rows:
            if tuple(headers) == CATEGORY_HEADERS and len(values[2].value if isinstance(values[2], report.ReportCell) else values[2]) > 32767:
                raise ValueError("Путь категории превышает лимит ячейки XLSX (32767 символов).")
            self.write(values, formats=formats)
        last = self.row - 1
        if last > header and len(headers) > 1:
            # Each independent block has its own filter, never one worksheet-wide AutoFilter.
            name = f"Stats{self.ws.parent.index(self.ws) + 1}Block{header}"
            table = Table(displayName=name, ref=f"A{header}:{get_column_letter(len(headers))}{last}")
            table.tableStyleInfo = TableStyleInfo(name="TableStyleLight1", showFirstColumn=False,
                showLastColumn=False, showRowStripes=False, showColumnStripes=False)
            self.ws.add_table(table)
        elif last == header:
            self.write(["Нет данных для таблицы."])
        self.row += 1

    def finish(self):
        self.ws.freeze_panes = None
        self.ws.print_title_rows = "1:2"
        self.ws.print_options.horizontalCentered = False
        wide = self.max_columns > 3
        self.ws.page_setup.orientation = "landscape" if wide else "portrait"
        self.ws.page_setup.paperSize = self.ws.PAPERSIZE_A3 if wide else self.ws.PAPERSIZE_A4
        self.ws.page_setup.fitToWidth = 1
        self.ws.page_setup.fitToHeight = 0
        self.ws.page_margins.left = self.ws.page_margins.right = .25
        self.ws.print_area = f"A1:{get_column_letter(self.max_columns)}{self.row}"
        self.ws.oddFooter.center.text = "Страница &P из &N"


    def cards(self, values):
        block = report.ReportBlock("cards", values=tuple(values))
        self.table(("Показатель", "Значение"), [(label, cell) for label, cell in report.kpi_cells(block)])

    def buyer_cards(self, values):
        self.cards(values)



def suggested_filename(result):
    return "Статистика_" + parse_timestamp(result["calculated_at"]).strftime("%Y-%m-%d_%H-%M-%S") + ".xlsx"


def write_statistics_xlsx(result, stream, *, check=lambda: None, workspace=None, snapshot_diagnostics=None, presentation=None):
    """The exact five GUI pages; no report-only analytics or chart duplicates."""
    model = presentation if presentation is not None else report.report_presentation(result,snapshot_diagnostics=snapshot_diagnostics)
    book = Workbook()
    book.remove(book.active)
    try:
        for page in model.pages:
            check()
            sheet = _Sheet(book.create_sheet(page.title),check)
            sheet.table(("Параметр отчёта", "Значение"), model.header)
            sheet.cards(model.global_cards)
            for group in page.groups:
                if page.key == "customers":
                    sheet.heading(group.title)
                for block in group.blocks:
                    check()
                    if block.kind == "heading":
                        sheet.heading(block.title)
                    elif block.kind == "cards":
                        sheet.table(("Показатель", "Значение"),
                                    [(label, cell) for label, cell in report.kpi_cells(block)])
                    elif block.kind == "table":
                        sheet.table(block.headers, report.table_cells(block))
                    elif block.kind == "text":
                        sheet.write((block.title,))
                    # Adjacent approved tables already contain all plotted values,
                    # including six RFM combinations and the numeric retention matrix.
            sheet.finish()
        check()
        if workspace is None:
            _save_report_book(book,stream,check)
        else:
            from Application.statistics_detail_xlsx import save_aggregate_book
            save_aggregate_book(book, stream, workspace)
        check()
    finally:
        book.close()


def _save_report_book(book, stream, check):
    """Aggregate sheets only: one in-memory XML buffer, no individual/temp XML."""
    from io import BytesIO
    from zipfile import ZipFile, ZIP_DEFLATED
    from openpyxl.writer.excel import ExcelWriter
    from openpyxl.worksheet._writer import WorksheetWriter
    from openpyxl.drawing.spreadsheet_drawing import SpreadsheetDrawing
    class ReportWriter(ExcelWriter):
        def write_worksheet(self, sheet):
            sheet._drawing=SpreadsheetDrawing()
            sheet._drawing.charts=sheet._charts
            sheet._drawing.images=sheet._images
            with BytesIO() as xml:
                writer=WorksheetWriter(sheet,out=xml)
                original=writer.write_row
                def row(*args,**kwargs):
                    check()
                    return original(*args,**kwargs)
                writer.write_row=row
                try:
                    writer.write()
                    sheet._rels=writer._rels
                    xml.seek(0)
                    with self._archive.open(sheet.path[1:],"w",force_zip64=True) as member:
                        while chunk:=xml.read(64*1024):
                            check()
                            member.write(chunk)
                    self.manifest.append(sheet)
                finally:
                    writer.close()
    with ZipFile(stream,"w",ZIP_DEFLATED,allowZip64=True) as archive:
        ReportWriter(book,archive).write_data()


def export_statistics(result, path):
    """Atomic public XLSX API; the stream renderer is also used by unified ZIP."""
    destination = Path(path)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=destination.parent, prefix=".statistics-", suffix=".xlsx", delete=False) as stream:
            temporary = Path(stream.name)
            write_statistics_xlsx(result, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
