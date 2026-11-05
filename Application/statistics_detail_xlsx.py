"""Streaming, pseudonymized XLSX members over the existing prepared detail rows."""
from contextlib import closing
from decimal import Decimal
import hashlib
import hmac
import os
import re

from zipfile import ZipFile, ZIP_DEFLATED
from openpyxl import Workbook
from openpyxl.writer.excel import ExcelWriter
from openpyxl.cell import WriteOnlyCell
from openpyxl.worksheet._writer import WorksheetWriter
from openpyxl.styles import Alignment, Font
from openpyxl.cell.cell import ILLEGAL_CHARACTERS_RE

from Application.customer_demographics import age_group
from Application.statistics_detail_contract import (
    CLIENT_COLUMNS, ORDER_COLUMNS, ORDER_LINE_COLUMNS, NULL_TOKEN, CSV_TEXT_PREFIX,
)
from Application.statistics_detail_storage import DetailExportError

EXCEL_MAX_ROWS = 1_048_576
DEFAULT_MAX_ROWS = 1_000_000  # Includes the repeated header; 999,999 data rows.
SAFE_CLIENT_COLUMNS = ("customer_id", "profile_present", "profile_status", "sex", "age_group", "as_of_date") + tuple(
    c for c in CLIENT_COLUMNS if c.startswith(("strict_", "extended_")))
SAFE_ORDER_COLUMNS = tuple(c for c in ORDER_COLUMNS if c not in {"source_customer_id", "retail_order_id"})
SAFE_LINE_COLUMNS = tuple(c for c in ORDER_LINE_COLUMNS if c != "source_customer_id")
SCHEMAS = {
    "clients": ("Клиенты", CLIENT_COLUMNS, SAFE_CLIENT_COLUMNS),
    "orders": ("Заказы", ORDER_COLUMNS, SAFE_ORDER_COLUMNS),
    "order_lines": ("Позиции заказов", ORDER_LINE_COLUMNS, SAFE_LINE_COLUMNS),
}


def decode_cell(value):
    """Reverse the existing lossless cell encoding without guessing ID types."""
    if value == NULL_TOKEN:
        return None
    if value.startswith(CSV_TEXT_PREFIX):
        return value[len(CSV_TEXT_PREFIX):]
    if re.fullmatch(r"[+-]?\d+(?:\.\d+)?", value):
        return Decimal(value)
    return value  # Exact ISO date/time strings, including UTC and subsecond precision.


class SafeProjection:
    def __init__(self):
        self.key = os.urandom(32)

    def alias(self, kind, value):
        if not isinstance(value, str) or not value:
            raise DetailExportError("BROKEN_DETAIL_RELATIONS")
        return kind + hmac.new(self.key, kind.encode()+b"\0"+value.encode("utf-8"), hashlib.sha256).hexdigest()

    def row(self, kind, encoded):
        _, source, target = SCHEMAS[kind]
        if len(encoded) != len(source):
            raise DetailExportError("XLSX_SCHEMA_MISMATCH")
        values = {name: decode_cell(value) for name, value in zip(source, encoded)}
        original_customer = values["customer_id"]
        values["customer_id"] = self.alias("C", original_customer)
        if kind == "clients":
            age = values.get("age")
            values["age_group"] = age_group(int(age)) if age is not None else "Возраст не определён"
            values["sex"] = ("Мужчины" if values.get("sex") == "male" else
                             "Женщины" if values.get("sex") == "female" else "Не указан")
        else:
            original_order = values["order_id"]
            values["order_id"] = self.alias("O", original_order)
            if kind == "order_lines":
                values["line_id"] = self.alias("L", original_order+"\0"+str(values["line_ordinal"])+"\0"+str(values["line_id"]))
        return tuple(values.get(name) for name in target)


def _money_column(name):
    return any(part in name for part in ("amount", "price", "monetary", "complete_m", "known_m"))


def excel_cell(sheet, value, name, *, header=False):
    if isinstance(value, Decimal):
        if not value.is_finite():
            raise DetailExportError("XLSX_NUMERIC_INVALID")
        # normalize() rounds under the ambient Decimal context; inspect digits directly.
        significant = len("".join(map(str, value.as_tuple().digits)).rstrip("0"))
        if significant > 15:
            if _money_column(name):
                # Never silently corrupt exact money at Excel's 15-digit boundary.
                raise DetailExportError("XLSX_MONEY_PRECISION")
            value = format(value, "f")  # Exact non-monetary ratios may exceed Excel precision.
    if isinstance(value, str):
        if len(value) > 32767:
            raise DetailExportError("XLSX_CELL_TOO_LONG")
        if ILLEGAL_CHARACTERS_RE.search(value):
            raise DetailExportError("XLSX_INVALID_TEXT")
    cell = WriteOnlyCell(sheet, value=value)
    if isinstance(value, str):
        cell.data_type = "s"  # Includes =,+,-,@ and leading-zero identifiers.
        cell.number_format = "@"
    elif isinstance(value, Decimal):
        places = len(format(value, "f").partition(".")[2])
        cell.number_format = "#,##0" + ("."+"0"*places if places else "")
    cell.alignment = Alignment(vertical="center", horizontal="left" if isinstance(value,str) else "right")
    if header:
        cell.font = Font(bold=True)
    return cell


class _OwnedWorksheetWriter(WorksheetWriter):
    def __init__(self, sheet, workspace):
        self.workspace = workspace
        self.path = workspace.create_state_file(prefix="xlsx-", suffix=".xml")
        super().__init__(sheet, out=str(self.path))

    def write_row(self, *args, **kwargs):
        self.workspace.control.check()
        return super().write_row(*args, **kwargs)

    def cleanup(self):
        self.workspace.release_state_file(self.path)


class CheckedStream:
    def __init__(self, stream, control):
        self.stream, self.control = stream, control

    def write(self, data):
        self.control.consume(len(data))
        return self.stream.write(data)

    def flush(self):
        self.stream.flush()

    def tell(self):
        return self.stream.tell()

    def seek(self, *args):
        return self.stream.seek(*args)


def write_detail_members(archive, kind, rows, workspace, projection, *, max_rows=DEFAULT_MAX_ROWS):
    if type(max_rows) is not int or not 2 <= max_rows <= EXCEL_MAX_ROWS:
        raise DetailExportError("INVALID_EXPORT_OPTIONS")
    title, _, columns = SCHEMAS[kind]
    count = part = in_part = 0
    book = sheet = None
    names = []

    def begin():
        nonlocal book, sheet, part, in_part
        part += 1
        in_part = 0
        book = Workbook(write_only=True)
        sheet = book.create_sheet(title)
        sheet.freeze_panes = "A2"
        sheet._writer = _OwnedWorksheetWriter(sheet, workspace)
        sheet._writer.write_top()
        sheet.append([excel_cell(sheet, c, c, header=True) for c in columns])

    def save():
        name = f"Данные покупателей/{title}_{part:02d}.xlsx"
        workspace.control.check()
        with archive.open(name, "w", force_zip64=True) as output:
            # openpyxl.save_workbook does not close its ZIP when write_data fails.
            with ZipFile(CheckedStream(output, workspace.control), "w", ZIP_DEFLATED, allowZip64=True) as nested:
                ExcelWriter(book, nested).write_data()
        names.append(name)
        book.close()
        workspace.control.check()

    try:
        begin()
        with closing(rows):
            for encoded in rows:
                workspace.control.check()
                if in_part == max_rows-1:
                    save()
                    begin()
                values = projection.row(kind, encoded)
                sheet.append([excel_cell(sheet, value, name) for name, value in zip(columns, values)])
                count += 1
                in_part += 1
                if count % 256 == 0:
                    workspace.control.stage(kind, count)
            save()
        return count, names
    finally:
        if sheet is not None and sheet._writer is not None and not sheet.closed:
            sheet.close()
        if sheet is not None and sheet._writer is not None and sheet._writer.path.exists():
            sheet._writer.cleanup()
        if book is not None:
            book.close()



def save_aggregate_book(book, stream, workspace):
    """Use owned D: worksheet XML for normal (styled aggregate) workbooks too."""
    from openpyxl.drawing.spreadsheet_drawing import SpreadsheetDrawing

    class PrivateExcelWriter(ExcelWriter):
        def write_worksheet(self, sheet):
            sheet._drawing = SpreadsheetDrawing()
            sheet._drawing.charts = sheet._charts
            sheet._drawing.images = sheet._images
            writer = _OwnedWorksheetWriter(sheet, workspace)
            try:
                writer.write()
                sheet._rels = writer._rels
                self._archive.write(writer.out, sheet.path[1:])
                self.manifest.append(sheet)
            finally:
                writer.close()
                writer.cleanup()

    with ZipFile(stream, "w", ZIP_DEFLATED, allowZip64=True) as nested:
        PrivateExcelWriter(book, nested).write_data()
