"""Dedicated incremental Customers JSON boundary; generic raw readers unchanged."""

from pathlib import Path
from decimal import InvalidOperation
import ijson
from ijson.common import ObjectBuilder

from .raw_reader import RawExportError, part_files


class _UniqueDict(dict):
    def __setitem__(self, key, value):
        if key in self:
            raise RawExportError("Duplicate Customers JSON key")
        super().__setitem__(key, value)


def iter_customers_stream(*, input_dir):
    """One raw customer at a time, int/Decimal numbers, strict envelope/duplicates.

    The parser buffers 64 KiB; ObjectBuilder owns only the current customer, never
    the root array. Callers must exhaust the iterator to validate the trailing JSON.
    """
    for path in part_files(Path(input_dir), "customers"):
        try:
            with path.open("rb") as stream:
                if stream.read(3) != b"\xef\xbb\xbf":
                    stream.seek(0)
                events = iter(ijson.basic_parse(stream, use_float=False, buf_size=65536))
                def expect(expected):
                    if next(events) != expected:
                        raise RawExportError("Expected Customers object with a customers array")
                expect(("start_map", None))
                expect(("map_key", "customers"))
                expect(("start_array", None))
                for event, value in events:
                    if event == "end_array":
                        break
                    if event != "start_map":
                        raise RawExportError("Customers array must contain objects")
                    builder = ObjectBuilder(map_type=_UniqueDict)
                    builder.event(event, value)
                    depth = 1
                    while depth:
                        event, value = next(events)
                        builder.event(event, value)
                        depth += int(event in ("start_map", "start_array")) - int(event in ("end_map", "end_array"))
                    yield builder.value
                    del builder
                else:
                    raise RawExportError("Incomplete Customers array")
                expect(("end_map", None))
                if next(events, None) is not None:
                    raise RawExportError("Unexpected trailing Customers JSON")
        except (ijson.JSONError, StopIteration, OSError, UnicodeError, ValueError, OverflowError, InvalidOperation):
            raise RawExportError("Invalid or unreadable Customers JSON part") from None
