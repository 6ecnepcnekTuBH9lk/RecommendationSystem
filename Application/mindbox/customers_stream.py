"""Shared strict incremental JSON boundary; one export record at a time."""

from pathlib import Path
from decimal import InvalidOperation
import ijson
from ijson.common import ObjectBuilder

from .raw_reader import RawExportError, part_files


class _UniqueDict(dict):
    def __setitem__(self, key, value):
        if key in self:
            raise RawExportError("Duplicate raw JSON key")
        super().__setitem__(key, value)


def iter_json_records(path, root_key):
    """Strict incremental envelope validation; exhaust to validate trailing JSON."""
    try:
        with path.open("rb") as stream:
            if stream.read(3) != b"\xef\xbb\xbf":
                stream.seek(0)
            events = iter(ijson.basic_parse(stream, use_float=False, buf_size=65536))
            def expect(expected):
                if next(events) != expected:
                    raise RawExportError("Expected object with the selected export array")
            expect(("start_map", None))
            expect(("map_key", root_key))
            expect(("start_array", None))
            for event, value in events:
                if event == "end_array":
                    break
                if event != "start_map":
                    raise RawExportError("Export array must contain objects")
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
                raise RawExportError("Incomplete Export array")
            expect(("end_map", None))
            if next(events, None) is not None:
                raise RawExportError("Unexpected trailing raw JSON")
    except (ijson.JSONError, StopIteration, OSError, UnicodeError, ValueError, OverflowError, InvalidOperation):
        raise RawExportError("Invalid or unreadable raw JSON part") from None


def iter_customers_stream(*, input_dir):
    for path in part_files(Path(input_dir), "customers"):
        yield from iter_json_records(path, "customers")
