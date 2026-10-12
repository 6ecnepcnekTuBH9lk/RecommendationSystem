"""Shared strict incremental JSON boundary; one export record at a time."""

from pathlib import Path
from decimal import InvalidOperation
import ijson
from ijson.common import ObjectBuilder

from .raw_reader import RawExportError, part_files


class _UniqueDict(dict):
    def __setitem__(self, key, value):
        if key in self:
            raise RawExportError("Duplicate raw JSON key", category="duplicate_key")
        super().__setitem__(key, value)


def iter_json_records(path, root_key):
    """Strict incremental envelope validation; exhaust to validate trailing JSON."""
    try:
        with path.open("rb") as stream:
            prefix = stream.read(4)
            if prefix.startswith((b'\xff\xfe', b'\xfe\xff', b'\x00\x00\xfe\xff')):
                raise RawExportError("Invalid or unreadable raw JSON part", category="encoding")
            stream.seek(3 if prefix.startswith(b"\xef\xbb\xbf") else 0)
            events = iter(ijson.basic_parse(stream, use_float=False, buf_size=65536))
            def expect(expected, category="json_structure"):
                if next(events) != expected:
                    raise RawExportError("Expected object with the selected export array", category=category)
            expect(("start_map", None), "root_object")
            expect(("map_key", root_key), "root_key")
            expect(("start_array", None), "root_array")
            for event, value in events:
                if event == "end_array":
                    break
                if event != "start_map":
                    raise RawExportError("Export array must contain objects", category="record_object")
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
                raise RawExportError("Incomplete Export array", category="incomplete_json")
            expect(("end_map", None))
            if next(events, None) is not None:
                raise RawExportError("Unexpected trailing raw JSON", category="trailing_json")
    except FileNotFoundError:
        raise RawExportError("Invalid or unreadable raw JSON part", category="file_missing") from None
    except PermissionError:
        raise RawExportError("Invalid or unreadable raw JSON part", category="file_access") from None
    except OSError:
        raise RawExportError("Invalid or unreadable raw JSON part", category="file_read") from None
    except UnicodeError:
        raise RawExportError("Invalid or unreadable raw JSON part", category="encoding") from None
    except StopIteration:
        raise RawExportError("Invalid or unreadable raw JSON part", category="incomplete_json") from None
    except ijson.JSONError as exc:
        # Interpret only known parser headers; never expose its payload excerpts.
        argument = exc.args[0] if exc.args else ''
        header = (argument.split(b'\n', 1)[0].decode('ascii', errors='replace') if isinstance(argument, bytes)
                  else argument.split('\n', 1)[0] if isinstance(argument, str) else '')
        if header.startswith(("b'", 'b"')):
            # Some C-backend versions wrap an undecodable diagnostic in bytes repr.
            header = header[2:].split(r'\n', 1)[0]
        category = ("incomplete_json" if header.startswith("parse error: premature EOF") else
                    "trailing_json" if header.startswith("parse error: trailing garbage") else
                    "encoding" if header.startswith("lexical error: invalid bytes in UTF8 string") else
                    "json_structure")
        raise RawExportError("Invalid or unreadable raw JSON part", category=category) from None
    except (ValueError, OverflowError, InvalidOperation):
        raise RawExportError("Invalid or unreadable raw JSON part") from None


def iter_customers_stream(*, input_dir):
    for path in part_files(Path(input_dir), "customers"):
        yield from iter_json_records(path, "customers")
