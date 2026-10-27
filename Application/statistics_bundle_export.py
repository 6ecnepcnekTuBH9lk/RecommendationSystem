"""Qt-free, atomic publication of the two reports of one captured snapshot."""

import os
from pathlib import Path
import tempfile
from zipfile import ZIP_DEFLATED, ZipFile

from Application.statistics_export import export_statistics, suggested_filename as xlsx_filename
from Application.statistics_html_export import export_statistics_html


def suggested_filename(result):
    return Path(xlsx_filename(result)).with_suffix('.zip').name


def export_statistics_bundle(result, path):
    """Publish only a complete, CRC-checked bundle; preserve an existing destination."""
    destination = Path(path)
    stem = Path(xlsx_filename(result)).stem
    names = (stem + '.xlsx', stem + '.html')
    # A private, unique directory on the destination volume owns all intermediate files.
    with tempfile.TemporaryDirectory(dir=destination.parent, prefix='.statistics-bundle-') as directory:
        working = Path(directory)
        xlsx, html = (working / name for name in names)
        export_statistics(result, xlsx)
        export_statistics_html(result, html)
        if not xlsx.is_file() or not html.is_file() or not xlsx.stat().st_size or not html.stat().st_size:
            raise OSError('Не удалось сформировать оба отчёта статистики.')
        temporary = working / 'bundle.zip'
        with ZipFile(temporary, 'w', compression=ZIP_DEFLATED) as archive:
            for source in (xlsx, html):
                archive.write(source, arcname=source.name)
        with ZipFile(temporary) as archive:
            if archive.namelist() != list(names) or archive.testzip() is not None:
                raise OSError('Архив статистики не прошёл проверку целостности.')
        with temporary.open('r+b') as stream:
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
