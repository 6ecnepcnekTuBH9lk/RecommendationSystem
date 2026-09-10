"""Локальный CLI профилирования схемы уже скачанных экспортов; без API."""

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Mindbox raw JSON -> отчёт о схеме без значений данных")
    parser.add_argument("export", choices=("actions", "orders", "customers", "customer_merges", "all"))
    parser.add_argument("--input-dir", type=Path,
                        help="Каталог с part-файлами; для all — общий корень с подкаталогами четырёх типов")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "ВходныеДанные" / "MindboxReports")
    return parser


def main(argv: list[str] | None = None) -> int:
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from Application.mindbox.schema_profiler import (
        DEFAULT_RAW_ROOT, EXPORT_ROOTS, SchemaProfileError, profile_export, write_report,
    )

    args = build_parser().parse_args(argv)
    names = list(EXPORT_ROOTS) if args.export == "all" else [args.export]
    try:
        if args.input_dir is not None:
            source = args.input_dir.resolve()
            output = args.output_dir.resolve()
            if output == source or source in output.parents:
                raise SchemaProfileError("Отчёты должны сохраняться отдельно от входного каталога")
        # Сначала проверяем все входы: malformed JSON не оставляет частичного all-отчёта.
        reports = []
        for name in names:
            reports.append(profile_export(
                name, input_dir=args.input_dir if args.export != "all" else None,
                raw_root=(args.input_dir or DEFAULT_RAW_ROOT) if args.export == "all" else DEFAULT_RAW_ROOT,
            ))
        for report in reports:
            write_report(report, args.output_dir)
            print(f"{report['export_name']}: parts={report['file_count']}, objects={report['total_objects']}, "
                  f"paths={len(report['paths'])}; saved {report['export_name']}_schema.json / .md")
        return 0
    except SchemaProfileError as exc:
        print(f"Ошибка профилирования: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("Профилирование прервано пользователем.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
