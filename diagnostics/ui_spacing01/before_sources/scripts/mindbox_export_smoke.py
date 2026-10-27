"""Ручной smoke-test. Реальный API вызывается только из main()."""

import argparse
import json
import math
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Mindbox -> raw JSON, без запуска PyQt")
    parser.add_argument("export", choices=("actions", "orders", "customers", "customer_merges"))
    parser.add_argument("--env-file", type=Path, default=PROJECT_ROOT / ".env")
    parser.add_argument("--payload-file", type=Path, help="JSON-объект параметров выбранной операции")
    parser.add_argument("--since", help="Для actions: начало UTC, YYYY-MM-DD HH:MM")
    parser.add_argument("--until", help="Для actions: конец UTC (не включительно), YYYY-MM-DD HH:MM")
    parser.add_argument("--poll-interval", type=float, default=5.0)
    parser.add_argument("--timeout", type=float, default=600.0, help="Ожидание Ready в секундах")
    return parser


def build_payload(args: argparse.Namespace) -> dict:
    if args.payload_file is not None:
        if args.since or args.until:
            raise ValueError("Используйте либо --payload-file, либо --since/--until")
        try:
            payload = json.loads(args.payload_file.read_text(encoding="utf-8-sig"))
        except (OSError, ValueError):
            raise ValueError("Не удалось прочитать JSON из --payload-file") from None
        if not isinstance(payload, dict):
            raise ValueError("--payload-file должен содержать JSON-объект")
        return payload
    if args.export != "actions":
        if args.since or args.until:
            raise ValueError("--since/--until предназначены только для actions")
        return {}
    if bool(args.since) != bool(args.until):
        raise ValueError("Задайте одновременно --since и --until")
    if args.since:
        try:
            since = datetime.strptime(args.since, "%Y-%m-%d %H:%M").replace(tzinfo=timezone.utc)
            until = datetime.strptime(args.until, "%Y-%m-%d %H:%M").replace(tzinfo=timezone.utc)
        except ValueError:
            raise ValueError("Даты должны иметь формат UTC YYYY-MM-DD HH:MM") from None
        if since >= until:
            raise ValueError("--since должен быть раньше --until")
    else:
        until = datetime.now(timezone.utc).replace(second=0, microsecond=0)
        since = until - timedelta(hours=1)
    return {
        "sinceDateTimeUtc": since.strftime("%Y-%m-%d %H:%M"),
        "tillDateTimeUtc": until.strftime("%Y-%m-%d %H:%M"),
    }


def main(argv: list[str] | None = None) -> int:
    # Поддерживаем python scripts/mindbox_export_smoke.py из любой рабочей папки.
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from Application.mindbox import MindboxClient, MindboxConfig, MindboxError

    args = build_parser().parse_args(argv)
    try:
        payload = build_payload(args)
        if any(not math.isfinite(value) or value <= 0 for value in (args.poll_interval, args.timeout)):
            raise ValueError("--poll-interval и --timeout должны быть конечными положительными числами")
        config = MindboxConfig.from_env(args.env_file)
        operation = config.operations[args.export]
        with MindboxClient(config) as client:
            print(f"Mindbox export: {args.export}")
            print("Operation: " + client.safe_text(operation))
            export_id = client.start_export(operation, payload)
            print("Export started: " + client.safe_text(export_id))
            print("Waiting for export...")
            urls = client.wait_for_export(
                operation, export_id, poll_interval=args.poll_interval, timeout=args.timeout,
            )
            print(f"Export ready. Parts: {len(urls)}")
            paths = client.download_export(args.export, urls)
            print("Saved:")
            for path in paths:
                print(client.safe_text(path))
        return 0
    except (MindboxError, ValueError) as exc:
        print(f"Ошибка: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("Выгрузка прервана пользователем.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
