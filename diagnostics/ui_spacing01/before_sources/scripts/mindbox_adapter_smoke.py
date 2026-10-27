"""Локальная проверка адаптеров. Только counts; records не печатаются и не сохраняются."""

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main(argv: list[str] | None = None) -> int:
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from Application.mindbox.adapters import (
        AdapterError, adapt_action, adapt_order, adapt_customer, adapt_customer_merge,
    )
    from Application.mindbox.identity import CustomerIdResolver, CustomerIdentityError
    from Application.mindbox.adapters._common import identifier
    from Application.mindbox.raw_reader import DEFAULT_RAW_ROOT, RawExportError, iter_export

    parser = argparse.ArgumentParser(description="Mindbox adapters: только локальные raw exports и агрегированные counts")
    parser.add_argument("export", choices=("all", "actions", "orders", "customers", "customer_merges"))
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT,
                        help="Общий корень с подкаталогами четырёх экспортов")
    args = parser.parse_args(argv)
    counts = {"Customer merges": 0, "Merge aliases": 0, "Customers": 0, "Actions": 0,
              "Orders": 0, "Order lines": 0}
    canonical = {"Actions": 0, "Orders": 0, "Customers": 0}
    stage = "customer_merges"
    position = 0
    try:
        def merges():
            nonlocal position
            for position, raw in enumerate(iter_export("customer_merges", raw_root=args.raw_root), 1):
                record = adapt_customer_merge(raw)
                counts["Customer merges"] += 1
                yield record

        resolver = CustomerIdResolver(merges())
        counts["Merge aliases"] = resolver.alias_count
        for stage, label, adapter in (("customers", "Customers", adapt_customer),
                                      ("actions", "Actions", adapt_action), ("orders", "Orders", adapt_order)):
            if args.export not in ("all", stage):
                continue
            position = 0
            for position, raw in enumerate(iter_export(stage, raw_root=args.raw_root), 1):
                record = adapter(raw, resolver)
                counts[label] += 1
                if stage == "orders":
                    counts["Order lines"] += len(record)
                    # Считаем канонизированные ссылки по заказам, даже при пустых lines.
                    source = identifier(raw, "customer.ids.mindboxId")
                    canonical[label] += int(source != resolver.resolve(source))
                else:
                    canonical[label] += int(record.source_customer_id != record.customer_id)
        # При любой ошибке не публикуем частичные counts как успешный результат.
        for label, count in counts.items():
            print(f"{label}: {count}")
        print("Canonicalized customer references:")
        for label, count in canonical.items():
            print(f"{label}: {count}")
        return 0
    except (AdapterError, CustomerIdentityError, RawExportError) as exc:
        print(f"Ошибка: {stage}, объект #{position}: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("Проверка адаптеров прервана.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
