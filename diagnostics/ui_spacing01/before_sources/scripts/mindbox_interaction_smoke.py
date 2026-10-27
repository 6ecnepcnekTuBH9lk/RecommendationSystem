"""Локальный raw -> adapters -> interactions. Вывод только агрегированных counts."""

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main(argv: list[str] | None = None) -> int:
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from Application.interactions import InteractionBuilder, InteractionBuildError
    from Application.mindbox.adapters import AdapterError, adapt_action, adapt_order, adapt_customer_merge
    from Application.mindbox.identity import CustomerIdResolver, CustomerIdentityError
    from Application.mindbox.raw_reader import DEFAULT_RAW_ROOT, RawExportError, iter_export

    parser = argparse.ArgumentParser(description="Mindbox interaction smoke: только локальные данные и counts")
    parser.add_argument("export", choices=("all",))
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT,
                        help="Общий корень опубликованных raw exports")
    parser.add_argument("--diagnose", action="store_true",
                        help="Продолжить после malformed business events, вывести counts и вернуть код 1")
    args = parser.parse_args(argv)
    stage = "customer_merges"
    position = 0
    orders = 0
    try:
        def merges():
            nonlocal position
            for position, raw in enumerate(iter_export("customer_merges", raw_root=args.raw_root), 1):
                yield adapt_customer_merge(raw)

        resolver = CustomerIdResolver(merges())
        builder = InteractionBuilder()

        def actions():
            nonlocal stage, position
            stage, position = "actions", 0
            for position, raw in enumerate(iter_export("actions", raw_root=args.raw_root), 1):
                yield adapt_action(raw, resolver)

        def order_lines():
            nonlocal stage, position, orders
            stage, position = "orders", 0
            for position, raw in enumerate(iter_export("orders", raw_root=args.raw_root), 1):
                records = adapt_order(raw, resolver)
                orders += 1
                yield from records

        for action in actions():
            try:
                builder.from_action(action)
            except InteractionBuildError:
                if not args.diagnose:
                    raise
                # Builder уже записал malformed count. Итог будет явно неуспешным.
        for line in order_lines():
            builder.from_order_line(line)
        stats = builder.diagnostics
        # Только после полного прохода. Никаких статусов, IDs, PII или raw-значений.
        for label, count in (
            ("Actions total", stats.actions_total), ("View actions", stats.actions_view),
            ("View interactions", stats.view_interactions), ("Favorite actions", stats.actions_favorite),
            ("Favorite interactions", stats.favorite_interactions), ("Unmapped actions", stats.actions_unmapped),
            ("Malformed actions", stats.actions_malformed),
            ("Orders", orders), ("Order lines", stats.order_lines_total),
            ("Purchase lines", stats.order_lines_purchase),
            ("Filtered order lines", stats.order_lines_filtered_by_status),
        ):
            print(f"{label}: {count}")
        print("Interactions:")
        print(f"VIEW: {stats.view_interactions}")
        print(f"FAVORITE: {stats.favorite_interactions}")
        print(f"PURCHASE: {stats.purchase_interactions}")
        print(f"TOTAL: {stats.total_interactions}")
        if stats.actions_malformed:
            print("Диагностический проход выявил malformed actions; interactions включают только корректные события.",
                  file=sys.stderr)
            return 1
        return 0
    except (RawExportError, AdapterError, CustomerIdentityError, InteractionBuildError) as exc:
        print(f"Ошибка: {stage}, объект #{position}: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("Проверка interactions прервана.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
