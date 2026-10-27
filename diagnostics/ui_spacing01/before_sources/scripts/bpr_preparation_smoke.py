"""Local diagnostic BPR preparation. No training, API or dataset output."""

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main(argv: list[str] | None = None) -> int:
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from Application.interactions import InteractionBuilder, InteractionBuildError
    from Application.mindbox.adapters import AdapterError, adapt_action, adapt_order, adapt_customer_merge
    from Application.mindbox.identity import CustomerIdResolver, CustomerIdentityError
    from Application.mindbox.raw_reader import DEFAULT_RAW_ROOT, RawExportError, iter_export
    from Application.product_resolution import (
        DEFAULT_CATALOG_PATH, CatalogError, ProductResolver, ProductResolutionError, load_catalog,
    )
    from Application.model.bpr_preparation import (
        BprPreparationConfig, BprPreparationError, DateMode, prepare_bpr, to_bpr_event,
    )
    from Application.model.training_data import PreparedDataError, validate_prepared_data

    parser = argparse.ArgumentParser(description="Local BPR preparation aggregates only")
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG_PATH)
    parser.add_argument("--diagnose", action="store_true")
    parser.add_argument("--date-mode", choices=[mode.value for mode in DateMode], default=DateMode.LEGACY_DATE.value)
    args = parser.parse_args(argv)
    stage = "catalog"
    try:
        products = ProductResolver(load_catalog(args.catalog))
        stage = "customer_merges"
        customers = CustomerIdResolver(
            adapt_customer_merge(raw) for raw in iter_export("customer_merges", raw_root=args.raw_root)
        )
        builder = InteractionBuilder()
        config = BprPreparationConfig(date_mode=DateMode(args.date_mode))

        def events():
            nonlocal stage
            stage = "actions"
            for raw in iter_export("actions", raw_root=args.raw_root):
                action = adapt_action(raw, customers)
                try:
                    interactions = builder.from_action(action)
                except InteractionBuildError:
                    if not args.diagnose:
                        raise
                    continue
                for interaction in interactions:
                    resolved = products.resolve_interaction(interaction, strict=not args.diagnose)
                    if resolved is not None:
                        yield to_bpr_event(resolved, config.weights)
            stage = "orders"
            for raw in iter_export("orders", raw_root=args.raw_root):
                for line in adapt_order(raw, customers):
                    interaction = builder.from_order_line(line)
                    if interaction is not None:
                        resolved = products.resolve_interaction(interaction, strict=not args.diagnose)
                        if resolved is not None:
                            yield to_bpr_event(resolved, config.weights)

        result = prepare_bpr(events(), config)
        validate_prepared_data(result)
        stats = result.diagnostics
        print("Prepared input validation: OK")
        print(f"Date mode: {config.date_mode.value}")
        print(f"Resolved interactions: {products.diagnostics.total.resolved}")
        print(f"Unresolved products: {products.diagnostics.total.unresolved}")
        print(f"Unsupported namespace: {products.diagnostics.total.unsupported_namespace}")
        print("BPR events:")
        print(f"VIEW: {stats.view_events}")
        print(f"FAVORITE: {stats.favorite_events}")
        print(f"PURCHASE: {stats.purchase_events}")
        print(f"TOTAL: {stats.events_total}")
        print(f"Unique users: {stats.unique_users}")
        print(f"Unique items: {stats.unique_items}")
        print(f"Eval eligible users: {stats.eligible_eval_users}")
        print(f"Eval events: {stats.eval_events}")
        print(f"Train events before aggregation: {stats.train_events_before_aggregation}")
        print(f"Train user-item pairs after aggregation: {stats.train_pairs_after_aggregation}")
        print(f"Total train weight: {stats.total_train_weight:.12g}")
        print(f"Upstream malformed actions: {builder.diagnostics.actions_malformed}")
        if builder.diagnostics.actions_malformed or products.diagnostics.total.unresolved:
            print("Diagnostic failure: upstream malformed or unresolved; statistics cover valid events only.",
                  file=sys.stderr)
            return 1
        return 0
    except (CatalogError, ProductResolutionError, BprPreparationError, PreparedDataError, RawExportError, AdapterError,
            CustomerIdentityError, InteractionBuildError) as exc:
        print(f"Failure in {stage}: {type(exc).__name__}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("BPR preparation interrupted.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
