"""Local raw exports -> interactions -> catalog identity; safe aggregate output only."""

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def print_counts(label, counts):
    print(f"{label}:")
    print(f"  Interactions considered: {counts.interactions_total}")
    print(f"  Resolved: {counts.resolved}")
    print(f"  Unresolved: {counts.unresolved}")
    print(f"  Unsupported namespace: {counts.unsupported_namespace}")
    print(f"  Resolution rate: {counts.resolution_rate_percent:.2f}%")


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

    parser = argparse.ArgumentParser(description="Local product resolution; no API requests")
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG_PATH)
    parser.add_argument("--diagnose", action="store_true",
                        help="Continue after malformed actions and unresolved products; exit 1 if any")
    args = parser.parse_args(argv)
    stage = "catalog"
    try:
        catalog = load_catalog(args.catalog)
        products = ProductResolver(catalog)
        stage = "customer_merges"
        customers = CustomerIdResolver(
            adapt_customer_merge(raw) for raw in iter_export("customer_merges", raw_root=args.raw_root)
        )
        builder = InteractionBuilder()
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
                products.resolve_interaction(interaction, strict=not args.diagnose)
        stage = "orders"
        for raw in iter_export("orders", raw_root=args.raw_root):
            for line in adapt_order(raw, customers):
                interaction = builder.from_order_line(line)
                if interaction is not None:
                    products.resolve_interaction(interaction, strict=not args.diagnose)
        stats = products.diagnostics
        print(f"Catalog rows: {catalog.diagnostics.rows_total}")
        print(f"Catalog unique items: {catalog.diagnostics.unique_items}")
        print(f"Catalog empty code rows: {catalog.diagnostics.empty_code_rows}")
        print(f"Catalog duplicate code rows: {catalog.diagnostics.duplicate_code_rows}")
        print(f"Malformed upstream actions: {builder.diagnostics.actions_malformed}")
        print_counts("All interactions", stats.total)
        for kind, counts in stats.by_type.items():
            print_counts(kind.value, counts)
        print("Namespaces:")
        for namespace, counts in stats.by_namespace.items():
            print_counts(namespace, counts)
        print(f"Unique source product keys: {stats.unique_source_product_keys}")
        print(f"Unique resolved catalog items: {stats.unique_resolved_catalog_items}")
        print(f"Catalog items with multiple source keys: {stats.catalog_items_with_multiple_source_keys}")
        print(f"Max source keys per catalog item: {stats.max_source_keys_per_catalog_item}")
        if builder.diagnostics.actions_malformed or stats.total.unresolved:
            print("Diagnostic failure: malformed upstream actions or unresolved products; "
                  "not a complete downstream dataset.", file=sys.stderr)
            return 1
        return 0
    except (CatalogError, ProductResolutionError) as exc:
        print(f"Failure in {stage}: {exc}", file=sys.stderr)
        return 1
    except (RawExportError, AdapterError, CustomerIdentityError, InteractionBuildError) as exc:
        # Exception text may contain untrusted raw values in future upstream changes.
        print(f"Failure in {stage}: {type(exc).__name__}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("Product resolution interrupted.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
