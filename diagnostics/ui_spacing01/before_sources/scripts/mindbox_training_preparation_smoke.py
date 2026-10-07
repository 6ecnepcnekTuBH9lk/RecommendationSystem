"""Convenience latest selection only here; library always receives explicit dirs."""

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main(argv: list[str] | None = None) -> int:
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from Application.model.BPRMF import TrainConfig
    from Application.model.mindbox_training_preparation import prepare_training_data_from_mindbox
    from Application.model.training_data import PreparedDataError
    from Application.model.bpr_preparation import BprPreparationError
    from Application.mindbox.raw_reader import DEFAULT_RAW_ROOT, RawExportError, select_export_directory
    from Application.mindbox.adapters import AdapterError
    from Application.mindbox.identity import CustomerIdentityError
    from Application.interactions import InteractionBuildError
    from Application.product_resolution import DEFAULT_CATALOG_PATH, CatalogError, ProductResolutionError

    parser = argparse.ArgumentParser(description="Local Mindbox preparation and validation; no training")
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--actions-export-dir", type=Path)
    parser.add_argument("--orders-export-dir", type=Path)
    parser.add_argument("--customer-merges-export-dir", type=Path)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG_PATH)
    parser.add_argument("--diagnose", action="store_true")
    args = parser.parse_args(argv)
    try:
        directories = {name: select_export_directory(name, raw_root=args.raw_root, input_dir=directory)
                       for name, directory in (("actions", args.actions_export_dir),
                                               ("orders", args.orders_export_dir),
                                               ("customer_merges", args.customer_merges_export_dir))}
        result = prepare_training_data_from_mindbox(
            actions_export_dir=directories["actions"], orders_export_dir=directories["orders"],
            customer_merges_export_dir=directories["customer_merges"], catalog_path=args.catalog,
            train_config=TrainConfig(), diagnose=args.diagnose,
        )
        stats = result.diagnostics
        print("Prepared input validation: OK")
        for label, count in (
            ("Customer merges", stats.customer_merges), ("Actions", stats.actions), ("Order lines", stats.order_lines),
            ("Malformed actions", stats.malformed_actions), ("Unmapped actions", stats.unmapped_actions),
            ("VIEW", stats.view_interactions), ("FAVORITE", stats.favorite_interactions),
            ("PURCHASE", stats.purchase_interactions), ("Resolved", stats.resolution.total.resolved),
            ("Unresolved", stats.resolution.total.unresolved),
            ("Unsupported namespace", stats.resolution.total.unsupported_namespace),
            ("BPR events", stats.bpr.events_total), ("Users", stats.bpr.unique_users), ("Items", stats.bpr.unique_items),
            ("Eligible eval users", stats.bpr.eligible_eval_users), ("Eval events", stats.bpr.eval_events),
            ("Train events before aggregation", stats.bpr.train_events_before_aggregation),
            ("Train pairs", stats.bpr.train_pairs_after_aggregation),
        ):
            print(f"{label}: {count}")
        print(f"Total train weight: {stats.bpr.total_train_weight:.12g}")
        print(f"Complete: {result.complete}")
        if not result.complete:
            print("Incomplete diagnostic result; not a successful production dataset.", file=sys.stderr)
        return 0 if result.complete else 1
    except (RawExportError, AdapterError, CustomerIdentityError, InteractionBuildError, CatalogError,
            ProductResolutionError, PreparedDataError, BprPreparationError) as exc:
        print(f"Preparation failed: {type(exc).__name__}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
