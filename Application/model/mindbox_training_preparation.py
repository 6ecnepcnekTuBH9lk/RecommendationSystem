"""Explicit local exports -> validated training input. No training or API calls."""

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from Application.interactions import InteractionBuilder, InteractionBuildError
from Application.mindbox.adapters import adapt_action, adapt_customer_merge, adapt_order
from Application.mindbox.identity import CustomerIdResolver
from Application.mindbox.raw_reader import iter_export
from Application.product_resolution import ProductResolver, ProductResolutionDiagnostics, load_catalog
from Application.model.bpr_preparation import (
    BprDiagnostics, BprPreparationConfig, BprWeightConfig, DateMode, prepare_bpr, to_bpr_event,
)
from Application.model.training_data import PreparedBprData, validate_prepared_data


class PreparationTrainConfig(Protocol):
    w_view_item: float
    w_favorite: float
    w_purchase: float
    min_user_interactions_for_eval: int


@dataclass(frozen=True)
class MindboxPreparationDiagnostics:
    customer_merges: int
    actions: int
    order_lines: int
    malformed_actions: int
    unmapped_actions: int
    view_interactions: int
    favorite_interactions: int
    purchase_interactions: int
    resolution: ProductResolutionDiagnostics
    bpr: BprDiagnostics


@dataclass(frozen=True, repr=False)
class MindboxPreparationResult:
    # Frozen wrapper, shared prepared arrays retain M02-07 controlled mutability.
    prepared_data: PreparedBprData
    diagnostics: MindboxPreparationDiagnostics
    complete: bool


def prepare_training_data_from_mindbox(
    *, actions_export_dir: str | Path, orders_export_dir: str | Path,
    customer_merges_export_dir: str | Path, catalog_path: str | Path,
    train_config: PreparationTrainConfig, diagnose: bool = False,
) -> MindboxPreparationResult:
    for directory in (actions_export_dir, orders_export_dir, customer_merges_export_dir, catalog_path):
        if not isinstance(directory, (str, Path)) or not str(directory).strip():
            raise ValueError("Explicit export directories and catalog path are required")
    if type(diagnose) is not bool:
        raise ValueError("diagnose must be boolean")
    config = BprPreparationConfig(
        weights=BprWeightConfig(view_weight=train_config.w_view_item, favorite_weight=train_config.w_favorite,
                                purchase_weight=train_config.w_purchase),
        min_user_interactions_for_eval=train_config.min_user_interactions_for_eval,
        date_mode=DateMode.LEGACY_DATE,
    )
    products = ProductResolver(load_catalog(Path(catalog_path)))
    merge_count = 0

    def merges():
        nonlocal merge_count
        for raw in iter_export("customer_merges", input_dir=customer_merges_export_dir):
            merge_count += 1
            yield adapt_customer_merge(raw)

    customers = CustomerIdResolver(merges())
    builder = InteractionBuilder()

    def events():
        for raw in iter_export("actions", input_dir=actions_export_dir):
            action = adapt_action(raw, customers)
            try:
                interactions = builder.from_action(action)
            except InteractionBuildError:
                if not diagnose:
                    raise
                continue
            for interaction in interactions:
                resolved = products.resolve_interaction(interaction, strict=not diagnose)
                if resolved is not None:
                    yield to_bpr_event(resolved, config.weights)
        for raw in iter_export("orders", input_dir=orders_export_dir):
            for line in adapt_order(raw, customers):
                interaction = builder.from_order_line(line)
                if interaction is not None:
                    resolved = products.resolve_interaction(interaction, strict=not diagnose)
                    if resolved is not None:
                        yield to_bpr_event(resolved, config.weights)

    prepared = prepare_bpr(events(), config)
    validate_prepared_data(prepared)
    interactions, resolution = builder.diagnostics, products.diagnostics
    diagnostics = MindboxPreparationDiagnostics(
        merge_count, interactions.actions_total, interactions.order_lines_total,
        interactions.actions_malformed, interactions.actions_unmapped,
        interactions.view_interactions, interactions.favorite_interactions, interactions.purchase_interactions,
        resolution, prepared.diagnostics,
    )
    return MindboxPreparationResult(prepared, diagnostics,
                                    not (interactions.actions_malformed or resolution.total.unresolved))
