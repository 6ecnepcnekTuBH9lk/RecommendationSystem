"""Explicit local exports -> validated training input. No training or API calls."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Protocol
from .interaction_analytics import AnalyticsCollector, load_catalog_kinds

from Application.interactions import InteractionBuilder, InteractionBuildError, classify_action_system_name
from Application.mindbox.adapters import adapt_action, adapt_action_system_name, adapt_customer_merge, adapt_order
from Application.mindbox.adapters._common import timestamp
from Application.mindbox.identity import CustomerIdResolver
from Application.mindbox.selection import DEFAULT_SELECTION, MindboxSelectionConfig
from Application.mindbox.raw_reader import iter_export
from Application.mindbox.order_dedup import OrderSnapshots
from Application.mindbox.training_batch import TrainingBatchWindow
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
    malformed_action_system_names: Mapping[str, int] = field(default_factory=dict)
    orders: int = 0
    actions_view: int = 0
    actions_favorite: int = 0
    unmapped_action_system_names: Mapping[str, int] = field(default_factory=dict)
    orders_raw: int = 0
    orders_unique: int = 0
    orders_duplicate_identical: int = 0
    orders_duplicate_conflicting: int = 0
    order_snapshot_policy: str = "first_source_order"
    orders_in_period: int = 0
    actions_before_start: int = 0
    actions_at_or_after_end: int = 0
    orders_before_start: int = 0
    orders_at_or_after_end: int = 0

    def __post_init__(self):
        object.__setattr__(self, "malformed_action_system_names",
                           MappingProxyType(dict(self.malformed_action_system_names)))
        object.__setattr__(self, "unmapped_action_system_names",
                           MappingProxyType(dict(self.unmapped_action_system_names)))


@dataclass(frozen=True, repr=False)
class MindboxPreparationResult:
    # Frozen wrapper, shared prepared arrays retain M02-07 controlled mutability.
    prepared_data: PreparedBprData
    diagnostics: MindboxPreparationDiagnostics
    complete: bool

    @property
    def analytics(self):
        return self.prepared_data.analytics


def prepare_training_data_from_mindbox(
    *, actions_export_dir: str | Path, orders_export_dir: str | Path,
    customer_merges_export_dir: str | Path, catalog_path: str | Path,
    train_config: PreparationTrainConfig, diagnose: bool = False,
    selection: MindboxSelectionConfig = DEFAULT_SELECTION,
) -> MindboxPreparationResult:
    return _prepare_training_data_from_mindbox_sources(
        actions_export_dirs=(actions_export_dir,), orders_export_dirs=(orders_export_dir,),
        customer_merges_export_dir=customer_merges_export_dir, catalog_path=catalog_path,
        train_config=train_config, diagnose=diagnose, selection=selection)


def _prepare_training_data_from_mindbox_sources(
    *, actions_export_dirs, orders_export_dirs, customer_merges_export_dir,
    catalog_path, train_config, diagnose=False, selection: MindboxSelectionConfig = DEFAULT_SELECTION,
    interaction_window: TrainingBatchWindow | None = None,
) -> MindboxPreparationResult:
    if not actions_export_dirs or not orders_export_dirs:
        raise ValueError("Explicit export directory sequences required")
    for directories in (actions_export_dirs, orders_export_dirs):
        if not isinstance(directories, (tuple, list)):
            raise ValueError("Explicit export directory sequences required")
    for directory in (*actions_export_dirs, *orders_export_dirs, customer_merges_export_dir, catalog_path):
        if not isinstance(directory, (str, Path)) or not str(directory).strip():
            raise ValueError("Explicit export directories and catalog path are required")
    if type(diagnose) is not bool:
        raise ValueError("diagnose must be boolean")
    if interaction_window is not None and not isinstance(interaction_window, TrainingBatchWindow):
        raise ValueError("Validated interaction window required")
    config = BprPreparationConfig(
        weights=BprWeightConfig(view_weight=train_config.w_view_item, favorite_weight=train_config.w_favorite,
                                purchase_weight=train_config.w_purchase),
        min_user_interactions_for_eval=train_config.min_user_interactions_for_eval,
        # Canonical records retain exact UTC event times; legacy CSV keeps its own date mode.
        date_mode=DateMode.FULL_TIMESTAMP,
    )
    products = ProductResolver(load_catalog(Path(catalog_path)))
    merge_count = 0
    order_count = 0
    snapshots = OrderSnapshots()
    orders_raw = 0
    excluded = {"actions_before_start": 0, "actions_at_or_after_end": 0,
                "orders_before_start": 0, "orders_at_or_after_end": 0}

    def in_window(stamp, source):
        if interaction_window is None:
            return True
        if stamp < interaction_window.interaction_since:
            excluded[source + "_before_start"] += 1
            return False
        if stamp >= interaction_window.interaction_until:
            excluded[source + "_at_or_after_end"] += 1
            return False
        return True

    def merges():
        nonlocal merge_count
        for raw in iter_export("customer_merges", input_dir=customer_merges_export_dir):
            merge_count += 1
            yield adapt_customer_merge(raw)

    customers = CustomerIdResolver(merges())
    builder = InteractionBuilder(selection.interaction_rules())
    analytics = AnalyticsCollector()

    def collect(resolved):
        interaction = resolved.interaction
        analytics.add(interaction.customer_id, resolved.item_id, interaction.interaction_type,
                      interaction.event_datetime_utc, interaction.quantity)
        return to_bpr_event(resolved, config.weights)

    def sources(name, directories):
        for directory in directories:
            yield from iter_export(name, input_dir=directory)

    def events():
        nonlocal order_count, orders_raw
        for raw in sources("actions", actions_export_dirs):
            system_name = adapt_action_system_name(raw)
            if classify_action_system_name(system_name, builder.rules) is None:
                builder.record_unmapped_action(system_name)
                continue
            action = adapt_action(raw, customers, product_namespaces=selection.action_product_namespaces)
            if not in_window(action.event_datetime_utc, "actions"):
                continue
            try:
                interactions = builder.from_action(action)
            except InteractionBuildError:
                if not diagnose:
                    raise
                continue
            for interaction in interactions:
                resolved = products.resolve_interaction(interaction, strict=not diagnose)
                if resolved is not None:
                    yield collect(resolved)
        for raw in sources("orders", orders_export_dirs):
            orders_raw += 1
            if not in_window(timestamp(raw, "firstAction.dateTimeUtc"), "orders"):
                continue
            # Validate changed repeats too: a conflict cannot hide malformed raw.
            lines = adapt_order(raw, customers, product_namespaces=selection.order_product_namespaces)
            if not snapshots.accept(raw, diagnose=diagnose):
                continue
            order_count += 1
            for line in lines:
                interaction = builder.from_order_line(line)
                if interaction is not None:
                    resolved = products.resolve_interaction(interaction, strict=not diagnose)
                    if resolved is not None:
                        yield collect(resolved)

    prepared = prepare_bpr(events(), config)
    validate_prepared_data(prepared)
    prepared.analytics = analytics.finalize(prepared.mappings, load_catalog_kinds(catalog_path))
    interactions, resolution = builder.diagnostics, products.diagnostics
    diagnostics = MindboxPreparationDiagnostics(
        merge_count, interactions.actions_total, interactions.order_lines_total,
        interactions.actions_malformed, interactions.actions_unmapped,
        interactions.view_interactions, interactions.favorite_interactions, interactions.purchase_interactions,
        resolution, prepared.diagnostics,
        malformed_action_system_names=interactions.malformed_action_system_names,
        orders=order_count, orders_raw=orders_raw, orders_unique=len(snapshots.fingerprints),
        orders_duplicate_identical=snapshots.identical, orders_duplicate_conflicting=snapshots.conflicting,
        actions_view=interactions.actions_view,
        actions_favorite=interactions.actions_favorite,
        unmapped_action_system_names=interactions.unmapped_action_system_names,
        order_snapshot_policy="first_in_period_source_order" if interaction_window is not None else "first_source_order",
        orders_in_period=snapshots.raw, **excluded,
    )
    return MindboxPreparationResult(prepared, diagnostics,
                                    not (interactions.actions_malformed or resolution.total.unresolved or snapshots.conflicting))
