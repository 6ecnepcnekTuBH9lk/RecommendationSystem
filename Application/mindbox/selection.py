"""Reproducible local interaction selection; never part of an API payload."""

from dataclasses import dataclass, fields


DEFAULT_VIEW_ACTIONS = ("ProsmotrProdukta", "ProsmotrProduktaVApiMethod")
DEFAULT_FAVORITE_ACTIONS = (
    "DobavlenieProduktaVSpisokVOperaciiDobavlenie",
    "DobavlenieProduktaVSpisokVOperaciiDobavlenieTovara",
)
DEFAULT_PURCHASE_STATUSES = ("CP", "delivering", "F")
DEFAULT_ACTION_NAMESPACES = ("offline1C",)
DEFAULT_ORDER_NAMESPACES = ("offline1C", "kanzlerKz")
SUPPORTED_NAMESPACES = frozenset(DEFAULT_ORDER_NAMESPACES)

# Shared spelling of export-daily options; resume has no selection options.
SELECTION_OPTIONS = {
    "view_action_system_names": "--view-action",
    "favorite_action_system_names": "--favorite-action",
    "purchase_line_statuses": "--purchase-status",
    "action_product_namespaces": "--action-product-namespace",
    "order_product_namespaces": "--order-product-namespace",
}


@dataclass(frozen=True)
class MindboxSelectionConfig:
    view_action_system_names: tuple[str, ...] = DEFAULT_VIEW_ACTIONS
    favorite_action_system_names: tuple[str, ...] = DEFAULT_FAVORITE_ACTIONS
    purchase_line_statuses: tuple[str, ...] = DEFAULT_PURCHASE_STATUSES
    action_product_namespaces: tuple[str, ...] = DEFAULT_ACTION_NAMESPACES
    order_product_namespaces: tuple[str, ...] = DEFAULT_ORDER_NAMESPACES

    def __post_init__(self):
        for field in fields(self):
            values = getattr(self, field.name)
            if not isinstance(values, (tuple, list)) or not values or any(
                not isinstance(value, str) or not value.strip() for value in values
            ):
                raise ValueError("Каждая группа отбора должна содержать непустые строки.")
            object.__setattr__(self, field.name, tuple(dict.fromkeys(value.strip() for value in values)))
        if set(self.view_action_system_names) & set(self.favorite_action_system_names):
            raise ValueError("События просмотров и избранного не должны пересекаться.")
        if any(value not in SUPPORTED_NAMESPACES
               for value in (*self.action_product_namespaces, *self.order_product_namespaces)):
            raise ValueError("ID товаров: поддерживаются только offline1C и kanzlerKz.")

    def interaction_rules(self):
        from Application.interactions import InteractionRules

        return InteractionRules(
            view_action_system_names=frozenset(self.view_action_system_names),
            favorite_action_system_names=frozenset(self.favorite_action_system_names),
            purchase_line_statuses=frozenset(self.purchase_line_statuses),
        )


DEFAULT_SELECTION = MindboxSelectionConfig()
