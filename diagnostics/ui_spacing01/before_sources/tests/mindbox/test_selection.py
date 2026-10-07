from dataclasses import fields, replace

import pytest

from Application.interactions import DEFAULT_RULES
from Application.mindbox.selection import DEFAULT_SELECTION, MindboxSelectionConfig


def test_legacy_defaults_and_ordered_normalization():
    assert DEFAULT_SELECTION.interaction_rules() == DEFAULT_RULES
    assert DEFAULT_SELECTION.action_product_namespaces == (
        "offline1C",
        "kanzlerKz",
    )
    assert DEFAULT_SELECTION.order_product_namespaces == ("offline1C", "kanzlerKz")
    assert DEFAULT_SELECTION.purchase_line_statuses == ("CP", "delivering", "F")
    selection = replace(DEFAULT_SELECTION, view_action_system_names=[" second ", "first", "second"])
    assert selection.view_action_system_names == ("second", "first")


@pytest.mark.parametrize("field", [field.name for field in fields(MindboxSelectionConfig)])
@pytest.mark.parametrize("value", [(), ("",), ("  ",), (123,), "raw-secret", None])
def test_invalid_group_is_safe(field, value):
    with pytest.raises(ValueError) as exc:
        replace(DEFAULT_SELECTION, **{field: value})
    assert "raw-secret" not in str(exc.value)


def test_overlap_and_unsupported_namespace():
    with pytest.raises(ValueError, match="пересекаться"):
        MindboxSelectionConfig(view_action_system_names=("same",), favorite_action_system_names=("same",))
    for field in ("action_product_namespaces", "order_product_namespaces"):
        with pytest.raises(ValueError, match="поддерживаются"):
            replace(DEFAULT_SELECTION, **{field: ("unknown-secret",)})
