from dataclasses import FrozenInstanceError, replace
from datetime import datetime, timezone
from decimal import Decimal

import pytest

from Application.interactions import InteractionRecord, InteractionSource, InteractionType
from Application.mindbox.records import ActionRecord, OrderLineRecord, ProductKey
from Application.product_resolution import (
    CatalogError, ProductResolver, ProductResolutionError, ResolutionStatus, load_catalog,
)


@pytest.fixture
def catalog_path(tmp_path):
    path = tmp_path / "catalog.csv"
    path.write_text("КодНоменклатуры|Название\n001234|Synthetic\nABC|Short\nABCDEF|Letters\n",
                    encoding="utf-8-sig")
    return path


@pytest.fixture
def resolver(catalog_path):
    return ProductResolver(load_catalog(catalog_path))


@pytest.fixture
def interaction():
    return InteractionRecord(
        source_customer_id="SECRET_SOURCE", customer_id="SECRET_CUSTOMER",
        product=ProductKey("offline1C", "001234_SECRET_PRODUCT"),
        interaction_type=InteractionType.VIEW,
        event_datetime_utc=datetime(2026, 1, 1, tzinfo=timezone.utc),
        source=InteractionSource.ACTION, source_event_id="SECRET_EVENT",
    )


def test_catalog_bom_pipe_duplicates_empty_and_file_unchanged(catalog_path):
    with catalog_path.open("a", encoding="utf-8") as stream:
        stream.write("001234|Different description\n|Empty\n   |Whitespace\n\n")
    before = catalog_path.read_bytes()
    catalog = load_catalog(catalog_path)
    assert catalog.item_ids == frozenset({"001234", "ABC", "ABCDEF"})
    assert catalog.diagnostics.rows_total == 7
    assert catalog.diagnostics.empty_code_rows == 3
    assert catalog.diagnostics.duplicate_code_rows == 1
    assert catalog.diagnostics.unique_items == 3
    assert catalog_path.read_bytes() == before
    assert "001234" not in repr(catalog)


@pytest.mark.parametrize("text", [
    "", "Other|Header\n1|2\n", "КодНоменклатуры|КодНоменклатуры\n",
    "КодНоменклатуры|\n", "КодНоменклатуры|Name\n1\n",
    'КодНоменклатуры\n"unclosed', "КодНоменклатуры\n 001234\n",
    "КодНоменклатуры\n001234 \n", "КодНоменклатуры,Name\n001234,name\n",
])
def test_bad_catalog_is_safe_error(tmp_path, text):
    path = tmp_path / "SECRET_PATH.csv"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(CatalogError) as exc:
        load_catalog(path)
    assert "001234" not in str(exc.value)
    assert "SECRET" not in str(exc.value)


def test_catalog_missing_and_bad_encoding(tmp_path):
    path = tmp_path / "SECRET.csv"
    with pytest.raises(CatalogError):
        load_catalog(path)
    path.write_bytes(b"\xff\xff")
    with pytest.raises(CatalogError):
        load_catalog(path)


def test_empty_catalog_has_no_valid_items(tmp_path):
    path = tmp_path / "catalog.csv"
    path.write_text("КодНоменклатуры\n\n", encoding="utf-8")
    assert load_catalog(path).item_ids == frozenset()


@pytest.mark.parametrize("namespace", ["offline1C", "kanzlerKz"])
@pytest.mark.parametrize("value,expected", [
    ("001234", "001234"), ("001234-size", "001234"), ("ABC", "ABC"), ("ABCDEFx", "ABCDEF"),
])
def test_legacy_prefix_and_short_ids(resolver, namespace, value, expected):
    key = ProductKey(namespace, value)
    result = resolver.resolve(key)
    assert result.status is ResolutionStatus.RESOLVED
    assert result.item_id == expected
    assert key == ProductKey(namespace, value)
    assert value not in repr(result)


@pytest.mark.parametrize("namespace,value,status", [
    ("offline1C", "UNKNOWN", ResolutionStatus.UNKNOWN_CANDIDATE),
    ("offline1C", "1234", ResolutionStatus.UNKNOWN_CANDIDATE),
    ("kanzlerKz", "abcdef", ResolutionStatus.UNKNOWN_CANDIDATE),
    ("offline1C", "AB CDE", ResolutionStatus.UNKNOWN_CANDIDATE),
    ("offline1C", " 001234", ResolutionStatus.INVALID_ID),
    ("offline1C", "001234 ", ResolutionStatus.INVALID_ID),
    ("offline1C", "", ResolutionStatus.INVALID_ID),
    ("offline1C", 123456, ResolutionStatus.INVALID_ID),
    ("SECRET_NAMESPACE", "001234", ResolutionStatus.UNSUPPORTED_NAMESPACE),
    ("Offline1C", "001234", ResolutionStatus.UNSUPPORTED_NAMESPACE),
])
def test_unresolved_strict_and_diagnostic(resolver, namespace, value, status):
    key = ProductKey(namespace, value)
    result = resolver.resolve(key, strict=False)
    assert result.status is status
    assert result.item_id is None
    with pytest.raises(ProductResolutionError) as exc:
        resolver.resolve(key)
    assert exc.value.status is status
    assert "SECRET" not in str(exc.value)
    assert "001234" not in str(exc.value)


def test_full_id_match_does_not_override_prefix_policy(tmp_path):
    path = tmp_path / "catalog.csv"
    path.write_text("КодНоменклатуры\n001234-long\n", encoding="utf-8")
    resolver = ProductResolver(load_catalog(path))
    assert resolver.resolve(ProductKey("offline1C", "001234-long"), strict=False).item_id is None


@pytest.mark.parametrize("kind", list(InteractionType))
def test_wrapper_preserves_entire_event(resolver, interaction, kind):
    interaction = replace(interaction, interaction_type=kind,
                          source=InteractionSource.ORDER if kind is InteractionType.PURCHASE else InteractionSource.ACTION,
                          quantity=Decimal("2.50") if kind is InteractionType.PURCHASE else None)
    resolved = resolver.resolve_interaction(interaction)
    assert resolved.interaction is interaction
    assert resolved.item_id == "001234"
    assert interaction.product.value == "001234_SECRET_PRODUCT"
    assert "SECRET" not in repr(resolved)
    assert "001234" not in repr(resolved)
    with pytest.raises(FrozenInstanceError):
        resolved.item_id = "other"
    assert resolver.diagnostics.by_type[kind].resolved == 1


def test_collapses_count_unique_keys_without_aggregating_events(resolver, interaction):
    events = [interaction, interaction,
              replace(interaction, product=ProductKey("offline1C", "001234-other")),
              replace(interaction, product=ProductKey("kanzlerKz", "001234-other"))]
    results = [resolver.resolve_interaction(event) for event in events]
    assert len(results) == 4
    assert all(result.item_id == "001234" for result in results)
    assert all(result.interaction is event for result, event in zip(results, events))
    stats = resolver.diagnostics
    assert stats.total.interactions_total == stats.total.resolved == 4
    assert stats.unique_source_product_keys == 3
    assert stats.unique_resolved_catalog_items == 1
    assert stats.catalog_items_with_multiple_source_keys == 1
    assert stats.max_source_keys_per_catalog_item == 3
    assert stats.by_namespace["offline1C"].resolved == 3
    assert stats.by_namespace["kanzlerKz"].resolved == 1


def test_diagnostics_include_failures_and_are_immutable_snapshots(resolver, interaction):
    initial = resolver.diagnostics
    assert initial.total.resolution_rate_percent == 0
    assert initial.max_source_keys_per_catalog_item == 0
    resolver.resolve_interaction(interaction)
    unknown = replace(interaction, product=ProductKey("kanzlerKz", "999999"),
                      interaction_type=InteractionType.FAVORITE)
    assert resolver.resolve_interaction(unknown, strict=False) is None
    unsupported = replace(interaction, product=ProductKey("SECRET_NAMESPACE", "001234"),
                          interaction_type=InteractionType.PURCHASE)
    with pytest.raises(ProductResolutionError):
        resolver.resolve_interaction(unsupported)
    stats = resolver.diagnostics
    assert stats.total.interactions_total == 3
    assert stats.total.resolved == 1
    assert stats.total.unresolved == 2
    assert stats.total.unsupported_namespace == 1
    assert stats.total.resolution_rate_percent == pytest.approx(100 / 3)
    assert stats.by_type[InteractionType.FAVORITE].unresolved == 1
    assert stats.by_type[InteractionType.PURCHASE].unsupported_namespace == 1
    assert stats.by_namespace["kanzlerKz"].unresolved == 1
    assert stats.by_namespace["unsupported"].unsupported_namespace == 1
    assert stats.unique_source_product_keys == 3
    assert initial.total.interactions_total == 0
    assert "SECRET" not in repr(stats)
    with pytest.raises(TypeError):
        stats.by_namespace["new"] = stats.total


@pytest.fixture
def cli_data(monkeypatch, interaction):
    import socket
    from Application.mindbox import adapters, raw_reader

    def no_network(*args, **kwargs):
        pytest.fail("Network is forbidden")

    monkeypatch.setattr(socket.socket, "connect", no_network)
    action = ActionRecord(
        action_id="SECRET_ACTION", action_system_name="ProsmotrProdukta",
        event_datetime_utc=interaction.event_datetime_utc, creation_datetime_utc=interaction.event_datetime_utc,
        source_customer_id="SECRET_SOURCE", customer_id="secret@example.test",
        products=(interaction.product,),
    )
    line = OrderLineRecord(
        order_id="SECRET_ORDER", order_datetime_utc=interaction.event_datetime_utc,
        channel_external_id="SECRET_CHANNEL", channel_name="SECRET_NAME",
        source_customer_id="SECRET_SOURCE", customer_id="+79999999999",
        line_id="SECRET_LINE", line_number=1, product=ProductKey("kanzlerKz", "001234-size"),
        product_name="SECRET_PRODUCT_NAME", quantity=Decimal("2"), base_price_per_item=Decimal("10"),
        price_of_line=Decimal("20"), line_status="CP",
    )
    data = {"customer_merges": [], "actions": [action, replace(
        action, action_system_name="DobavlenieProduktaVSpisokVOperaciiDobavlenie")], "orders": [(line,)]}
    monkeypatch.setattr(adapters, "adapt_action", lambda record, resolver: record)
    monkeypatch.setattr(adapters, "adapt_order", lambda records, resolver: records)
    monkeypatch.setattr(raw_reader, "iter_export", lambda name, **kwargs: iter(data[name]))
    return data


@pytest.mark.parametrize("problem", [None, "malformed", "unknown", "unsupported"])
@pytest.mark.parametrize("diagnose", [False, True])
def test_cli_counts_exit_codes_no_ids_or_files(cli_data, catalog_path, capsys, problem, diagnose):
    from scripts.product_resolution_smoke import main

    action = cli_data["actions"][0]
    if problem == "malformed":
        cli_data["actions"].insert(0, replace(action, products=()))
    elif problem is not None:
        key = ProductKey("SECRET_NAMESPACE" if problem == "unsupported" else "offline1C", "999999_SECRET")
        cli_data["actions"].insert(0, replace(action, products=(key,)))
    before = {path: path.read_bytes() for path in catalog_path.parent.iterdir()}
    args = ["--catalog", str(catalog_path), "--raw-root", str(catalog_path.parent)]
    assert main(args + (["--diagnose"] if diagnose else [])) == int(problem is not None)
    output = capsys.readouterr()
    for secret in ("SECRET", "001234", "999999", "secret@example.test", "+79999999999", "ABCDEF"):
        assert secret not in output.out + output.err
    if problem is None or diagnose:
        assert "  Resolved: 3" in output.out
        for kind in InteractionType:
            total = 1 + int(kind is InteractionType.VIEW and problem in ("unknown", "unsupported"))
            assert f"{kind.value}:\n  Interactions considered: {total}\n  Resolved: 1" in output.out
        assert f"Malformed upstream actions: {int(problem == 'malformed')}" in output.out
        assert f"  Unresolved: {int(problem in ('unknown', 'unsupported'))}" in output.out
    else:
        assert output.out == ""
    assert before == {path: path.read_bytes() for path in catalog_path.parent.iterdir()}


def test_cli_upstream_error_is_safe_and_not_suppressed(cli_data, catalog_path, monkeypatch, capsys):
    from Application.mindbox import adapters
    from scripts.product_resolution_smoke import main

    def broken(*args):
        raise adapters.AdapterError("SECRET secret@example.test 001234")

    monkeypatch.setattr(adapters, "adapt_action", broken)
    assert main(["--catalog", str(catalog_path), "--diagnose"]) == 1
    output = capsys.readouterr()
    assert output.out == ""
    assert "AdapterError" in output.err
    assert "SECRET" not in output.err


def test_cli_missing_catalog_reports_safe_reason_without_reading_raw(tmp_path, monkeypatch, capsys):
    from Application.mindbox import raw_reader
    from scripts.product_resolution_smoke import main

    def forbidden(*args, **kwargs):
        pytest.fail("Missing catalog must stop before raw exports")

    monkeypatch.setattr(raw_reader, "iter_export", forbidden)
    assert main(["--catalog", str(tmp_path / "SECRET.csv"), "--diagnose"]) == 1
    output = capsys.readouterr()
    assert output.out == ""
    assert "Catalog file not found" in output.err
    assert "SECRET" not in output.err
