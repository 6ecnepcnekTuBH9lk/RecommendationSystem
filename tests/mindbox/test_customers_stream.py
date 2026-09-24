from dataclasses import fields
from decimal import Decimal
import gc
import json

import pytest

from Application.mindbox.customers_stream import iter_customers_stream
from Application.mindbox.raw_reader import RawExportError
from Application.mindbox.adapters import adapt_customer
from Application.mindbox.adapters.customer_contacts import adapt_customer_contact_candidate
from Application.mindbox.identity import CustomerIdResolver
from Application.customer_profiles import ContactCandidate, build_customer_contact_index


def write(tmp_path, text, number=1):
    (tmp_path / f"customers_part_{number:03d}.json").write_text(text, encoding="utf-8-sig")


def test_progressive_yield_no_whole_file_json_load(tmp_path, monkeypatch):
    # A valid first object followed by an invalid tail beyond the parser buffer.
    write(tmp_path, '{"customers":[{"ids":{"mindboxId":"first"},"balance":1.25},' + ' ' * 200000 + 'BROKEN]}')
    monkeypatch.setattr(json, "load", lambda *a, **kw: pytest.fail("Whole-file JSON load forbidden"))
    records = iter_customers_stream(input_dir=tmp_path)
    first = next(records)
    assert first["ids"]["mindboxId"] == "first"
    assert first["balance"] == Decimal("1.25")
    with pytest.raises(RawExportError):
        list(records)


@pytest.mark.parametrize("payload", ['{}', '[]', '{"other":[]}', '{"customers":{}}', '{"customers":[1]}',
    '{"customers":[]', '{"customers":[]} extra', '{"customers":[],"extra":1}',
    '{"customers":[{"x":1,"x":2}]}', '{"customers":[{"x":NaN}]}'])
def test_invalid_json_safe_error(tmp_path, payload):
    write(tmp_path, payload)
    with pytest.raises(RawExportError) as exc:
        list(iter_customers_stream(input_dir=tmp_path))
    assert payload not in str(exc.value)


def test_empty_and_multipart_tie_order(tmp_path):
    write(tmp_path, '{"customers":[]}', 1)
    write(tmp_path, '{"customers":[{"ids":{"mindboxId":"u"},"email":"first@secret.test"}]}', 2)
    write(tmp_path, '{"customers":[{"ids":{"mindboxId":"u"},"email":"last@secret.test"}]}', 3)
    resolver = CustomerIdResolver()
    index = build_customer_contact_index(adapt_customer_contact_candidate(r, resolver)
        for r in iter_customers_stream(input_dir=tmp_path))
    assert index.contacts[0].email == "last@secret.test"
    assert index.diagnostics.customer_records == 2


def test_projection_omits_large_nested_payload_and_parity():
    raw = {"ids": {"mindboxId": "u"}, "email": "secret@example.test", "mobilePhone": "89161234567",
           "changeDateTimeUtc": "2026-01-01T00:00:00+03:00",
           "lastActivatedCard": {"ids": {"number": "secret-card"}},
           "customFields": {"huge": ["irrelevant"] * 100000}, "balances": [{"unused": [1] * 10000}],
           "subscriptions": [{"unused": True}], "segmentations": [{"unused": True}]}
    resolver = CustomerIdResolver()
    candidate = adapt_customer_contact_candidate(raw, resolver)
    assert {f.name for f in fields(candidate)} == {"customer_id", "is_canonical_source", "change_datetime_utc",
                                                  "email", "mobile_phone", "discount_card", "card_counts"}
    assert not hasattr(candidate, "__dict__")
    old = build_customer_contact_index([adapt_customer(raw, resolver)])
    new = build_customer_contact_index([candidate])
    assert new == old
    assert "secret" not in repr(candidate)


def test_model_projection_retains_only_target_candidates():
    resolver = CustomerIdResolver()
    wanted = {"target"}
    baseline = sum(type(o) is ContactCandidate for o in gc.get_objects())
    peak = 0
    def records():
        nonlocal peak
        for i in range(2000):
            raw = {"ids": {"mindboxId": "target" if i % 10 == 0 else f"other-{i}"}, "email": "safe@secret.test",
                   "customFields": {"unused": [1] * 1000}}
            candidate = adapt_customer_contact_candidate(raw, resolver, wanted)
            if candidate.customer_id != "target":
                assert not isinstance(candidate, ContactCandidate)
                assert not hasattr(candidate, "email")
            yield candidate
            if i % 100 == 0:
                peak = max(peak, sum(type(o) is ContactCandidate for o in gc.get_objects()) - baseline)
    result = build_customer_contact_index(records(), ["target"])
    assert peak <= len(wanted) + 2
    assert result.diagnostics.matched_profiles == 1
    assert result.diagnostics.canonical_profiles == 1801


@pytest.mark.parametrize("cards", [[], [{"ids": {"number": "one"}}],
    [{"ids": {"number": "one"}}, {"ids": {"number": "two"}}],
    [{"ids": {"number": "one"}}, {"ids": {"number": "one"}}],
    [{"ids": {"unknown": "value"}}, {"ids": {"number": False}}]])
def test_projection_selection_card_diagnostics_parity(cards):
    from Application.mindbox.records import CustomerMergeRecord
    from datetime import datetime, timezone
    resolver = CustomerIdResolver([CustomerMergeRecord(1, datetime(2026, 1, 1, tzinfo=timezone.utc), "canonical", ("alias",))])
    rows = [{"ids": {"mindboxId": "alias"}, "email": "alias@secret.test", "changeDateTimeUtc": "2026-08-01T00:00:00Z"},
            {"ids": {"mindboxId": "canonical"}, "mobilePhone": "89161234567", "discountCards": cards},
            {"ids": {"mindboxId": "canonical"}, "email": "winner@secret.test", "discountCards": cards}]
    old = build_customer_contact_index(adapt_customer(r, resolver) for r in rows)
    new = build_customer_contact_index(adapt_customer_contact_candidate(r, resolver) for r in rows)
    assert new == old
    assert new.contacts[0].mobile_phone is None
