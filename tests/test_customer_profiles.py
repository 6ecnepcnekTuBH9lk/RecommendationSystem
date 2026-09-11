from dataclasses import FrozenInstanceError

import pytest

from Application.customer_profiles import build_customer_contact_index, CustomerContactError
from Application.mindbox.adapters import adapt_customer, adapt_customer_merge
from Application.mindbox.identity import CustomerIdResolver


def record(source="source-secret", **kwargs):
    return adapt_customer({"ids": {"mindboxId": source}, **kwargs}, CustomerIdResolver())


@pytest.mark.parametrize("activated,cards,expected,ambiguous,fallback", [
    ({"ids": {"number": "activated-secret"}}, [{"ids": {"number": "other-secret"}}], "activated-secret", 0, 0),
    (None, [{"ids": {"number": "one-secret"}}], "one-secret", 0, 1),
    (None, [{"ids": {"number": "one-secret"}}, {"ids": {"number": "one-secret"}}], "one-secret", 0, 1),
    (None, [{"ids": {"number": "one-secret"}}, {"ids": {"number": "two-secret"}}], None, 1, 0),
    (None, [], None, 0, 0),
    ({"ids": {"otherKey": "secret"}}, [{"ids": {"number": "one-secret"}}], "one-secret", 0, 1),
    ({"ids": {"number": ["secret"]}}, [], None, 0, 0),
])
def test_contacts_cards_and_safe_repr(activated, cards, expected, ambiguous, fallback):
    profile = record(email="secret@example.test", mobilePhone="+7 (916) 123-45-67",
                     lastActivatedCard=activated, discountCards=cards)
    index = build_customer_contact_index([profile], ["source-secret", "missing-secret"])
    contact = index.contacts[0]
    assert contact.email == "secret@example.test"
    assert contact.mobile_phone == "+7 (916) 123-45-67"
    assert contact.discount_card == expected
    assert index.diagnostics.ambiguous_discount_cards == ambiguous
    assert index.diagnostics.single_card_fallback == fallback
    assert index.diagnostics.missing_profiles == 1
    assert index.contacts[1].email is None
    assert "secret" not in repr(index) + repr(contact)
    with pytest.raises(FrozenInstanceError):
        contact.email = "other"
    with pytest.raises(FrozenInstanceError):
        index.contacts = ()
    with pytest.raises(CustomerContactError):
        index.validate_alignment(["missing-secret", "source-secret"])


@pytest.mark.parametrize("canonical", [True, False])
def test_whole_record_priority_latest_and_export_tie(canonical):
    resolver = CustomerIdResolver([adapt_customer_merge({"id": 1, "dateTimeUtc": "2026-01-01T00:00:00Z",
        "resultingCustomer": {"ids": {"mindboxId": "canonical-secret"}},
        "mergedCustomers": [{"ids": {"mindboxId": "alias-secret"}}]})])
    rows = [{"ids": {"mindboxId": "alias-secret"}, "changeDateTimeUtc": "2026-08-02T00:00:00Z",
             "email": "alias-new-secret@example.test", "mobilePhone": "89161234567"},
            {"ids": {"mindboxId": "alias-secret"}, "changeDateTimeUtc": "2026-08-01T00:00:00Z",
             "email": "alias-old-secret@example.test"}]
    if canonical:
        rows += [{"ids": {"mindboxId": "canonical-secret"}, "changeDateTimeUtc": "2026-07-01T00:00:00Z",
                  "mobilePhone": "89169999999"},
                 {"ids": {"mindboxId": "canonical-secret"}, "changeDateTimeUtc": "2026-07-02T00:00:00Z",
                  "email": "canonical-first-secret@example.test"},
                 {"ids": {"mindboxId": "canonical-secret"}, "changeDateTimeUtc": "2026-07-02T00:00:00Z",
                  "email": "canonical-last-secret@example.test"}]
    rows += [{"ids": {"mindboxId": "outsider-secret"}, "email": "outside-secret@example.test"}]
    index = build_customer_contact_index((adapt_customer(r, resolver) for r in rows), ["canonical-secret"])
    assert index.contacts[0].email == ("canonical-last-secret@example.test" if canonical else "alias-new-secret@example.test")
    assert index.contacts[0].mobile_phone == (None if canonical else "89161234567")
    assert index.diagnostics.matched_profiles == 1
    assert index.diagnostics.canonical_profiles == 2


def test_absent_timestamp_last_export_row_and_unknown_card_schema():
    index = build_customer_contact_index([record(email="old-secret@example.test"),
        record(mobilePhone="89161234567", discountCards=[{"ids": {"externalId": "secret"}}, {"ids": {"number": False}}])])
    assert index.contacts[0].email is None
    assert index.diagnostics.cards_without_number == 1
    assert index.diagnostics.invalid_card_numbers == 1


def test_contacts_not_accepted_or_written_by_model_publication(tmp_path, monkeypatch):
    from Application.model import BPRMF as core
    from Application.model.training_data import Mappings
    monkeypatch.chdir(tmp_path)
    index = build_customer_contact_index([record(email="pii@example.test", mobilePhone="89161234567",
        lastActivatedCard={"ids": {"number": "PII_CARD"}})])
    cfg = core.TrainConfig(data_dir=str(tmp_path), use_item_features=False)
    maps = Mappings({"source-secret": 0}, ["source-secret"], {"item": 0}, ["item"])
    model = core.BPRMF(1, 1, cfg.embedding_dim)
    with pytest.raises(TypeError):
        core._save_artifacts(cfg, maps, model, customer_contacts=index)
    assert not (tmp_path / "Модель").exists()
    core._save_artifacts(cfg, maps, model)
    _, ckpt = core._load_artifacts()
    assert all(s not in repr(ckpt) for s in ("pii@example.test", "89161234567", "PII_CARD", "CustomerContact", "CustomerRecord"))
