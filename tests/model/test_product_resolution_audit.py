"""Local audit fields are explicitly product-only; preparation stays identical."""
from dataclasses import replace
from datetime import datetime, timezone
import json

from Application.interactions import InteractionRecord, InteractionSource, InteractionType
from Application.mindbox.records import ProductKey
from Application.model.product_resolution_audit import UnresolvedProductAudit
from Application.product_resolution import ProductCatalog, CatalogDiagnostics, ProductResolver


def test_product_only_grouping_attempt_denominator_and_no_customer_payload():
    catalog = ProductCatalog(frozenset({'001234'}), CatalogDiagnostics())
    resolver = ProductResolver(catalog)
    audit = UnresolvedProductAudit(catalog)
    record = InteractionRecord('SECRET_SOURCE_CUSTOMER', 'SECRET_CUSTOMER', ProductKey('offline1C', '999999-size'),
        InteractionType.VIEW, datetime(2025, 1, 1, tzinfo=timezone.utc), InteractionSource.ACTION, 'SECRET_EVENT')
    for event in (record, record, replace(record, interaction_type=InteractionType.PURCHASE, source=InteractionSource.ORDER),
                  replace(record, product=ProductKey('offline1C', '001234-size'))):
        resolved = resolver.resolve_interaction(event, strict=False)
        audit.observe(resolver, event, resolved)
    report = audit.report()
    assert report['total_events'] == 3 and report['product_resolution_attempts'] == 4
    assert report['resolved_events'] == 1 and report['unresolved_rate'] == .75
    assert report['unique_source_keys'] == report['unique_candidates'] == 1
    assert report['by_source'] == {'ACTION': 2, 'ORDER': 1}
    assert report['by_interaction_type'] == {'PURCHASE': 1, 'VIEW': 2}
    assert all(row['candidate'] == '999999' and not row['catalog_present'] for row in report['products'])
    assert sorted(row['count'] for row in report['products']) == [1, 2]
    assert not any(secret in json.dumps(report) for secret in ('SECRET_CUSTOMER', 'SECRET_SOURCE_CUSTOMER', 'SECRET_EVENT'))
    assert resolver.diagnostics.total.interactions_total == 4  # Observe never increments resolver counters twice.


def test_contact_shaped_product_and_unknown_namespace_are_redacted():
    catalog = ProductCatalog(frozenset(), CatalogDiagnostics())
    resolver = ProductResolver(catalog)
    audit = UnresolvedProductAudit(catalog)
    base = InteractionRecord('SECRET', 'SECRET', ProductKey('offline1C', 'person@example.com'), InteractionType.VIEW,
        datetime(2025, 1, 1, tzinfo=timezone.utc), InteractionSource.ACTION, 'SECRET')
    for record in (base, replace(base, product=ProductKey('PRIVATE_NAMESPACE', ' 555555 ')),
                   replace(base, product=ProductKey('offline1C', '79995551122'))):
        audit.observe(resolver, record, resolver.resolve_interaction(record, strict=False))
    text = json.dumps(audit.report())
    assert not any(value in text for value in ('SECRET', 'person@example.com', 'PRIVATE_NAMESPACE', '555555', '79995551122'))
    assert all(row['source_product'].startswith('sha256:') for row in audit.report()['products'])


def test_significant_unicode_size_suffix_and_prefix_are_not_normalized():
    catalog = ProductCatalog(frozenset({'268506'}), CatalogDiagnostics())
    resolver = ProductResolver(catalog)
    audit = UnresolvedProductAudit(catalog)
    base = InteractionRecord('SECRET', 'SECRET', ProductKey('offline1C', '009598_Без размера'), InteractionType.PURCHASE,
        datetime(2025, 1, 1, tzinfo=timezone.utc), InteractionSource.ORDER, 'SECRET')
    for event in (base, replace(base, product=ProductKey('offline1C', 'r268506_L'))):
        audit.observe(resolver, event, resolver.resolve_interaction(event, strict=False))
    rows = audit.report()['products']
    assert {row['candidate'] for row in rows} == {'009598', 'r26850'}
    assert all(not row['catalog_present'] for row in rows)
    assert any(row['source_product'] == '009598_Без размера' for row in rows)
