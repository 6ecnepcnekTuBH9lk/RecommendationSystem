"""Opt-in local product-only audit; never embedded in GUI/production reports."""
from collections import Counter
import hashlib
import re

from Application.product_resolution import ResolutionStatus
from Application.mindbox.selection import SUPPORTED_NAMESPACES


class UnresolvedProductAudit:
    def __init__(self, catalog):
        self.catalog = catalog
        self.attempts = 0
        self.resolved = 0
        self._products = Counter()

    def observe(self, resolver, interaction, resolved):
        self.attempts += 1
        if resolved is not None:
            self.resolved += 1
            return
        product = interaction.product
        status = resolver.resolve(product, strict=False).status
        namespace = product.namespace if product.namespace in SUPPORTED_NAMESPACES else 'unsupported'
        # Product keys are allowed only in this explicitly requested local artifact.
        # Invalid/arbitrary strings (including contact-shaped values) are redacted.
        safe = (status is ResolutionStatus.UNKNOWN_CANDIDATE and isinstance(product.value, str)
                and (re.fullmatch(r'[A-Za-z0-9_.:-]{1,128}', product.value) is not None
                     or (re.fullmatch(r'[\w .:-]{1,128}', product.value) is not None and re.search(r'\d', product.value)))
                and re.fullmatch(r'\+?\d[\d ()-]{8,}\d', product.value) is None)
        value = product.value if safe else 'sha256:' + hashlib.sha256(
            str(product.value).encode('utf-8')).hexdigest()
        candidate = product.value[:6] if safe else None
        self._products[(interaction.source.value, interaction.interaction_type.value, namespace,
                        status.value, value, candidate)] += 1

    def report(self):
        rows = []
        for (source, kind, namespace, status, value, candidate), count in sorted(
                self._products.items(), key=lambda item: tuple(str(value) for value in item[0])):
            rows.append({'source': source, 'interaction_type': kind, 'namespace': namespace,
                         'status': status, 'source_product': value, 'normalized_product': value,
                         'candidate': candidate, 'catalog_present': candidate in self.catalog.item_ids,
                         'full_key_catalog_present': value in self.catalog.item_ids,
                         'identifier_length': len(value) if not value.startswith('sha256:') else None,
                         'count': count})
        def breakdown(field):
            counts = Counter()
            for row in rows:
                counts[row[field]] += row['count']
            return dict(sorted(counts.items()))
        total = sum(self._products.values())
        return {'total_events': total, 'product_resolution_attempts': self.attempts,
                'resolved_events': self.resolved, 'unresolved_rate': total / self.attempts if self.attempts else 0,
                'unique_source_keys': len({(row['namespace'], row['source_product']) for row in rows}),
                'unique_candidates': len({row['candidate'] for row in rows if row['candidate'] is not None}),
                'by_source': breakdown('source'), 'by_interaction_type': breakdown('interaction_type'),
                'by_namespace': breakdown('namespace'), 'by_status': breakdown('status'), 'products': rows}
