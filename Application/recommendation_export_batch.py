"""Read-only batch session. Reuses the card's scoring and purchase consensus.

Immutable model/catalog/resolver preparation is shared within ONE export only.
Source validation moves to bounded batch boundaries and final publication; any
change invalidates the entire unpublished archive. No canonical lock is held
around the run, so independent customer-card readers are not locked out.
"""

from collections import Counter
from contextlib import ExitStack
from copy import copy
from dataclasses import replace
from datetime import date
from decimal import Decimal
import json
from pathlib import Path
import sqlite3

from Application import customer_history_index as history
from Application import customer_profile_service as profiles
from Application.customer_card_data import CustomerCardService
from Application.customer_propensity_service import (
    CustomerPropensityService,
    CustomerPropensity,
    PropensityStatus,
    pseudonym,
    _stamp,
    _KeyUnavailable,
    _Stale,
)
from Application.hybrid_recommendations import HybridRecommendationService
from Application.model.raw_bpr_inference import RawBprInferenceService
from Application.model.propensity_ml.predictions import PredictionStatus
from Application.purchased_products import _PurchaseScope
from Application.recommendation_catalog import signature
from Application.recommendation_export_records import (
    ExportCustomer,
    ExportMetadata,
    ExportError,
    ExportCancelled,
)


class _SnapshotInference(RawBprInferenceService):
    def __init__(self, model_dir):
        super().__init__()
        self.snapshot = super()._get_model(model_dir)

    def _get_model(self, model_dir):
        # The owning session guards the complete source set before/after a batch.
        return self.snapshot

    def get_candidate_sources(self, model_user_id, *, model_dir):
        # Scoring, dtype, stable tie order and full ranking are inherited intact.
        return (
            self.get_recommendations(model_user_id, k=2**31 - 1, model_dir=model_dir),
            self.snapshot.item_metadata,
        )


def _fresh_scope(template):
    scope = copy(template)
    scope.purchased, scope.uncertain = set(template.purchased), set(template.uncertain)
    scope.statuses, scope.identities = Counter(), {}
    return scope


class _SnapshotPurchases:
    def __init__(self, connection, ready, catalog, historical_codes, cancelled):
        self.connection, self.ready, self.cancelled = connection, ready, cancelled
        self.current = _PurchaseScope(catalog.all_codes)
        self.historical = _PurchaseScope(historical_codes)
        self.historical.uncertain.update(
            code
            for code in historical_codes
            if not code
            or len(code) > 6
            or any(not c.isprintable() or c.isspace() for c in code)
        )

    def get(
        self,
        customer,
        catalog,
        *,
        raw_root,
        cancelled=lambda: False,
        historical_codes=None,
    ):
        if (
            frozenset(historical_codes or ()) != self.historical.codes
            or catalog.all_codes != self.current.codes
        ):
            raise ExportError("INCOMPATIBLE_PURCHASE_SCOPE")
        current, historical = _fresh_scope(self.current), _fresh_scope(self.historical)
        db = self.connection
        # Identical indexed SQL/row fields and assess_order/evidence as the card.
        events = db.execute(
            "SELECT e.event_key,e.owner_conflict FROM event_customers ec "
            "JOIN events e USING(event_key) WHERE ec.canonical_customer_id=? AND e.source='ORDER'",
            (customer,),
        )
        for event in events:
            if cancelled() or self.cancelled():
                raise ExportCancelled
            snapshots = []
            for (variant,) in db.execute(
                "SELECT variant_key FROM variants WHERE event_key=?",
                (event["event_key"],),
            ):
                snapshots.append(
                    db.execute(
                        "SELECT l.source_line_key,l.ordinal,l.raw_status,l.quantity,"
                        "p.selected_namespace,p.selected_value FROM order_lines l "
                        "LEFT JOIN product_occurrences p ON p.variant_key=l.variant_key "
                        "AND p.role='line' AND p.ordinal=l.ordinal WHERE l.variant_key=?",
                        (variant,),
                    ).fetchall()
                )
            current.add(snapshots, bool(event["owner_conflict"]))
            historical.add(snapshots, bool(event["owner_conflict"]))
        return replace(
            current.evidence(self.ready), historical=historical.evidence(self.ready)
        )

    def is_current(self, evidence, *, raw_root):
        return (
            evidence is not None
            and evidence.generation == self.ready.generation
            and evidence.source_version == self.ready.source_version
        )


def _ro(path):
    db = sqlite3.connect(
        Path(path).resolve().as_uri() + "?mode=ro", uri=True, timeout=0.5
    )
    db.execute("PRAGMA query_only=ON")
    db.execute("PRAGMA trusted_schema=OFF")
    return db


class RecommendationBatchSession:
    def __init__(
        self, *, audience="bpr", limit=None, cancelled=lambda: False, batch_size=100
    ):
        if audience != "bpr" or (
            limit is not None and (type(limit) is not int or limit < 1)
        ):
            raise ExportError("INVALID_INPUT")
        if type(batch_size) is not int or not 1 <= batch_size <= 1000:
            raise ExportError("INVALID_INPUT")
        self.audience, self.limit, self.cancelled, self.batch_size = (
            audience,
            limit,
            cancelled,
            batch_size,
        )
        self.stack = ExitStack()
        self.skipped_profiles = 0
        self.propensity = None

    def _check_cancelled(self):
        if self.cancelled():
            raise ExportCancelled

    def __enter__(self):
        try:
            self._check_cancelled()
            card = CustomerCardService(hybrid=HybridRecommendationService())
            self.card, self.root = card, card.raw_root
            self.identity_sources, self.index = card.bridge._get_index(
                card.model_dir, self.root
            )
            self._check_cancelled()
            self.inference = _SnapshotInference(card.model_dir)
            self.catalog = card.hybrid.catalog.load()
            self.profile_path = profiles.database(self.root)
            profile_identity = profiles._identity_source(self.root)
            self.profile_key = profile_identity[0]
            self.profile_version = profiles._database_version(self.profile_path)
            self.history_path = card.hybrid.purchases._index.index_path(
                raw_root=self.root
            )
            self.history_sources = history._capture_sources(self.root)
            if (
                self.history_sources.components[0].directory
                != self.identity_sources.merges_directory
                or profile_identity[1] != self.identity_sources.merges_directory
            ):
                raise ExportError("INCOMPATIBLE_IDENTITY_SOURCES")
            self.history_version = history._database_version(self.history_path)
            self.db = self.stack.enter_context(history._read_index(self.history_path))
            self.ready = history._inspect_connection(
                self.db, self.history_sources
            )  # No history integrity scan.
            self.db.commit()
            # Verify coverage once, using the exact card purchase-request contract.
            from Application.customer_history_service import (
                _request,
                _coverage_complete,
            )

            if not _coverage_complete(
                self.ready.coverage, _request("EXPORT", ("ORDER",), None, None, 1)
            ):
                raise ExportError("INCOMPLETE_HISTORY_COVERAGE")
            self.db.row_factory = sqlite3.Row
            self.db.set_progress_handler(lambda: int(self.cancelled()), 1000)
            self.profiles = _ro(self.profile_path)
            self.stack.callback(self.profiles.close)
            fields = {
                row[1]: row
                for row in self.profiles.execute("PRAGMA table_info(profiles)")
            }
            if not {"id", "changed", "raw"} <= set(fields) or fields["id"][5] != 1:
                raise ExportError("INCOMPATIBLE_PROFILES")
            purchases = _SnapshotPurchases(
                self.db,
                self.ready,
                self.catalog,
                frozenset(self.inference.snapshot.idx2item),
                self.cancelled,
            )
            self.hybrid = HybridRecommendationService(purchases=purchases)
            self.hybrid.catalog = self.catalog
            self.rules_version = signature(self.hybrid.rules_path)
            from Application.recommendation_rules import load_rules

            rules = load_rules(self.hybrid.rules_path)
            rules.validate_catalog(self.catalog)
            self.catalog_version = self.catalog._key
            self._prepare_propensity()
            total = sum(len(g) == 1 for g in self.index._groups.values())
            total = min(total, self.limit) if self.limit else total
            propensity_versions = [
                ("Propensity availability", self.propensity_status.value)
            ]
            if self.propensity is not None:
                manifest = self.propensity._store.manifest
                propensity_versions.extend(
                    (f"Propensity {field}", str(manifest["contract"][field]))
                    for field in (
                        "model_version",
                        "dataset_version",
                        "feature_schema_version",
                        "prediction_date",
                        "target_until",
                    )
                )
                propensity_versions.append(
                    ("Prediction artifact version", manifest["version"])
                )
            self.metadata = ExportMetadata(
                self.audience,
                total,
                len(self.inference.snapshot.user2idx),
                self.propensity._store.manifest["rows"]
                if self.propensity is not None
                else 0,
                (
                    (
                        "BPR generation",
                        str(
                            self.inference.snapshot.generation or "legacy checked pair"
                        ),
                    ),
                    ("History generation", self.ready.generation),
                    ("History source version", self.ready.source_version),
                    ("Canonical revision", self.identity_sources.revision),
                    ("Selected kinds", ", ".join(rules.kinds)),
                    ("Selected collections", ", ".join(rules.collections)),
                    ("Minimum stock", str(rules.minimum_stock)),
                    ("Primary collection", rules.primary_collection or "Не выбрана"),
                    ("Audience", "Подтверждённые профили клиентов BPR-MF"),
                    ("Catalog signature", json.dumps(self.catalog_version)),
                    ("Rules signature", json.dumps(self.rules_version)),
                )
                + tuple(propensity_versions),
            )
            self.guard()
            return self
        except BaseException:
            self.secret = None
            self.stack.close()
            raise

    def _prepare_propensity(self):
        self.propensity_status = PropensityStatus.NOT_READY
        self.secret = None
        service = CustomerPropensityService()
        try:
            store = (
                service._load_store()
            )  # Existing manifest/SHA/schema/small integrity/model/dataset checks.
            c = store.manifest["contract"]
            if (
                c["source_version"] != self.ready.source_version
                or c["generation"] != self.ready.generation
            ):
                self.propensity_status = PropensityStatus.STALE
                return
            secret = service._key()
            # Domain/key_id verification without serializing a customer binding.
            pseudonym("EXPORT-KEY-COMPATIBILITY-CHECK", secret, c["key_id"])
            self.propensity, self.secret = service, secret
            self.propensity_status = PropensityStatus.READY
        except _KeyUnavailable:
            self.propensity_status = PropensityStatus.KEY_UNAVAILABLE
        except _Stale:
            self.propensity_status = PropensityStatus.STALE
        except FileNotFoundError:
            self.propensity_status = PropensityStatus.NOT_READY
        except Exception:
            self.propensity_status = PropensityStatus.ERROR

    def guard(self):
        self._check_cancelled()
        from Application.model.customer_identity_bridge import _capture_sources

        expected = (
            self.identity_sources.key,
            self.profile_key,
            self.profile_version,
            self.history_sources.version,
            self.history_version,
            self.catalog_version,
            self.rules_version,
        )
        actual = (
            _capture_sources(self.card.model_dir, self.root).key,
            profiles._identity_source(self.root)[0],
            profiles._database_version(self.profile_path),
            history._capture_sources(self.root).version,
            history._database_version(self.history_path),
            (signature(self.catalog.path), signature(self.catalog.categories_path)),
            signature(self.hybrid.rules_path),
        )
        if actual != expected:
            raise ExportError("SOURCES_CHANGED")
        if self.propensity is not None:
            if (
                self.propensity._files() != self.propensity._artifact_version
                or _stamp(self.propensity.key_path) != self.propensity._key_version
            ):
                raise ExportError("PROPENSITY_SOURCES_CHANGED")

    def _prediction(self, customer):
        if self.propensity is None:
            return CustomerPropensity(self.propensity_status)
        c = self.propensity._store.manifest["contract"]
        if self.index._resolver.resolve(customer) != customer:
            raise ExportError("IDENTITY_CHANGED")
        key = pseudonym(customer, self.secret, c["key_id"])
        found = self.propensity._store.get_prediction(
            key, c["model_version"], c["prediction_date"]
        )
        if found.status is PredictionStatus.NOT_FOUND:
            return CustomerPropensity(PropensityStatus.NOT_FOUND, customer)
        if found.status is not PredictionStatus.READY:
            raise ExportError("PROPENSITY_SOURCES_CHANGED")
        p = found.prediction
        return CustomerPropensity(
            PropensityStatus.WARNING if p.warnings else PropensityStatus.READY,
            customer,
            p.probability,
            p.prediction_date,
            p.target_until,
            p.model_version,
            p.warnings,
        )

    def _ids(self):
        yield from sorted(c for c, g in self.index._groups.items() if len(g) == 1)

    def __iter__(self):
        try:
            for i, customer in enumerate(self._ids()):
                if self.limit is not None and i >= self.limit:
                    break
                self._check_cancelled()
                if i % self.batch_size == 0:
                    self.guard()
                    self.db.execute("BEGIN")
                    self.profiles.execute("BEGIN")
                row = self.profiles.execute(
                    "SELECT id,changed,raw FROM profiles WHERE id=? LIMIT 1",
                    (customer,),
                ).fetchone()
                record = None
                if row is not None:
                    try:
                        raw = json.loads(
                            row[2],
                            object_pairs_hook=profiles._unique_object,
                            parse_float=Decimal,
                            parse_constant=profiles._reject_number,
                        )
                        p = profiles._adapt(
                            raw, row[1], customer, self.index._resolver, date.today()
                        )
                    except Exception:
                        p = None
                    if p is not None:
                        identity = replace(
                            self.index.resolve(customer),
                            model_generation=self.identity_sources.generation,
                            canonical_revision=self.identity_sources.revision,
                            source_version=self.identity_sources.version,
                        )
                        result = (
                            self.hybrid.get(
                                identity,
                                self.inference,
                                model_dir=self.card.model_dir,
                                raw_root=self.root,
                                cancelled=self.cancelled,
                            )
                            if identity.is_matched
                            else None
                        )
                        if result is not None and (
                            result.status == "STALE" or result.raw_status == "STALE"
                        ):
                            raise ExportError("SOURCES_CHANGED")
                        if result is not None and result.status not in (
                            "READY",
                            "EMPTY",
                            "PURCHASES_NOT_READY",
                            "CANCELLED",
                        ):
                            raise ExportError("RECOMMENDATIONS_UNAVAILABLE")
                        record = ExportCustomer(
                            customer,
                            p.contact,
                            p.phone_confirmed,
                            p.phone_invalid,
                            result,
                            self._prediction(customer),
                            result.status
                            if result is not None
                            else identity.status.value,
                        )
                if record is None:
                    self.skipped_profiles += 1
                yield record
                if (i + 1) % self.batch_size == 0:
                    self.db.commit()
                    self.profiles.commit()
                    self.guard()
            self.db.commit()
            self.profiles.commit()
            self.guard()
        except BaseException:
            self.db.rollback()
            self.profiles.rollback()
            raise

    def __exit__(self, *exc):
        self.secret = None
        self.stack.close()
