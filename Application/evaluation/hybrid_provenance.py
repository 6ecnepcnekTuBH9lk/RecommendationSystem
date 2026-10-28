"""In-memory evidence receipts; no source discovery, file reads or certification.

A review binds an artifact's bytes AND its temporal/context statement. A receipt
is an auditable attestation by its supplier, not proof that an external reviewer
or source is truthful. Real source adapters must independently establish that.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
import hashlib
import json


class HybridEvaluationError(ValueError):
    """Only static error codes: never interpolate source data or identities."""


def utc(value):
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise HybridEvaluationError("AWARE_TIMESTAMP_REQUIRED")
    try:
        return value.astimezone(timezone.utc)
    except (ValueError, OverflowError):
        raise HybridEvaluationError("INVALID_TIMESTAMP") from None


def identifier(value):
    if not isinstance(value, str) or not value or value != value.strip():
        raise HybridEvaluationError("INVALID_IDENTIFIER")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=True,
                                     separators=(",", ":")).encode("ascii")).hexdigest()


def _checksum(value):
    if (not isinstance(value, str) or len(value) != 64
            or any(c not in "0123456789abcdef" for c in value)):
        raise HybridEvaluationError("INVALID_CHECKSUM")


class InputRole(str, Enum):
    TRAINING_HISTORY = "TRAINING_HISTORY"
    MODEL = "MODEL"
    HISTORICAL_ATTRIBUTES = "HISTORICAL_ATTRIBUTES"
    CATALOG = "CATALOG"
    CATEGORIES = "CATEGORIES"
    AVAILABILITY = "AVAILABILITY"
    RULES = "RULES"
    IDENTITY = "IDENTITY"
    PURCHASE_HISTORY = "PURCHASE_HISTORY"
    FUTURE_LABELS = "FUTURE_LABELS"


class EvidenceBasis(str, Enum):
    CONTROLLED_SYNTHETIC = "CONTROLLED_SYNTHETIC"
    REVIEWED_POINT_IN_TIME = "REVIEWED_POINT_IN_TIME"


class Validity(str, Enum):
    STRICT = "STRICT"
    PROXY = "PROXY"
    UNAVAILABLE = "UNAVAILABLE"


@dataclass(frozen=True, slots=True, repr=False)
class EvidenceReview:
    basis: EvidenceBasis
    statement_sha256: str
    reference: str
    reviewed_by: str

    def __post_init__(self):
        if not isinstance(self.basis, EvidenceBasis):
            raise HybridEvaluationError("INVALID_REVIEW_BASIS")
        _checksum(self.statement_sha256)
        identifier(self.reference)
        identifier(self.reviewed_by)


@dataclass(frozen=True, slots=True, repr=False)
class InputEvidence:
    role: InputRole
    content: bytes
    sha256: str
    context_sha256: str
    model_generation: str
    as_of: datetime
    available_at: datetime | None = None
    latest_event_at: datetime | None = None
    coverage_start: datetime | None = None
    coverage_end: datetime | None = None
    review: EvidenceReview | None = None

    def __post_init__(self):
        if not isinstance(self.role, InputRole) or type(self.content) is not bytes or not self.content:
            raise HybridEvaluationError("INVALID_EVIDENCE")
        _checksum(self.sha256)
        _checksum(self.context_sha256)
        identifier(self.model_generation)
        if hashlib.sha256(self.content).hexdigest() != self.sha256:
            raise HybridEvaluationError("ARTIFACT_CHECKSUM_MISMATCH")
        for name in ("as_of", "available_at", "latest_event_at", "coverage_start", "coverage_end"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, utc(value))
        if self.as_of is None:
            raise HybridEvaluationError("AWARE_TIMESTAMP_REQUIRED")
        if (self.coverage_start is None) != (self.coverage_end is None):
            raise HybridEvaluationError("INCOMPLETE_COVERAGE_BOUNDS")
        if self.coverage_start is not None and self.coverage_start >= self.coverage_end:
            raise HybridEvaluationError("INVALID_COVERAGE_BOUNDS")
        if self.review is not None:
            if not isinstance(self.review, EvidenceReview):
                raise HybridEvaluationError("INVALID_REVIEW")
            if self.review.statement_sha256 != self.statement_digest():
                raise HybridEvaluationError("REVIEW_BINDING_MISMATCH")

    def statement_digest(self):
        return digest({"role": self.role.value, "sha256": self.sha256,
                       "context": self.context_sha256, "generation": self.model_generation,
                       **{name: getattr(self, name).isoformat() if getattr(self, name) is not None else None
                          for name in ("as_of", "available_at", "latest_event_at", "coverage_start", "coverage_end")}})


@dataclass(frozen=True, slots=True)
class EvidenceIssue:
    role: InputRole
    reason: str  # Created only by assess_evidence's fixed-code branches.


@dataclass(frozen=True, slots=True)
class EvidenceAssessment:
    status: Validity
    issues: tuple[EvidenceIssue, ...] = ()


def assess_evidence(evidence, *, context_sha256, model_generation, cutoff, future_end, synthetic):
    """Missing inputs/leaking history are unavailable; unproven as-of is proxy."""
    cutoff, future_end = utc(cutoff), utc(future_end)
    _checksum(context_sha256)
    identifier(model_generation)
    if cutoff >= future_end or type(synthetic) is not bool or not isinstance(evidence, (tuple, list)):
        raise HybridEvaluationError("INVALID_EVIDENCE_ASSESSMENT")
    records = {}
    for record in evidence:
        if not isinstance(record, InputEvidence) or record.role in records:
            raise HybridEvaluationError("INVALID_EVIDENCE_SET")
        if record.context_sha256 != context_sha256:
            raise HybridEvaluationError("EVIDENCE_CONTEXT_MISMATCH")
        expected_as_of = future_end if record.role is InputRole.FUTURE_LABELS else cutoff
        if record.as_of != expected_as_of:
            raise HybridEvaluationError("EVIDENCE_CUTOFF_MISMATCH")
        records[record.role] = record
    issues, unavailable = [], False
    for role in InputRole:
        record = records.get(role)
        if record is None:
            issues.append(EvidenceIssue(role, "MISSING_INPUT"))
            unavailable = True
            continue
        if record.model_generation != model_generation:
            issues.append(EvidenceIssue(role, "MODEL_GENERATION_MISMATCH"))
            unavailable = True
        if role in (InputRole.TRAINING_HISTORY, InputRole.PURCHASE_HISTORY):
            if record.latest_event_at is None:
                issues.append(EvidenceIssue(role, "HISTORY_BOUND_UNPROVEN"))
            elif record.latest_event_at >= cutoff:
                issues.append(EvidenceIssue(role, "FUTURE_HISTORY"))
                unavailable = True
        if role is InputRole.FUTURE_LABELS:
            if (record.coverage_start is None or record.coverage_start > cutoff
                    or record.coverage_end < future_end):
                issues.append(EvidenceIssue(role, "INCOMPLETE_LABEL_WINDOW"))
                unavailable = True
            if record.available_at is None or record.available_at < future_end:
                issues.append(EvidenceIssue(role, "LABEL_COMPLETENESS_UNPROVEN"))
        elif record.available_at is None or record.available_at > cutoff:
            issues.append(EvidenceIssue(role, "AS_OF_UNPROVEN"))
        basis = EvidenceBasis.CONTROLLED_SYNTHETIC if synthetic else EvidenceBasis.REVIEWED_POINT_IN_TIME
        if record.review is None or record.review.basis is not basis:
            issues.append(EvidenceIssue(role, "REVIEW_UNPROVEN"))
    status = Validity.UNAVAILABLE if unavailable else Validity.PROXY if issues else Validity.STRICT
    return EvidenceAssessment(status, tuple(issues))
