"""Source-neutral quality policy over prepared input and safe counters."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
import math
from types import MappingProxyType

from .training_data import PreparedBprData, PreparedDataError, validate_prepared_data


class QualityLevel(str, Enum):
    PASS = "PASS"
    WARN = "WARN"
    BLOCK = "BLOCK"


@dataclass(frozen=True)
class TrainingQualityConfig:
    mapped_action_malformed_warn_rate: float = 0.0

    def __post_init__(self):
        value = self.mapped_action_malformed_warn_rate
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("Malformed warning rate must be finite and within [0, 1]")


@dataclass(frozen=True)
class TrainingQualityDiagnostics:
    actions_view: int = 0
    actions_favorite: int = 0
    malformed_mapped_actions: int = 0
    unresolved_products: int = 0
    unsupported_products: int = 0
    bpr_events: int = 0
    unmapped_actions: int = 0
    malformed_action_system_names: Mapping[str, int] = field(default_factory=dict)

    def __post_init__(self):
        counts = (self.actions_view, self.actions_favorite, self.malformed_mapped_actions,
                  self.unresolved_products, self.unsupported_products, self.bpr_events, self.unmapped_actions)
        if any(type(value) is not int or value < 0 for value in counts):
            raise ValueError("Quality counters must be non-negative integers")
        if self.malformed_mapped_actions > self.actions_view + self.actions_favorite:
            raise ValueError("Malformed count exceeds mapped action count")
        breakdown = dict(self.malformed_action_system_names)
        if any(not isinstance(name, str) or not name.strip() or type(count) is not int or count < 0
               for name, count in breakdown.items()):
            raise ValueError("Invalid malformed technical-name breakdown")
        if breakdown and sum(breakdown.values()) != self.malformed_mapped_actions:
            raise ValueError("Malformed breakdown does not match total")
        object.__setattr__(self, "malformed_action_system_names", MappingProxyType(breakdown))


@dataclass(frozen=True)
class QualityIssue:
    code: str
    level: QualityLevel
    count: int
    message: str
    rate: float | None = None
    breakdown: Mapping[str, int] = field(default_factory=dict)

    def __post_init__(self):
        object.__setattr__(self, "breakdown", MappingProxyType(dict(self.breakdown)))


@dataclass(frozen=True)
class TrainingQualityReport:
    level: QualityLevel
    issues: tuple[QualityIssue, ...]
    metrics: Mapping[str, int | float | None]

    def __post_init__(self):
        object.__setattr__(self, "issues", tuple(self.issues))
        object.__setattr__(self, "metrics", MappingProxyType(dict(self.metrics)))

    @property
    def training_allowed(self) -> bool:
        return self.level is not QualityLevel.BLOCK


def evaluate_training_quality(
    prepared_data: PreparedBprData, diagnostics: TrainingQualityDiagnostics,
    config: TrainingQualityConfig = TrainingQualityConfig(),
) -> TrainingQualityReport:
    """No transport decision. Caller must supply successful preparation of complete transport.

    Invalid prepared input is BLOCK; invalid diagnostic/config contracts raise.
    The report holds only counts, never a reference to prepared input or its IDs.
    """
    if not isinstance(diagnostics, TrainingQualityDiagnostics) or not isinstance(config, TrainingQualityConfig):
        raise ValueError("Quality diagnostics/config contract required")
    issues = []
    # On invalid input, summary cardinalities are unavailable, not guessed/repaired.
    sizes = {"users": None, "items": None, "train_pairs": None, "eval_events": None}
    try:
        validate_prepared_data(prepared_data)
    except PreparedDataError:
        issues.append(QualityIssue("INVALID_PREPARED_DATA", QualityLevel.BLOCK, 1,
                                   "Prepared input failed structural training validation"))
    else:
        sizes = {"users": len(prepared_data.mappings.idx2user), "items": len(prepared_data.mappings.idx2item),
                 "train_pairs": len(prepared_data.splits.train_pairs), "eval_events": len(prepared_data.splits.eval_users)}
    mapped = diagnostics.actions_view + diagnostics.actions_favorite
    rate = diagnostics.malformed_mapped_actions / mapped if mapped else 0.0
    if diagnostics.malformed_mapped_actions and rate > config.mapped_action_malformed_warn_rate:
        issues.append(QualityIssue("MAPPED_ACTION_WITHOUT_PRODUCT", QualityLevel.WARN,
            diagnostics.malformed_mapped_actions, "Mapped actions without products were excluded", rate,
            diagnostics.malformed_action_system_names))
    for code, count, message in (
        ("UNRESOLVED_PRODUCT", diagnostics.unresolved_products, "Product identity could not be resolved"),
        ("UNSUPPORTED_PRODUCT_NAMESPACE", diagnostics.unsupported_products, "Unsupported product namespace"),
    ):
        if count:
            issues.append(QualityIssue(code, QualityLevel.BLOCK, count, message))
    level = (QualityLevel.BLOCK if any(issue.level is QualityLevel.BLOCK for issue in issues)
             else QualityLevel.WARN if issues else QualityLevel.PASS)
    metrics = {"mapped_actions": mapped, "malformed_mapped_actions": diagnostics.malformed_mapped_actions,
               "malformed_rate": rate, "unresolved_products": diagnostics.unresolved_products,
               "unsupported_products": diagnostics.unsupported_products, "bpr_events": diagnostics.bpr_events,
               "unmapped_actions": diagnostics.unmapped_actions, **sizes}
    return TrainingQualityReport(level, tuple(issues), metrics)
