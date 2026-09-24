"""In-memory training only. No model artifact publication entrypoint."""

from dataclasses import dataclass, field
from collections.abc import Mapping
from types import MappingProxyType

from . import BPRMF as core
from .training_metrics import TrainingRunMetrics
from .training_quality import TrainingQualityReport, evaluate_training_quality


@dataclass(frozen=True)
class ShadowTrainingResult:
    quality_report: TrainingQualityReport
    preparation_summary: Mapping[str, int | float | bool | None]
    training_started: bool
    training_completed: bool
    training_metrics: TrainingRunMetrics | None = None
    error_code: str | None = None
    model: object = field(default=None, repr=False, compare=False)
    report_path: str | None = field(default=None, repr=False)

    def __post_init__(self):
        object.__setattr__(self, "preparation_summary", MappingProxyType(dict(self.preparation_summary)))


def run_shadow_training(cfg, prepared_data, diagnostics, device, *, complete=True, on_quality=None):
    """Evaluate fresh validation/quality; BLOCK never reaches the training core.

    Like the legacy process orchestration, seed before entering the shared loop.
    Exceptions from training become a safe status, never exception text or IDs.
    """
    quality = evaluate_training_quality(prepared_data, diagnostics)
    summary = dict(quality.metrics)
    summary["complete"] = complete
    summary["total_train_weight"] = (float(prepared_data.splits.train_weights.sum())
                                      if quality.metrics["train_pairs"] is not None else None)
    if on_quality is not None:
        on_quality(quality)
    if not quality.training_allowed:
        return ShadowTrainingResult(quality, summary, False, False, error_code="QUALITY_BLOCK")
    try:
        core._set_seed(cfg.seed)
        model, _splits, metrics = core.train_prepared_data_with_metrics(cfg, prepared_data, device)
    except Exception:
        return ShadowTrainingResult(quality, summary, True, False, error_code="TRAINING_FAILED")
    return ShadowTrainingResult(quality, summary, True, True, metrics, model=model)
