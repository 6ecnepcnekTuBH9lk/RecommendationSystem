"""Small export contracts; no Qt, source I/O or model initialization."""

from dataclasses import dataclass, field
from typing import Any


class ExportError(Exception):
    """Static safe code only: never attach a source record or OS error text."""

    def __init__(self, code):
        self.code = code
        super().__init__(code)


class ExportCancelled(ExportError):
    def __init__(self):
        super().__init__("CANCELLED")


@dataclass(frozen=True, slots=True, repr=False)
class ExportCustomer:
    canonical_id: str
    contact: Any
    phone_confirmed: bool | None
    phone_invalid: bool | None
    hybrid: Any
    propensity: Any
    data_status: str = "READY"


@dataclass(frozen=True, slots=True)
class ExportMetadata:
    audience: str
    total_candidates: int
    model_users: int
    propensity_rows: int
    provenance: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True, slots=True)
class ExportSummary:
    checked: int
    customers: int
    online_rows: int
    mindbox_rows: int
    raw_rows: int
    hybrid_rows: int
    zip_bytes: int
    elapsed_seconds: float
    skipped_profiles: int = 0
    excluded_phone_collisions: int = 0
    statuses: dict = field(default_factory=dict)
