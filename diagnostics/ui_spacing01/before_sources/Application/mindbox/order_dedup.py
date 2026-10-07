"""Snapshot equality, not a business rule choosing between changed orders."""

from decimal import Decimal
import hashlib
import json

from .adapters._common import identifier


class OrderSnapshotConflict(ValueError):
    """Safe failure: never include an order identity or snapshot."""


def semantic_fingerprint(value):
    digest = hashlib.sha256()

    def emit(item):
        if item is None or isinstance(item, (str, bool)):
            digest.update(json.dumps(item, ensure_ascii=True).encode())
        elif isinstance(item, (int, float, Decimal)):
            number = Decimal(str(item))
            if not number.is_finite():
                raise ValueError("Invalid order number")
            sign, digits, exponent = number.as_tuple()
            digits = list(digits)
            while digits and digits[-1] == 0:
                digits.pop()
                exponent += 1
            digest.update(repr((sign if digits else 0, digits, exponent if digits else 0)).encode())
        elif isinstance(item, dict):
            digest.update(b"{")
            for key in sorted(item):
                emit(key)
                digest.update(b":")
                emit(item[key])
                digest.update(b",")
            digest.update(b"}")
        elif isinstance(item, list):
            digest.update(b"[")
            for element in item:
                emit(element)
                digest.update(b",")
            digest.update(b"]")
        else:
            raise ValueError("Invalid order JSON type")

    emit(value)
    return digest.digest()


class OrderSnapshots:
    def __init__(self):
        self.fingerprints = {}
        self.raw = self.identical = self.conflicting = 0

    def accept(self, raw, *, diagnose=False):
        self.raw += 1
        key = identifier(raw, "ids.mindboxId")
        fingerprint = semantic_fingerprint(raw)
        previous = self.fingerprints.get(key)
        if previous is None:
            self.fingerprints[key] = fingerprint
            return True
        if previous == fingerprint:
            self.identical += 1
        else:
            self.conflicting += 1
            if not diagnose:
                raise OrderSnapshotConflict("Conflicting order snapshots; preparation blocked")
        return False
