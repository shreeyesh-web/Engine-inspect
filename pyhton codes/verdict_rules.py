#!/usr/bin/env python3
"""Shared verdict rules for notch-facing-hole detection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Collection, Mapping


CLASS_NAMES = {
    0: "circle",
    1: "notch_toward_hole",
    2: "notch_not_toward_hole",
}

REQUIRED_OK_CLASS_IDS = (0, 1)
INVALID_CLASS_IDS = (2,)


@dataclass(frozen=True)
class VerdictResult:
    verdict: str
    reasons: tuple[str, ...]


def normalize_counts(counts: Mapping[int, int] | None) -> dict[int, int]:
    normalized = {cls_id: 0 for cls_id in CLASS_NAMES}
    if not counts:
        return normalized
    for cls_id in normalized:
        normalized[cls_id] = max(0, int(counts.get(cls_id, 0)))
    return normalized


def strict_verdict_from_counts(counts: Mapping[int, int] | None) -> VerdictResult:
    counts = normalize_counts(counts)
    reasons: list[str] = []

    if counts[2] > 0:
        reasons.append("notch_not_toward_hole detected")
    if counts[0] <= 0:
        reasons.append("circle missing")
    if counts[1] <= 0:
        reasons.append("notch_toward_hole missing")

    verdict = "OK" if not reasons else "NOT_OKAY"
    return VerdictResult(verdict=verdict, reasons=tuple(reasons))


def strict_verdict_from_names(detected_names: Collection[str]) -> VerdictResult:
    detected = {str(name).strip().lower() for name in detected_names if str(name).strip()}
    counts = {
        0: 1 if CLASS_NAMES[0] in detected else 0,
        1: 1 if CLASS_NAMES[1] in detected else 0,
        2: 1 if CLASS_NAMES[2] in detected else 0,
    }
    return strict_verdict_from_counts(counts)


def named_counts(counts: Mapping[int, int] | None) -> dict[str, int]:
    normalized = normalize_counts(counts)
    return {CLASS_NAMES[cls_id]: normalized[cls_id] for cls_id in sorted(CLASS_NAMES)}

