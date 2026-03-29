from __future__ import annotations

import ast
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import cv2
import numpy as np
import onnxruntime as ort

from verdict_rules import strict_verdict_from_names

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = ROOT / "models" / "best.onnx"
DEFAULT_SAVE_DIR = ROOT / "predictions"

CLASS_NAMES = {
    0: "Circle",
    1: "Notch Toward Hole",
    2: "Notch Not Toward Hole",
}
CLASS_COLORS = {
    0: (84, 193, 255),
    1: (26, 122, 60),
    2: (192, 57, 43),
}
CLASS_ORDER = tuple(sorted(CLASS_NAMES))
CANONICAL_CLASS_NAMES = tuple(CLASS_NAMES[cid] for cid in CLASS_ORDER)
INVALID_CLASS_NAME = "Notch Not Toward Hole"
CLASS_ALIASES = {
    "Circle": {"circle", "circles", "hole", "holes"},
    "Notch Toward Hole": {
        "notchtowardhole",
        "notch_toward_hole",
        "toward",
        "towards",
        "notchtohole",
        "pointingtowardhole",
    },
    "Notch Not Toward Hole": {
        "notchnottowardhole",
        "notch_not_toward_hole",
        "away",
        "notchaway",
        "notpointingtowardhole",
        "nottoward",
    },
}


def _normalize_label(name: str) -> str:
    return "".join(ch for ch in name.lower() if ch.isalnum())


_ALIAS_TO_CANONICAL: dict[str, str] = {}
for _canonical, _aliases in CLASS_ALIASES.items():
    for _alias in set(_aliases) | {_canonical}:
        _ALIAS_TO_CANONICAL[_normalize_label(_alias)] = _canonical


def _canonical_from_label(name: str) -> str | None:
    normalized = _normalize_label(name)
    if not normalized:
        return None
    if normalized in _ALIAS_TO_CANONICAL:
        return _ALIAS_TO_CANONICAL[normalized]
    for alias, canonical in _ALIAS_TO_CANONICAL.items():
        if alias and alias in normalized:
            return canonical
    return None


def _iter_model_names(
    model_names: Mapping[int, str] | Sequence[str] | None,
) -> list[tuple[int, str]]:
    if model_names is None:
        return []
    if isinstance(model_names, Mapping):
        out: list[tuple[int, str]] = []
        for raw_id, raw_name in model_names.items():
            try:
                cid = int(raw_id)
            except (TypeError, ValueError):
                continue
            out.append((cid, str(raw_name)))
        return sorted(out)
    if isinstance(model_names, Sequence) and not isinstance(model_names, (str, bytes)):
        return [(cid, str(name)) for cid, name in enumerate(model_names)]
    return []


@dataclass(frozen=True)
class ClassResolver:
    id_to_display_name: dict[int, str]
    id_to_canonical_name: dict[int, str]

    def display_name(self, class_id: int, detected_ids: set[int] | None = None) -> str:
        _ = detected_ids
        return self.id_to_canonical_name.get(class_id, self.id_to_display_name.get(class_id, str(class_id)))

    def label_lookup_for_detection(self, detected_ids: set[int]) -> dict[int, str]:
        _ = detected_ids
        return {cid: self.display_name(cid) for cid in self.id_to_display_name}

    def detected_canonical(self, detected_ids: set[int]) -> set[str]:
        return {
            self.id_to_canonical_name[cid]
            for cid in detected_ids
            if cid in self.id_to_canonical_name
        }

    def present_absent(self, detected_ids: set[int]) -> tuple[list[str], list[str]]:
        detected_canonical = self.detected_canonical(detected_ids)
        tracked = self.tracked_class_names()
        present = [name for name in tracked if name in detected_canonical]
        absent = [name for name in tracked if name not in detected_canonical]
        return present, absent

    def tracked_class_names(self) -> list[str]:
        tracked = list(dict.fromkeys(self.id_to_canonical_name.values()))
        if not tracked:
            return list(CANONICAL_CLASS_NAMES)
        ordered = [name for name in CANONICAL_CLASS_NAMES if name in tracked]
        extras = sorted(name for name in tracked if name not in CANONICAL_CLASS_NAMES)
        return ordered + extras


def build_class_resolver(
    model_names: Mapping[int, str] | Sequence[str] | None = None,
) -> ClassResolver:
    model_items = _iter_model_names(model_names)
    if not model_items:
        default_map = {cid: CLASS_NAMES[cid] for cid in CLASS_ORDER}
        return ClassResolver(dict(default_map), dict(default_map))

    id_to_display: dict[int, str] = {}
    id_to_canonical: dict[int, str] = {}
    mapped_canonical: set[str] = set()

    for cid, raw_name in model_items:
        name = raw_name.strip() or str(cid)
        canonical = _canonical_from_label(name)
        if canonical:
            id_to_canonical[cid] = canonical
            id_to_display[cid] = canonical
            mapped_canonical.add(canonical)
        else:
            id_to_display[cid] = name

    for cid in CLASS_ORDER:
        canonical = CLASS_NAMES[cid]
        if canonical in mapped_canonical:
            continue
        if cid in id_to_canonical:
            continue
        if cid in id_to_display:
            id_to_canonical[cid] = canonical
            id_to_display[cid] = canonical
            mapped_canonical.add(canonical)

    return ClassResolver(id_to_display, id_to_canonical)


DEFAULT_CLASS_RESOLVER = build_class_resolver()


def _coerce_name_mapping(raw: object) -> dict[int, str]:
    out: dict[int, str] = {}
    if isinstance(raw, Mapping):
        for key, value in raw.items():
            try:
                cid = int(key)
            except (TypeError, ValueError):
                continue
            out[cid] = str(value)
        return out
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        for cid, value in enumerate(raw):
            out[cid] = str(value)
        return out
    return out


def extract_model_names_from_ort_session(model: ort.InferenceSession) -> dict[int, str]:
    try:
        metadata = model.get_modelmeta().custom_metadata_map or {}
    except Exception:
        return {}

    candidates = (
        metadata.get("names"),
        metadata.get("class_names"),
        metadata.get("classes"),
        metadata.get("labels"),
    )
    for raw in candidates:
        if raw is None:
            continue
        parsed = _coerce_name_mapping(raw)
        if parsed:
            return parsed
        if isinstance(raw, str):
            text = raw.strip()
            if not text:
                continue
            for parser in (json.loads, ast.literal_eval):
                try:
                    payload = parser(text)
                except Exception:
                    continue
                parsed = _coerce_name_mapping(payload)
                if parsed:
                    return parsed
    return {}


def evaluate(
    detected_ids: set[int],
    resolver: ClassResolver | None = None,
) -> tuple[str, list[str]]:
    resolver = resolver or DEFAULT_CLASS_RESOLVER
    detected_canonical = resolver.detected_canonical(detected_ids)
    verdict = strict_verdict_from_names(detected_canonical)
    if verdict.verdict == "OK":
        return "VALID", []
    return "INVALID", list(verdict.reasons)


def draw_results(
    image: np.ndarray,
    boxes,
    verdict: str,
    reasons: list[str],
    present_names: list[str],
    absent_names: list[str],
    class_name_lookup: Mapping[int, str] | None = None,
    tracked_names: Sequence[str] | None = None,
) -> np.ndarray:
    output = image.copy()
    names = class_name_lookup or CLASS_NAMES
    canonical_colors = {CLASS_NAMES[cid]: CLASS_COLORS[cid] for cid in CLASS_ORDER}

    for box in (boxes or []):
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        disp_name = str(names.get(cls_id, cls_id))
        canonical = _canonical_from_label(disp_name)
        color = canonical_colors.get(canonical or "", CLASS_COLORS.get(cls_id, (128, 128, 128)))
        label = f"{disp_name} {conf:.2f}"

        cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.56, 1)
        cv2.rectangle(output, (x1, y1 - th - 7), (x1 + tw + 5, y1), color, -1)
        cv2.putText(output, label, (x1 + 2, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (255, 255, 255), 1, cv2.LINE_AA)

    font = cv2.FONT_HERSHEY_SIMPLEX
    panel_margin = 16
    panel_pad = 14
    panel_width = min(560, max(420, output.shape[1] // 2))
    panel_width = min(panel_width, output.shape[1] - 2 * panel_margin)
    header_h = 64
    line_h = 36

    reason_lines = reasons if reasons else ["No defect detected"]
    status_rows = []
    present_set = set(present_names)
    names_for_status = list(tracked_names) if tracked_names else list(CANONICAL_CLASS_NAMES)
    for cname in names_for_status:
        is_present = cname in present_set
        status_rows.append((cname, "Present" if is_present else "Not Present", is_present))

    body_rows = len(reason_lines) + 1 + len(status_rows)
    panel_height = header_h + panel_pad + body_rows * line_h + panel_pad
    panel_height = min(panel_height, output.shape[0] - 2 * panel_margin)

    x1 = output.shape[1] - panel_margin - panel_width
    y1 = panel_margin
    x2 = x1 + panel_width
    y2 = y1 + panel_height

    overlay = output.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (25, 25, 25), -1)
    cv2.addWeighted(overlay, 0.72, output, 0.28, 0, output)

    header_color = (0, 170, 0) if verdict == "VALID" else (0, 0, 210)
    cv2.rectangle(output, (x1, y1), (x2, y1 + header_h), header_color, -1)
    cv2.rectangle(output, (x1, y1), (x2, y2), (235, 235, 235), 2)

    cv2.putText(output, f"RESULT: {verdict}", (x1 + panel_pad, y1 + 42), font, 1.10, (255, 255, 255), 2, cv2.LINE_AA)

    y = y1 + header_h + panel_pad + 8
    for reason in reason_lines:
        cv2.putText(output, f"Reason: {reason}", (x1 + panel_pad, y), font, 0.82, (255, 255, 255), 2, cv2.LINE_AA)
        y += line_h

    cv2.putText(output, "Class Presence", (x1 + panel_pad, y), font, 0.86, (200, 230, 255), 2, cv2.LINE_AA)
    y += line_h

    for cname, status, is_present in status_rows:
        status_color = (80, 230, 120) if is_present and cname != INVALID_CLASS_NAME else (90, 170, 255)
        if is_present and cname == INVALID_CLASS_NAME:
            status_color = (80, 80, 230)
        cv2.putText(output, f"{cname}:", (x1 + panel_pad, y), font, 0.82, (255, 255, 255), 2, cv2.LINE_AA)
        (sw, _), _ = cv2.getTextSize(status, font, 0.82, 2)
        cv2.putText(output, status, (x2 - panel_pad - sw, y), font, 0.82, status_color, 2, cv2.LINE_AA)
        y += line_h

    return output
