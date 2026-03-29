from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AppProfile:
    key: str
    app_title: str
    build_name: str
    settings_filename: str
    output_dirname: str
    crash_log_dirname: str
    default_confidence: float
    default_class_names: dict[int, str]
    default_model_relpath: str
    invalid_present_class_names: frozenset[str]

    @property
    def tracked_class_names(self) -> list[str]:
        return list(self.default_class_names.values())

    def source_model_path(self, root: Path) -> Path:
        return root / Path(self.default_model_relpath)


PROFILES: dict[str, AppProfile] = {
    "jd_block": AppProfile(
        key="jd_block",
        app_title="Harmony JD Block Inspector",
        build_name="Harmony JD block v1.2",
        settings_filename="settings_jd_block.json",
        output_dirname="captures_jd_block",
        crash_log_dirname="HarmonyJDBlockInspector",
        default_confidence=0.25,
        default_class_names={
            0: "Circle",
            1: "Notch Toward Hole",
            2: "Notch Not Toward Hole",
        },
        default_model_relpath="runs/detect/runs/notch_face_v2/weights/best.onnx",
        invalid_present_class_names=frozenset({"Notch Not Toward Hole"}),
    ),
}

DEFAULT_PROFILE_KEY = "jd_block"

_ALIASES = {
    "jd": "jd_block",
    "jdblock": "jd_block",
    "jd_block": "jd_block",
}


def resolve_profile(raw: str | None = None) -> AppProfile:
    text = (raw or os.getenv("BROACH_APP_PROFILE") or DEFAULT_PROFILE_KEY).strip().lower()
    normalized = "".join(ch for ch in text if ch.isalnum() or ch == "_")
    key = _ALIASES.get(normalized, normalized)
    return PROFILES.get(key, PROFILES[DEFAULT_PROFILE_KEY])
