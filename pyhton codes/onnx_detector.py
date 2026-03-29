from __future__ import annotations

import ast
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import cv2
import numpy as np
import onnxruntime as ort


@dataclass(frozen=True)
class Detection:
    cls_id: int
    confidence: float
    xyxy: tuple[int, int, int, int]


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


def extract_model_names(model: ort.InferenceSession) -> dict[int, str]:
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


def _resolve_providers(device: str | None) -> list[str]:
    available = set(ort.get_available_providers())
    requested = str(device or "cpu").strip().lower()
    providers: list[str] = []
    if requested not in {"", "cpu", "-1"}:
        for provider in ("CUDAExecutionProvider", "DmlExecutionProvider"):
            if provider in available:
                providers.append(provider)
    providers.append("CPUExecutionProvider")
    return providers


def _letterbox(image: np.ndarray, new_shape: tuple[int, int]) -> tuple[np.ndarray, float, tuple[float, float]]:
    height, width = image.shape[:2]
    target_h, target_w = new_shape
    scale = min(target_w / width, target_h / height)
    resized_w = int(round(width * scale))
    resized_h = int(round(height * scale))

    resized = cv2.resize(image, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)
    pad_w = target_w - resized_w
    pad_h = target_h - resized_h
    left = int(round(pad_w / 2.0 - 0.1))
    right = int(round(pad_w / 2.0 + 0.1))
    top = int(round(pad_h / 2.0 - 0.1))
    bottom = int(round(pad_h / 2.0 + 0.1))
    bordered = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    return bordered, scale, (float(left), float(top))


class OnnxDetector:
    def __init__(self, model_path: Path, device: str | None = None) -> None:
        self.model_path = Path(model_path)
        session_options = ort.SessionOptions()
        session_options.log_severity_level = 3
        self.session = ort.InferenceSession(
            str(self.model_path),
            sess_options=session_options,
            providers=_resolve_providers(device),
        )
        input_meta = self.session.get_inputs()[0]
        self.input_name = input_meta.name
        self.output_name = self.session.get_outputs()[0].name
        self.input_height = int(input_meta.shape[2])
        self.input_width = int(input_meta.shape[3])
        self.class_names = extract_model_names(self.session)
        self.providers = self.session.get_providers()

    def infer(
        self,
        frame: np.ndarray,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
    ) -> list[Detection]:
        if frame is None or frame.size == 0:
            raise ValueError("Input frame is empty.")

        original_h, original_w = frame.shape[:2]
        prepared, scale, pad = _letterbox(frame, (self.input_height, self.input_width))
        rgb = cv2.cvtColor(prepared, cv2.COLOR_BGR2RGB)
        input_tensor = np.expand_dims(np.transpose(rgb.astype(np.float32) / 255.0, (2, 0, 1)), axis=0)

        raw = self.session.run([self.output_name], {self.input_name: input_tensor})[0]
        predictions = np.asarray(raw)
        if predictions.ndim == 3:
            predictions = predictions[0]
        if predictions.ndim != 2:
            raise RuntimeError(f"Unexpected ONNX output shape: {tuple(np.asarray(raw).shape)}")
        if predictions.shape[0] < predictions.shape[1]:
            predictions = predictions.T

        boxes_xywh: list[list[int]] = []
        confidences: list[float] = []
        class_ids: list[int] = []
        boxes_xyxy: list[tuple[int, int, int, int]] = []

        for row in predictions:
            if row.shape[0] <= 4:
                continue
            scores = row[4:]
            class_id = int(np.argmax(scores))
            confidence = float(scores[class_id])
            if confidence < conf_threshold:
                continue

            center_x, center_y, width, height = [float(v) for v in row[:4]]
            x1 = int(round((center_x - width / 2.0 - pad[0]) / scale))
            y1 = int(round((center_y - height / 2.0 - pad[1]) / scale))
            x2 = int(round((center_x + width / 2.0 - pad[0]) / scale))
            y2 = int(round((center_y + height / 2.0 - pad[1]) / scale))

            x1 = max(0, min(original_w - 1, x1))
            y1 = max(0, min(original_h - 1, y1))
            x2 = max(0, min(original_w - 1, x2))
            y2 = max(0, min(original_h - 1, y2))
            if x2 <= x1 or y2 <= y1:
                continue

            boxes_xywh.append([x1, y1, x2 - x1, y2 - y1])
            boxes_xyxy.append((x1, y1, x2, y2))
            confidences.append(confidence)
            class_ids.append(class_id)

        if not boxes_xywh:
            return []

        indices = cv2.dnn.NMSBoxes(boxes_xywh, confidences, conf_threshold, iou_threshold)
        detections: list[Detection] = []
        if len(indices) > 0:
            for idx in np.array(indices).reshape(-1):
                detections.append(
                    Detection(
                        cls_id=class_ids[int(idx)],
                        confidence=confidences[int(idx)],
                        xyxy=boxes_xyxy[int(idx)],
                    )
                )
        return detections
