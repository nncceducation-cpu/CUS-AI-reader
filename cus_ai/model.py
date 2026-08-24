from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps


@dataclass(slots=True)
class ModelManifest:
    model_id: str
    version: str
    onnx_file: str
    input_size: tuple[int, int]
    labels: list[str]
    thresholds: dict[str, float] = field(default_factory=dict)
    validated: bool = False
    intended_use: str = "Research only"

    @classmethod
    def from_path(cls, path: Path) -> "ModelManifest":
        raw = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            model_id=raw["model_id"],
            version=raw["version"],
            onnx_file=raw["onnx_file"],
            input_size=tuple(raw.get("input_size", [512, 512])),
            labels=list(raw["labels"]),
            thresholds=dict(raw.get("thresholds", {})),
            validated=bool(raw.get("validated", False)),
            intended_use=raw.get("intended_use", "Research only"),
        )


@dataclass(slots=True)
class FeaturePrediction:
    probabilities: dict[str, float]
    model_id: str
    model_version: str
    abstained: bool
    abstention_reasons: list[str]


def discover_model(model_dir: str | Path) -> tuple[ModelManifest | None, list[str]]:
    root = Path(model_dir)
    manifests = sorted(root.glob("*.manifest.json"))
    if not manifests:
        return None, ["No model manifest is installed. Clinical AI suggestions are disabled."]
    if len(manifests) > 1:
        return None, ["Multiple model manifests were found. Select one model explicitly before inference."]
    manifest = ModelManifest.from_path(manifests[0])
    model_path = root / manifest.onnx_file
    warnings: list[str] = []
    if not model_path.exists():
        warnings.append(f"Model weights are missing: {model_path.name}")
    if not manifest.validated:
        warnings.append("The installed model manifest is marked unvalidated.")
    return manifest, warnings


class OnnxFeatureModel:
    """Manifest-driven ONNX interface for validated feature models.

    Expected output is one probability per label, with shape [batch, labels].
    The deterministic clinical rule engine remains separate from this adapter.
    """

    def __init__(self, model_dir: str | Path, manifest: ModelManifest):
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError("ONNX inference requires onnxruntime.") from exc
        self.root = Path(model_dir)
        self.manifest = manifest
        self.session = ort.InferenceSession(
            str(self.root / manifest.onnx_file), providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name

    def _preprocess(self, image: Image.Image) -> np.ndarray:
        height, width = self.manifest.input_size
        gray = ImageOps.grayscale(image).resize((width, height))
        array = np.asarray(gray, dtype=np.float32) / 255.0
        return array[None, None, ...]

    def predict(self, images: list[Image.Image]) -> FeaturePrediction:
        if not images:
            raise ValueError("At least one image is required for inference.")
        values = []
        for image in images:
            output = np.asarray(self.session.run(None, {self.input_name: self._preprocess(image)})[0])
            values.append(output.reshape(-1))
        frame_values = np.vstack(values)
        if frame_values.shape[1] != len(self.manifest.labels):
            raise ValueError("Model output width does not match manifest labels.")
        aggregate = np.mean(frame_values, axis=0)
        probabilities = {label: float(value) for label, value in zip(self.manifest.labels, aggregate)}
        reasons: list[str] = []
        if not self.manifest.validated:
            reasons.append("manifest marks model as unvalidated")
        for label, threshold in self.manifest.thresholds.items():
            value = probabilities.get(label)
            if value is not None and abs(value - threshold) < 0.05:
                reasons.append(f"{label} probability is within 0.05 of its threshold")
        return FeaturePrediction(
            probabilities=probabilities,
            model_id=self.manifest.model_id,
            model_version=self.manifest.version,
            abstained=bool(reasons),
            abstention_reasons=reasons,
        )


def prediction_to_json(prediction: FeaturePrediction) -> dict[str, Any]:
    return {
        "model_id": prediction.model_id,
        "model_version": prediction.model_version,
        "probabilities": prediction.probabilities,
        "abstained": prediction.abstained,
        "abstention_reasons": prediction.abstention_reasons,
    }

