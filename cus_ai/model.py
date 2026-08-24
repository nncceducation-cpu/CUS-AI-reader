from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps

from .aggregation import AggregationConfig, LabelEvidence, aggregate_labels, frame_weight
from .calibration import CalibrationSet
from .consistency import enforce_consistency
from .media import MediaFrame, quality_metrics

PLANE_LABELS = {
    "plane_coronal": "coronal",
    "plane_sagittal": "sagittal",
    "plane_posterior_fossa": "posterior_fossa",
    "plane_other": "other",
}
REQUIRED_PLANE_LABELS = {"plane_coronal", "plane_sagittal"}
NON_DIAGNOSTIC_PLANES = {"other", "indeterminate"}


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
    batch_size: int = 16
    plane_confidence_threshold: float = 0.70
    plane_margin: float = 0.10
    decision_margin: float = 0.05
    calibration: CalibrationSet = field(default_factory=CalibrationSet)
    aggregation: AggregationConfig = field(default_factory=AggregationConfig)
    softmax_groups: tuple[str, ...] = ()
    # Retained so that older manifests load, and so that the previous pooling
    # behaviour can be reproduced when auditing a historical result.
    aggregation_top_k: int = 3
    aggregation_mode: str = "persistence"

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
            batch_size=max(1, int(raw.get("batch_size", 16))),
            plane_confidence_threshold=float(raw.get("plane_confidence_threshold", 0.70)),
            plane_margin=float(raw.get("plane_margin", 0.10)),
            decision_margin=max(0.0, float(raw.get("decision_margin", 0.05))),
            calibration=CalibrationSet.from_dict(raw.get("calibration")),
            aggregation=AggregationConfig.from_dict(raw.get("aggregation")),
            softmax_groups=tuple(raw.get("softmax_groups", ())),
            aggregation_top_k=max(1, int(raw.get("aggregation_top_k", 3))),
            aggregation_mode=str(raw.get("aggregation_mode", "persistence")),
        )

    @property
    def diagnostic_labels(self) -> list[str]:
        return [label for label in self.labels if label not in PLANE_LABELS]


@dataclass(slots=True)
class FramePrediction:
    source_name: str
    frame_index: int
    plane: str
    plane_confidence: float
    probabilities: dict[str, float]
    ambiguous_plane: bool
    weight: float = 1.0
    quality_flag: str = "reviewable"


@dataclass(slots=True)
class StudyPrediction:
    probabilities: dict[str, float]
    probabilities_by_plane: dict[str, dict[str, float]]
    plane_counts: dict[str, int]
    frame_predictions: list[FramePrediction]
    processed_frame_count: int
    model_id: str
    model_version: str
    abstained: bool
    abstention_reasons: list[str]
    label_evidence: dict[str, LabelEvidence] = field(default_factory=dict)
    consistency: dict[str, Any] = field(default_factory=dict)
    calibration_applied: bool = False
    aggregation_mode: str = "persistence"


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
    missing_plane_labels = sorted(REQUIRED_PLANE_LABELS.difference(manifest.labels))
    if missing_plane_labels:
        warnings.append("Model manifest is missing required plane labels: " + ", ".join(missing_plane_labels))
    if manifest.calibration.is_identity:
        warnings.append(
            "No probability calibration is installed. Raw network outputs are typically "
            "over-confident, so decision thresholds will not behave as their numbers suggest."
        )
    else:
        uncalibrated = manifest.calibration.uncalibrated_labels(manifest.diagnostic_labels)
        if uncalibrated:
            warnings.append(
                f"{len(uncalibrated)} diagnostic labels have no individual calibration and fall "
                "back to the global temperature."
            )
    return manifest, warnings


class OnnxFeatureModel:
    """Manifest-driven ONNX interface for validated feature models.

    Expected output is one probability per label, with shape [batch, labels].
    The deterministic clinical rule engine remains separate from this adapter.

    The frame to study step is delegated to :mod:`cus_ai.aggregation`, which
    requires a finding to persist across consecutive frames rather than rewarding
    its single best frame.
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

    def _run_all_frames(self, images: list[Image.Image]) -> np.ndarray:
        batches: list[np.ndarray] = []
        for start in range(0, len(images), self.manifest.batch_size):
            batch = np.concatenate(
                [self._preprocess(image) for image in images[start : start + self.manifest.batch_size]], axis=0
            )
            output = np.asarray(self.session.run(None, {self.input_name: batch})[0], dtype=np.float32)
            batches.append(output.reshape(batch.shape[0], -1))
        values = np.vstack(batches)
        if not np.isfinite(values).all() or np.any(values < 0) or np.any(values > 1):
            raise ValueError("Model outputs must be finite probabilities between 0 and 1.")
        return values

    def _assign_plane(self, probabilities: dict[str, float]) -> tuple[str, float, bool]:
        candidates = sorted(
            ((probabilities[label], plane) for label, plane in PLANE_LABELS.items() if label in probabilities),
            reverse=True,
        )
        if not candidates:
            return "indeterminate", 0.0, True
        best_probability, best_plane = candidates[0]
        second_probability = candidates[1][0] if len(candidates) > 1 else 0.0
        ambiguous = (
            best_probability < self.manifest.plane_confidence_threshold
            or best_probability - second_probability < self.manifest.plane_margin
        )
        return ("indeterminate" if ambiguous else best_plane), float(best_probability), ambiguous

    def predict(self, frames: list[MediaFrame]) -> StudyPrediction:
        if not frames:
            raise ValueError("At least one frame is required for inference.")
        frame_values = self._run_all_frames([frame.image for frame in frames])
        if frame_values.shape[1] != len(self.manifest.labels):
            raise ValueError("Model output width does not match manifest labels.")
        return build_study_prediction(self.manifest, frames, frame_values)


def build_study_prediction(
    manifest: ModelManifest,
    frames: list[MediaFrame],
    frame_values: "np.ndarray",
) -> StudyPrediction:
    """Calibrate, plane-gate, aggregate, and repair a batch of frame outputs.

    Kept separate from the ONNX adapter so that the scoring path can be tested,
    replayed, and benchmarked without loading a runtime.
    """
    calibration = manifest.calibration
    plane_counts = {"coronal": 0, "sagittal": 0, "posterior_fossa": 0, "other": 0, "indeterminate": 0}
    frame_predictions: list[FramePrediction] = []
    rows: list[dict[str, Any]] = []

    helper = OnnxFeatureModel.__new__(OnnxFeatureModel)
    helper.manifest = manifest  # type: ignore[attr-defined]

    for frame, values in zip(frames, frame_values):
        raw = {label: float(value) for label, value in zip(manifest.labels, values)}
        row = calibration.calibrate_row(raw)
        plane, confidence, ambiguous = OnnxFeatureModel._assign_plane(helper, row)
        plane_counts[plane] += 1
        quality = quality_metrics(frame.image) if manifest.aggregation.use_quality_weights else None
        weight = frame_weight(confidence, quality, manifest.aggregation.use_quality_weights)
        frame_predictions.append(
            FramePrediction(
                source_name=frame.source_name,
                frame_index=frame.frame_index,
                plane=plane,
                plane_confidence=confidence,
                probabilities=row,
                ambiguous_plane=ambiguous,
                weight=weight,
                quality_flag=str((quality or {}).get("quality_flag", "reviewable")),
            )
        )
        if plane not in NON_DIAGNOSTIC_PLANES:
            rows.append(
                {
                    "plane": plane,
                    "source_name": frame.source_name,
                    "frame_index": frame.frame_index,
                    "weight": weight,
                    "probabilities": row,
                }
            )

    diagnostic_labels = manifest.diagnostic_labels
    evidence = aggregate_labels(rows, diagnostic_labels, manifest.aggregation, manifest.thresholds)
    aggregated = {name: item.probability for name, item in evidence.items()}
    repair = enforce_consistency(aggregated, softmax_groups=manifest.softmax_groups)
    probabilities = repair.probabilities

    probabilities_by_plane: dict[str, dict[str, float]] = {}
    for plane in ("coronal", "sagittal", "posterior_fossa"):
        plane_view = {
            name: item.plane_probabilities[plane]
            for name, item in evidence.items()
            if plane in item.plane_probabilities
        }
        if plane_view:
            probabilities_by_plane[plane] = plane_view

    reasons: list[str] = []
    if not manifest.validated:
        reasons.append("manifest marks model as unvalidated")
    if plane_counts["coronal"] == 0:
        reasons.append("no accepted coronal frame")
    if plane_counts["sagittal"] == 0:
        reasons.append("no accepted sagittal frame")
    if not rows:
        reasons.append("no frame passed the plane gate")
    if repair.conflicts:
        reasons.append("model outputs violated anatomic constraints and were repaired")

    return StudyPrediction(
        probabilities=probabilities,
        probabilities_by_plane=probabilities_by_plane,
        plane_counts=plane_counts,
        frame_predictions=frame_predictions,
        processed_frame_count=len(frame_predictions),
        model_id=manifest.model_id,
        model_version=manifest.version,
        abstained=bool(reasons),
        abstention_reasons=reasons,
        label_evidence=evidence,
        consistency=repair.to_dict(),
        calibration_applied=not calibration.is_identity,
        aggregation_mode=manifest.aggregation_mode,
    )


def prediction_to_json(prediction: StudyPrediction) -> dict[str, Any]:
    return {
        "model_id": prediction.model_id,
        "model_version": prediction.model_version,
        "processed_frame_count": prediction.processed_frame_count,
        "plane_counts": prediction.plane_counts,
        "probabilities": prediction.probabilities,
        "probabilities_by_plane": prediction.probabilities_by_plane,
        "label_evidence": {name: item.to_dict() for name, item in prediction.label_evidence.items()},
        "consistency": prediction.consistency,
        "calibration_applied": prediction.calibration_applied,
        "aggregation_mode": prediction.aggregation_mode,
        "frame_predictions": [
            {
                "source_name": frame.source_name,
                "frame_index": frame.frame_index,
                "plane": frame.plane,
                "plane_confidence": frame.plane_confidence,
                "ambiguous_plane": frame.ambiguous_plane,
                "weight": round(frame.weight, 4),
                "quality_flag": frame.quality_flag,
                "probabilities": frame.probabilities,
            }
            for frame in prediction.frame_predictions
        ],
        "abstained": prediction.abstained,
        "abstention_reasons": prediction.abstention_reasons,
    }
