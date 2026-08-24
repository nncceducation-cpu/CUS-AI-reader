from __future__ import annotations

import hashlib
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
    model_type: str = "onnx_feature"
    prototype_file: str | None = None
    preprocessing: str = "grayscale_resize"
    training_studies: int | None = None
    training_infants: int | None = None
    onnx_sha256: str | None = None
    prototype_sha256: str | None = None

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
            model_type=str(raw.get("model_type", "onnx_feature")),
            prototype_file=raw.get("prototype_file"),
            preprocessing=str(raw.get("preprocessing", "grayscale_resize")),
            training_studies=(int(raw["training_studies"]) if raw.get("training_studies") is not None else None),
            training_infants=(int(raw["training_infants"]) if raw.get("training_infants") is not None else None),
            onnx_sha256=raw.get("onnx_sha256"),
            prototype_sha256=raw.get("prototype_sha256"),
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
    prediction_mode: str = "model_inference"
    matched_training_studies: list[str] = field(default_factory=list)
    nearest_references: list[dict[str, Any]] = field(default_factory=list)
    domain_predictions: dict[str, dict[str, Any]] = field(default_factory=dict)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
    elif manifest.onnx_sha256 and _sha256(model_path) != manifest.onnx_sha256.lower():
        warnings.append(f"Model weights failed SHA256 verification: {model_path.name}")
    if manifest.model_type == "pilot_similarity":
        if not manifest.prototype_file:
            warnings.append("Pilot similarity manifest does not name a prototype file.")
        elif not (prototype_path := root / manifest.prototype_file).exists():
            warnings.append(f"Pilot prototype file is missing: {manifest.prototype_file}")
        elif manifest.prototype_sha256 and _sha256(prototype_path) != manifest.prototype_sha256.lower():
            warnings.append(f"Pilot prototype failed SHA256 verification: {manifest.prototype_file}")
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

    def predict(
        self, frames: list[MediaFrame], source_hashes: list[str] | None = None
    ) -> StudyPrediction:
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


class PilotSimilarityModel:
    """Unvalidated every-frame transfer-learning feasibility model.

    A compact ImageNet-pretrained MobileNetV3 encoder processes every frame. A
    pilot plane head estimates coronal versus sagittal orientation, and an
    examination signature is compared with 15 provisional reference studies.
    Exact pilot-file matches exclude all studies from the same infant before
    voting. This adapter is for workflow testing, not clinical use.
    """

    unknown_values = {"", "not_reported", "indeterminate"}

    def __init__(self, model_dir: str | Path, manifest: ModelManifest):
        try:
            import cv2
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError("Pilot inference requires OpenCV and onnxruntime.") from exc
        if not manifest.prototype_file:
            raise RuntimeError("Pilot model manifest does not name a prototype file.")
        self.cv2 = cv2
        self.root = Path(model_dir)
        self.manifest = manifest
        self.session = ort.InferenceSession(
            str(self.root / manifest.onnx_file), providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name
        payload = np.load(self.root / manifest.prototype_file, allow_pickle=False)
        self.signatures = np.asarray(payload["signatures"], dtype=np.float32)
        self.study_codes = np.asarray(payload["study_codes"]).astype(str)
        self.infant_codes = np.asarray(payload["infant_codes"]).astype(str)
        self.file_sha256 = np.asarray(payload["file_sha256"]).astype(str)
        self.plane_weight = np.asarray(payload["plane_weight"], dtype=np.float32)
        self.plane_bias = np.asarray(payload["plane_bias"], dtype=np.float32)
        self.domain_fields = np.asarray(payload["domain_fields"]).astype(str)
        self.domain_values = np.asarray(payload["domain_values"]).astype(str)
        self.temperature = float(payload["temperature"].item())
        self.k_neighbors = int(payload["k_neighbors"].item())
        self.similarity_floor = float(payload["similarity_floor"].item())

    @staticmethod
    def _l2_rows(values: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(values, axis=1, keepdims=True)
        return values / np.maximum(norms, 1e-8)

    @staticmethod
    def _softmax(values: np.ndarray) -> np.ndarray:
        shifted = values - values.max(axis=1, keepdims=True)
        exponentiated = np.exp(shifted)
        return exponentiated / exponentiated.sum(axis=1, keepdims=True)

    def _detect_roi(self, image: Image.Image) -> tuple[int, int, int, int] | None:
        gray = np.asarray(ImageOps.grayscale(image), dtype=np.uint8)
        mask = (gray > 8).astype(np.uint8)
        kernel = np.ones((7, 7), dtype=np.uint8)
        mask = self.cv2.morphologyEx(mask, self.cv2.MORPH_CLOSE, kernel, iterations=2)
        count, _, stats, _ = self.cv2.connectedComponentsWithStats(mask, connectivity=8)
        if count <= 1:
            return None
        total_area = gray.shape[0] * gray.shape[1]
        candidates: list[tuple[int, int, int, int, int]] = []
        for index in range(1, count):
            x, y, width, height, area = stats[index]
            if area >= total_area * 0.02 and width >= gray.shape[1] * 0.20 and height >= gray.shape[0] * 0.20:
                candidates.append((int(area), int(x), int(y), int(width), int(height)))
        if not candidates:
            return None
        _, x, y, width, height = max(candidates)
        pad_x = max(4, round(width * 0.04))
        pad_y = max(4, round(height * 0.04))
        return (
            max(0, x - pad_x),
            max(0, y - pad_y),
            min(gray.shape[1], x + width + pad_x),
            min(gray.shape[0], y + height + pad_y),
        )

    def _preprocess(self, image: Image.Image, roi_box: tuple[int, int, int, int] | None) -> np.ndarray:
        gray = np.asarray(ImageOps.grayscale(image), dtype=np.uint8)
        if roi_box is not None:
            x0, y0, x1, y1 = roi_box
            gray = gray[y0:y1, x0:x1]
        side = max(gray.shape)
        canvas = np.zeros((side, side), dtype=np.uint8)
        y = (side - gray.shape[0]) // 2
        x = (side - gray.shape[1]) // 2
        canvas[y : y + gray.shape[0], x : x + gray.shape[1]] = gray
        height, width = self.manifest.input_size
        resized = self.cv2.resize(canvas, (width, height), interpolation=self.cv2.INTER_AREA)
        rgb = np.repeat(resized[:, :, None], 3, axis=2).astype(np.float32) / 255.0
        mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
        return np.transpose((rgb - mean) / std, (2, 0, 1))

    def _run_all_frames(self, frames: list[MediaFrame]) -> np.ndarray:
        boxes: dict[str, tuple[int, int, int, int] | None] = {}
        for frame in frames:
            if frame.source_name not in boxes:
                boxes[frame.source_name] = self._detect_roi(frame.image)
        batches: list[np.ndarray] = []
        for start in range(0, len(frames), self.manifest.batch_size):
            group = frames[start : start + self.manifest.batch_size]
            batch = np.stack(
                [self._preprocess(frame.image, boxes.get(frame.source_name)) for frame in group]
            )
            output = np.asarray(
                self.session.run(None, {self.input_name: batch})[0], dtype=np.float32
            )
            batches.append(output.reshape(len(group), -1))
        return np.vstack(batches)

    def _make_signature(self, embeddings: np.ndarray) -> np.ndarray:
        normalized = self._l2_rows(embeddings)
        signature = np.concatenate([normalized.mean(axis=0), normalized.std(axis=0)])
        return signature / max(float(np.linalg.norm(signature)), 1e-8)

    def _vote_domains(
        self, signature: np.ndarray, allowed: np.ndarray
    ) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
        similarities = self.signatures @ signature
        allowed_indices = np.flatnonzero(allowed)
        if not len(allowed_indices):
            raise ValueError("No pilot reference examination remains after infant exclusion.")
        nearest_order = allowed_indices[np.argsort(similarities[allowed_indices])[::-1]][:5]
        nearest = [
            {
                "study_code": str(self.study_codes[index]),
                "infant_code": str(self.infant_codes[index]),
                "similarity": float(similarities[index]),
            }
            for index in nearest_order
        ]
        domains: dict[str, dict[str, Any]] = {}
        for domain_index, domain in enumerate(self.domain_fields):
            candidates = np.asarray(
                [
                    index
                    for index in allowed_indices
                    if self.domain_values[index, domain_index] not in self.unknown_values
                ],
                dtype=np.int64,
            )
            if not len(candidates):
                domains[domain] = {
                    "answer": "indeterminate",
                    "probabilities": {},
                    "reason": "no observed reference labels available",
                }
                continue
            distinct = sorted(set(self.domain_values[candidates, domain_index].tolist()))
            if len(distinct) < 2:
                domains[domain] = {
                    "answer": "indeterminate",
                    "probabilities": {},
                    "reason": "only one observed category is represented",
                }
                continue
            ordered = candidates[np.argsort(similarities[candidates])[::-1]][: self.k_neighbors]
            selected_similarity = similarities[ordered]
            weights = np.exp(
                (selected_similarity - selected_similarity.max()) / max(self.temperature, 1e-4)
            )
            votes: dict[str, float] = {}
            for index, weight in zip(ordered, weights):
                value = str(self.domain_values[index, domain_index])
                votes[value] = votes.get(value, 0.0) + float(weight)
            total = sum(votes.values())
            probabilities = {name: value / total for name, value in votes.items()}
            ranked = sorted(probabilities.items(), key=lambda item: item[1], reverse=True)
            answer = ranked[0][0]
            margin = ranked[0][1] - (ranked[1][1] if len(ranked) > 1 else 0.0)
            top_similarity = float(selected_similarity.max())
            reason = None
            if top_similarity < self.similarity_floor:
                answer = "indeterminate"
                reason = "closest reference is below the patient-held-out similarity floor"
            elif ranked[0][1] < 0.55 or margin < 0.10:
                answer = "indeterminate"
                reason = "reference vote is uncertain"
            domains[domain] = {
                "answer": answer,
                "probabilities": probabilities,
                "top_similarity": top_similarity,
                "confidence": float(ranked[0][1]),
                "margin": float(margin),
                **({"reason": reason} if reason else {}),
            }
        return domains, nearest

    @staticmethod
    def _sum_probability(distribution: dict[str, float], values: set[str]) -> float | None:
        if not distribution:
            return None
        return float(sum(distribution.get(value, 0.0) for value in values))

    def _domains_to_features(self, domains: dict[str, dict[str, Any]]) -> dict[str, float]:
        output: dict[str, float] = {}
        for side in ("left", "right"):
            gmh = domains[f"{side}_gmh_ivh"]["probabilities"]
            if gmh:
                output[f"{side}_hemorrhage_present"] = self._sum_probability(
                    gmh, {"grade_1", "grade_2", "grade_3"}
                ) or 0.0
                output[f"{side}_confined_to_germinal_matrix"] = float(gmh.get("grade_1", 0.0))
                output[f"{side}_intraventricular_blood"] = self._sum_probability(
                    gmh, {"grade_2", "grade_3"}
                ) or 0.0
                output[f"{side}_ventricular_distension"] = float(gmh.get("grade_3", 0.0))
                output[f"{side}_ahw_above_6_mm"] = float(gmh.get("grade_3", 0.0))
            pvhi = domains[f"{side}_pvhi"]["probabilities"]
            if pvhi:
                output[f"{side}_focal_periventricular_echogenicity"] = self._sum_probability(
                    pvhi, {"present", "evolved"}
                ) or 0.0
            cyst = domains[f"{side}_porencephalic_cyst"]["probabilities"]
            if cyst:
                output[f"{side}_porencephalic_cyst"] = float(cyst.get("present", 0.0))
        wmi = domains["wmi"]["probabilities"]
        for value in ("none", "pve_under_7_days", "grade_1", "grade_2", "grade_3", "grade_4"):
            if wmi:
                output[f"wmi_{value}"] = float(wmi.get(value, 0.0))
        cbh = domains["cerebellar_hemorrhage"]["probabilities"]
        for value in ("none", "punctate", "limited", "large"):
            if cbh:
                output[f"cbh_{value}"] = float(cbh.get(value, 0.0))
        phvd = domains["phvd"]["probabilities"]
        if phvd:
            output["vi_above_97th"] = self._sum_probability(phvd, {"moderate", "severe"}) or 0.0
            output["vi_above_97th_plus_4_mm"] = float(phvd.get("severe", 0.0))
        severe = domains["severe_preterm_brain_injury"]["probabilities"]
        if severe:
            output["severe_preterm_brain_injury"] = float(severe.get("yes", 0.0))
        return output

    def predict(
        self, frames: list[MediaFrame], source_hashes: list[str] | None = None
    ) -> StudyPrediction:
        if not frames:
            raise ValueError("At least one frame is required for inference.")
        embeddings = self._run_all_frames(frames)
        normalized = self._l2_rows(embeddings)
        plane_probabilities = self._softmax(normalized @ self.plane_weight + self.plane_bias)
        plane_counts = {
            "coronal": 0,
            "sagittal": 0,
            "posterior_fossa": 0,
            "other": 0,
            "indeterminate": 0,
        }
        frame_predictions: list[FramePrediction] = []
        for frame, values in zip(frames, plane_probabilities):
            best = int(np.argmax(values))
            best_probability = float(values[best])
            margin = float(abs(values[0] - values[1]))
            ambiguous = (
                best_probability < self.manifest.plane_confidence_threshold
                or margin < self.manifest.plane_margin
            )
            plane = "indeterminate" if ambiguous else ("coronal" if best == 0 else "sagittal")
            plane_counts[plane] += 1
            frame_predictions.append(
                FramePrediction(
                    source_name=frame.source_name,
                    frame_index=frame.frame_index,
                    plane=plane,
                    plane_confidence=best_probability,
                    ambiguous_plane=ambiguous,
                    probabilities={
                        "plane_coronal": float(values[0]),
                        "plane_sagittal": float(values[1]),
                    },
                )
            )
        signature = self._make_signature(embeddings)
        supplied_hashes = set(source_hashes or [])
        matched_indices = np.asarray(
            [index for index, digest in enumerate(self.file_sha256) if digest in supplied_hashes],
            dtype=np.int64,
        )
        allowed = np.ones(len(self.study_codes), dtype=bool)
        mode = "full_pilot_reference_fit"
        matched_studies: list[str] = []
        if len(matched_indices):
            matched_infants = set(self.infant_codes[matched_indices].tolist())
            allowed = np.asarray([infant not in matched_infants for infant in self.infant_codes])
            matched_studies = self.study_codes[matched_indices].tolist()
            mode = "patient_held_out_for_known_pilot_media"
        domains, nearest = self._vote_domains(signature, allowed)
        probabilities = self._domains_to_features(domains)
        reasons = ["manifest marks model as unvalidated"]
        if plane_counts["coronal"] == 0:
            reasons.append("no accepted coronal frame")
        if plane_counts["sagittal"] == 0:
            reasons.append("no accepted sagittal frame")
        uncertain = [name for name, value in domains.items() if value["answer"] == "indeterminate"]
        if uncertain:
            reasons.append("uncertain pilot domains: " + ", ".join(uncertain))
        if mode == "full_pilot_reference_fit":
            reasons.append("new-media prediction uses all 15 provisional reference examinations")
        else:
            reasons.append("known pilot media was graded with all scans from that infant excluded")
        return StudyPrediction(
            probabilities=probabilities,
            probabilities_by_plane={},
            plane_counts=plane_counts,
            frame_predictions=frame_predictions,
            processed_frame_count=len(frames),
            model_id=self.manifest.model_id,
            model_version=self.manifest.version,
            abstained=True,
            abstention_reasons=reasons,
            calibration_applied=False,
            aggregation_mode="pilot_similarity_vote",
            prediction_mode=mode,
            matched_training_studies=matched_studies,
            nearest_references=nearest,
            domain_predictions=domains,
        )


def load_model(
    model_dir: str | Path, manifest: ModelManifest
) -> OnnxFeatureModel | PilotSimilarityModel:
    if manifest.model_type == "pilot_similarity":
        return PilotSimilarityModel(model_dir, manifest)
    if manifest.model_type != "onnx_feature":
        raise RuntimeError(f"Unsupported model_type: {manifest.model_type}")
    return OnnxFeatureModel(model_dir, manifest)


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
        "prediction_mode": prediction.prediction_mode,
        "matched_training_studies": prediction.matched_training_studies,
        "nearest_references": prediction.nearest_references,
        "domain_predictions": prediction.domain_predictions,
    }
