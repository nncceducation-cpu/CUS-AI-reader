"""Frame to study aggregation for cranial ultrasound sweeps.

The previous implementation averaged the ``k`` highest frame probabilities for
each label. On a sweep of several hundred frames that is close to a maximum, so
one noisy frame was enough to push a label over its decision threshold. Sweeps
are long, most frames do not contain the lesion, and the ultrasound speckle that
mimics a germinal matrix clot appears on isolated frames. Maximum-style pooling
therefore inflates the false positive rate for every finding at once.

Real lesions behave differently from speckle: they persist. A germinal matrix
clot stays visible across a run of consecutive frames as the probe sweeps
through it, and it reappears in the orthogonal plane. This module scores each
label on that behaviour instead of on its single best frame.

Three statistics are computed per label, per plane:

``order``       the k-th largest calibrated probability, where k is the number
                of frames the finding must be visible on. This is exactly the
                probability level that at least k frames reach, and it reduces
                to the maximum when k is 1.
``run``         the highest probability level p for which some run of at least
                ``min_run`` consecutive frames all reach p. Speckle does not
                survive this; anatomy does.
``prevalence``  the quality-weighted fraction of frames reaching the label
                threshold, reported for audit rather than used for the decision.

The study probability is the more conservative of the order and run statistics.
Frames are weighted by image quality and plane confidence, so a blurred or
off-axis frame contributes less than a clean one.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

DEFAULT_MIN_FRAMES = 2
DEFAULT_PERSISTENCE_FRACTION = 0.02
DEFAULT_MIN_RUN = 2
DEFAULT_PREVALENCE_THRESHOLD = 0.5

# Findings are only visible in the planes that show the relevant anatomy.
# Scoring a cerebellar label from a coronal frontal-horn frame is not evidence,
# it is noise, so each label family is restricted to the planes that can carry it.
DEFAULT_LABEL_PLANE_SCOPE: dict[str, tuple[str, ...]] = {
    "cbh_": ("posterior_fossa",),
    "vi_": ("coronal",),
    "left_ahw_": ("coronal",),
    "right_ahw_": ("coronal",),
    "left_ventricular_distension": ("coronal", "sagittal"),
    "right_ventricular_distension": ("coronal", "sagittal"),
    "wmi_": ("coronal", "sagittal"),
}

SCORABLE_PLANES = ("coronal", "sagittal", "posterior_fossa")


@dataclass(slots=True)
class AggregationConfig:
    """Tunable persistence requirements, normally supplied by the manifest."""

    min_frames: int = DEFAULT_MIN_FRAMES
    persistence_fraction: float = DEFAULT_PERSISTENCE_FRACTION
    min_run: int = DEFAULT_MIN_RUN
    prevalence_threshold: float = DEFAULT_PREVALENCE_THRESHOLD
    use_quality_weights: bool = True
    require_two_planes: bool = True
    label_plane_scope: dict[str, tuple[str, ...]] = field(
        default_factory=lambda: dict(DEFAULT_LABEL_PLANE_SCOPE)
    )

    def __post_init__(self) -> None:
        self.min_frames = max(1, int(self.min_frames))
        self.min_run = max(1, int(self.min_run))
        self.persistence_fraction = min(1.0, max(0.0, float(self.persistence_fraction)))
        self.prevalence_threshold = min(1.0, max(0.0, float(self.prevalence_threshold)))

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> "AggregationConfig":
        raw = raw or {}
        scope = dict(DEFAULT_LABEL_PLANE_SCOPE)
        for prefix, planes in (raw.get("label_plane_scope") or {}).items():
            scope[prefix] = tuple(planes)
        return cls(
            min_frames=int(raw.get("min_frames", DEFAULT_MIN_FRAMES)),
            persistence_fraction=float(
                raw.get("persistence_fraction", DEFAULT_PERSISTENCE_FRACTION)
            ),
            min_run=int(raw.get("min_run", DEFAULT_MIN_RUN)),
            prevalence_threshold=float(
                raw.get("prevalence_threshold", DEFAULT_PREVALENCE_THRESHOLD)
            ),
            use_quality_weights=bool(raw.get("use_quality_weights", True)),
            require_two_planes=bool(raw.get("require_two_planes", True)),
            label_plane_scope=scope,
        )

    def planes_for_label(self, label: str) -> tuple[str, ...]:
        for prefix, planes in self.label_plane_scope.items():
            if label.startswith(prefix):
                return planes
        return SCORABLE_PLANES

    def required_frames(self, frame_count: int) -> int:
        """How many frames a finding must appear on before it counts."""
        scaled = math.ceil(self.persistence_fraction * frame_count)
        return max(1, min(frame_count, max(self.min_frames, int(scaled))))


@dataclass(slots=True)
class LabelEvidence:
    """Aggregated evidence for one label, kept auditable end to end."""

    label: str
    probability: float
    order_statistic: float
    run_statistic: float
    prevalence: float
    frames_considered: int
    frames_required: int
    longest_run: int
    planes_used: list[str]
    plane_probabilities: dict[str, float] = field(default_factory=dict)
    in_scope: bool = True
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "probability": round(self.probability, 6),
            "order_statistic": round(self.order_statistic, 6),
            "run_statistic": round(self.run_statistic, 6),
            "prevalence": round(self.prevalence, 6),
            "frames_considered": self.frames_considered,
            "frames_required": self.frames_required,
            "longest_run": self.longest_run,
            "planes_used": list(self.planes_used),
            "plane_probabilities": {k: round(v, 6) for k, v in self.plane_probabilities.items()},
            "in_scope": self.in_scope,
            "notes": list(self.notes),
        }


def frame_weight(
    plane_confidence: float, quality: dict[str, Any] | None, use_quality: bool = True
) -> float:
    """Down-weight frames that are off-axis, blurred, washed out, or clipped."""
    weight = max(0.0, min(1.0, float(plane_confidence)))
    if not use_quality or not quality:
        return weight
    flag = str(quality.get("quality_flag", "reviewable"))
    penalty = {
        "reviewable": 1.0,
        "possible blur": 0.55,
        "low contrast": 0.5,
        "low resolution": 0.4,
    }.get(flag, 0.7)
    clipped = float(quality.get("clipped_fraction", 0.0) or 0.0)
    if clipped > 0.35:
        penalty *= 0.6
    return weight * penalty


def _order_statistic(values: Sequence[float], k: int) -> float:
    """The k-th largest value: the level that at least k frames reach."""
    if not values:
        return 0.0
    ordered = sorted(values, reverse=True)
    index = min(len(ordered), max(1, k)) - 1
    return float(ordered[index])


def _run_statistic(values: Sequence[float], min_run: int) -> tuple[float, int]:
    """Highest level sustained across a run of consecutive frames.

    Returns the level and the longest run observed at that level. Computed by
    sweeping candidate levels drawn from the observed values, which keeps the
    result exact without a continuous search.
    """
    if not values:
        return 0.0, 0
    # A study exported as still images has one frame per plane. Demanding a run
    # of consecutive frames there would zero every label, so the requirement
    # relaxes to what the acquisition can supply.
    run_length = max(1, min(int(min_run), len(values)))
    best_level = 0.0
    best_run = 0
    for level in sorted(set(values), reverse=True):
        current = 0
        longest = 0
        for value in values:
            current = current + 1 if value >= level else 0
            longest = max(longest, current)
        if longest >= run_length:
            best_level = float(level)
            best_run = longest
            break
        best_run = max(best_run, longest)
    return best_level, best_run


def _longest_run_above(values: Sequence[float], threshold: float) -> int:
    """How many consecutive frames actually showed the finding.

    Reported for the reader rather than used in the score: "visible on 18
    consecutive frames" is the sentence a sonographer can check against the clip.
    """
    longest = current = 0
    for value in values:
        current = current + 1 if value >= threshold else 0
        longest = max(longest, current)
    return longest


def _weighted_prevalence(
    values: Sequence[float], weights: Sequence[float], threshold: float
) -> float:
    total = sum(weights)
    if total <= 0:
        return 0.0
    hit = sum(w for v, w in zip(values, weights) if v >= threshold)
    return float(hit / total)


def aggregate_labels(
    frame_rows: Sequence[dict[str, Any]],
    labels: Iterable[str],
    config: AggregationConfig,
    thresholds: dict[str, float] | None = None,
) -> dict[str, LabelEvidence]:
    """Aggregate calibrated per-frame probabilities into per-study evidence.

    ``frame_rows`` must be ordered as acquired so that consecutive entries are
    consecutive frames of the same sweep. Each row carries ``plane``,
    ``probabilities``, ``weight``, and ``source_name``.
    """
    thresholds = thresholds or {}
    by_plane: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in frame_rows:
        by_plane[str(row.get("plane", "indeterminate"))].append(row)

    evidence: dict[str, LabelEvidence] = {}
    for label in labels:
        scope = config.planes_for_label(label)
        notes: list[str] = []
        plane_probabilities: dict[str, float] = {}
        pooled_values: list[float] = []
        pooled_weights: list[float] = []
        longest_run = 0
        planes_used: list[str] = []

        for plane in scope:
            rows = by_plane.get(plane) or []
            if not rows:
                continue
            # Keep acquisition order inside each source clip so that runs mean
            # consecutive frames of one sweep, not frames stitched across files.
            rows = sorted(rows, key=lambda r: (str(r.get("source_name", "")), int(r.get("frame_index", 0))))
            values = [float(r["probabilities"].get(label, 0.0)) for r in rows]
            weights = [float(r.get("weight", 1.0)) for r in rows]
            required = config.required_frames(len(values))
            order_value = _order_statistic(values, required)
            run_value, _ = _run_statistic(values, config.min_run)
            longest_run = max(
                longest_run,
                _longest_run_above(values, float(thresholds.get(label, config.prevalence_threshold))),
            )
            plane_probabilities[plane] = min(order_value, run_value)
            pooled_values.extend(values)
            pooled_weights.extend(weights)
            planes_used.append(plane)

        if not pooled_values:
            evidence[label] = LabelEvidence(
                label=label,
                probability=0.0,
                order_statistic=0.0,
                run_statistic=0.0,
                prevalence=0.0,
                frames_considered=0,
                frames_required=config.min_frames,
                longest_run=0,
                planes_used=[],
                in_scope=False,
                notes=[f"no accepted frame in {', '.join(scope)}"],
            )
            continue

        required = config.required_frames(len(pooled_values))
        order_value = _order_statistic(pooled_values, required)
        run_value, _ = _run_statistic(pooled_values, config.min_run)
        threshold = float(thresholds.get(label, config.prevalence_threshold))
        prevalence = _weighted_prevalence(pooled_values, pooled_weights, threshold)

        probability = min(order_value, run_value)
        if config.require_two_planes and len(scope) > 1 and len(planes_used) > 1:
            # A finding confirmed in a second plane keeps its score. One seen in
            # only a single plane is held back to the level that plane supports,
            # matching the consensus rule that echogenicity present in one plane
            # only should be read as artifact.
            confirmed = sorted(plane_probabilities.values(), reverse=True)
            if len(confirmed) >= 2:
                probability = min(probability, max(confirmed[1], confirmed[0] * 0.85))
            notes.append("scored across " + ", ".join(planes_used))
        elif len(planes_used) == 1 and len(scope) > 1:
            notes.append(f"seen in {planes_used[0]} only, second-plane confirmation absent")

        if run_value < order_value - 0.15:
            notes.append("signal is not sustained across consecutive frames")

        evidence[label] = LabelEvidence(
            label=label,
            probability=float(probability),
            order_statistic=float(order_value),
            run_statistic=float(run_value),
            prevalence=float(prevalence),
            frames_considered=len(pooled_values),
            frames_required=required,
            longest_run=int(longest_run),
            planes_used=planes_used,
            plane_probabilities=plane_probabilities,
            in_scope=True,
            notes=notes,
        )
    return evidence
