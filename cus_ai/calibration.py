"""Per-label probability calibration.

Raw multi-task network outputs are systematically over-confident, and the
consensus rule engine consumes them against fixed decision thresholds. An
uncalibrated 0.62 and a calibrated 0.62 mean very different things, so every
frame probability passes through this module before aggregation or thresholding.

Three calibrators are supported, all fitted offline and shipped inside the model
manifest so that inference stays deterministic and offline:

``identity``     no transform, the historical behaviour
``platt``        sigmoid(a * logit(p) + b), two parameters per label
``isotonic``     piecewise-linear interpolation over fitted knots

A single global temperature may also be supplied for labels that have no
individual calibration entry.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable

EPSILON = 1e-6


def _clip(value: float) -> float:
    return min(1.0 - EPSILON, max(EPSILON, float(value)))


def logit(value: float) -> float:
    p = _clip(value)
    return math.log(p / (1.0 - p))


def sigmoid(value: float) -> float:
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exponent = math.exp(value)
    return exponent / (1.0 + exponent)


@dataclass(slots=True)
class LabelCalibrator:
    """Calibration for one label."""

    method: str = "identity"
    slope: float = 1.0
    intercept: float = 0.0
    knots_x: list[float] = field(default_factory=list)
    knots_y: list[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.method not in {"identity", "platt", "isotonic"}:
            raise ValueError(f"Unsupported calibration method: {self.method}")
        if self.method == "isotonic":
            if len(self.knots_x) < 2 or len(self.knots_x) != len(self.knots_y):
                raise ValueError("Isotonic calibration needs at least two matched knots.")
            if any(b < a for a, b in zip(self.knots_x, self.knots_x[1:])):
                raise ValueError("Isotonic knot inputs must be non-decreasing.")
            if any(b < a for a, b in zip(self.knots_y, self.knots_y[1:])):
                raise ValueError("Isotonic knot outputs must be non-decreasing.")
        if self.method == "platt" and self.slope <= 0:
            raise ValueError("Platt slope must be positive to preserve ranking.")

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "LabelCalibrator":
        method = str(raw.get("method", "identity")).lower()
        return cls(
            method=method,
            slope=float(raw.get("slope", 1.0)),
            intercept=float(raw.get("intercept", 0.0)),
            knots_x=[float(x) for x in raw.get("knots_x", [])],
            knots_y=[float(y) for y in raw.get("knots_y", [])],
        )

    def apply(self, probability: float) -> float:
        p = _clip(probability)
        if self.method == "identity":
            return p
        if self.method == "platt":
            return _clip(sigmoid(self.slope * logit(p) + self.intercept))
        # isotonic, piecewise linear between fitted knots with flat extrapolation
        xs, ys = self.knots_x, self.knots_y
        if p <= xs[0]:
            return _clip(ys[0])
        if p >= xs[-1]:
            return _clip(ys[-1])
        for index in range(1, len(xs)):
            if p <= xs[index]:
                x0, x1 = xs[index - 1], xs[index]
                y0, y1 = ys[index - 1], ys[index]
                if x1 <= x0:
                    return _clip(y1)
                weight = (p - x0) / (x1 - x0)
                return _clip(y0 + weight * (y1 - y0))
        return _clip(ys[-1])


@dataclass(slots=True)
class CalibrationSet:
    """Calibrators for a whole model, plus an optional global temperature."""

    temperature: float = 1.0
    per_label: dict[str, LabelCalibrator] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.temperature <= 0:
            raise ValueError("Temperature must be positive.")

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> "CalibrationSet":
        raw = raw or {}
        per_label = {
            name: LabelCalibrator.from_dict(entry)
            for name, entry in (raw.get("per_label") or {}).items()
        }
        return cls(temperature=float(raw.get("temperature", 1.0)), per_label=per_label)

    @property
    def is_identity(self) -> bool:
        return self.temperature == 1.0 and all(
            item.method == "identity" for item in self.per_label.values()
        )

    def calibrate(self, label: str, probability: float) -> float:
        calibrator = self.per_label.get(label)
        if calibrator is not None and calibrator.method != "identity":
            return calibrator.apply(probability)
        if self.temperature != 1.0:
            return _clip(sigmoid(logit(probability) / self.temperature))
        return _clip(probability)

    def calibrate_row(self, row: dict[str, float]) -> dict[str, float]:
        return {label: self.calibrate(label, value) for label, value in row.items()}

    def uncalibrated_labels(self, labels: Iterable[str]) -> list[str]:
        return sorted(
            label
            for label in labels
            if self.per_label.get(label, LabelCalibrator()).method == "identity"
        )
