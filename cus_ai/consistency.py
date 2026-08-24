"""Logical repair of model probabilities before the consensus rules run.

A multi-task network emits each label from its own head, so nothing stops it
returning P(AHW > 10 mm) = 0.80 alongside P(AHW > 6 mm) = 0.40. That pair is
impossible: every ventricle wider than 10 mm is also wider than 6 mm. Feeding
such a pair to the grading rules produces a confident contradiction rather than
an honest uncertainty.

This module imposes the constraints that the anatomy guarantees:

* nested thresholds are monotone, so P(above the higher cut) never exceeds
  P(above the lower cut)
* a containment hierarchy, so intraventricular blood never exceeds hemorrhage
  present, and distension never exceeds intraventricular blood
* mutually exclusive grade families (WMI, cerebellar hemorrhage) are
  renormalised to a proper distribution when the manifest declares them softmax
  groups

Every repair is recorded, because a model that needs frequent repair is a model
whose calibration or training set is wrong, and the audit trail is how that gets
noticed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# (higher-bar label, lower-bar label): the first can never exceed the second.
MONOTONE_PAIRS: tuple[tuple[str, str], ...] = (
    ("left_ahw_above_10_mm", "left_ahw_above_6_mm"),
    ("right_ahw_above_10_mm", "right_ahw_above_6_mm"),
    ("vi_above_97th_plus_4_mm", "vi_above_97th"),
)

# (contained label, containing label)
CONTAINMENT_PAIRS: tuple[tuple[str, str], ...] = (
    ("left_intraventricular_blood", "left_hemorrhage_present"),
    ("right_intraventricular_blood", "right_hemorrhage_present"),
    ("left_ventricular_distension", "left_intraventricular_blood"),
    ("right_ventricular_distension", "right_intraventricular_blood"),
    ("left_echogenicity_brighter_than_choroid", "left_focal_periventricular_echogenicity"),
    ("right_echogenicity_brighter_than_choroid", "right_focal_periventricular_echogenicity"),
)

EXCLUSIVE_GROUPS: dict[str, tuple[str, ...]] = {
    "wmi": (
        "wmi_none",
        "wmi_pve_under_7_days",
        "wmi_grade_1",
        "wmi_grade_2",
        "wmi_grade_3",
        "wmi_grade_4",
    ),
    "cerebellar_hemorrhage": ("cbh_none", "cbh_punctate", "cbh_limited", "cbh_large"),
}


@dataclass(slots=True)
class ConsistencyReport:
    probabilities: dict[str, float]
    repairs: list[dict[str, Any]] = field(default_factory=list)
    conflicts: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "repairs": list(self.repairs),
            "conflicts": list(self.conflicts),
            "repair_count": len(self.repairs),
        }


def _record(
    report: ConsistencyReport, rule: str, label: str, before: float, after: float, detail: str
) -> None:
    if abs(before - after) < 1e-9:
        return
    report.repairs.append(
        {
            "rule": rule,
            "label": label,
            "before": round(before, 6),
            "after": round(after, 6),
            "detail": detail,
        }
    )


def enforce_consistency(
    probabilities: dict[str, float],
    *,
    softmax_groups: tuple[str, ...] = (),
    conflict_tolerance: float = 0.20,
) -> ConsistencyReport:
    """Return probabilities that satisfy the anatomic constraints."""
    values = {name: float(value) for name, value in probabilities.items()}
    report = ConsistencyReport(probabilities=values)

    for higher, lower in MONOTONE_PAIRS:
        if higher in values and lower in values:
            if values[higher] > values[lower]:
                gap = values[higher] - values[lower]
                if gap > conflict_tolerance:
                    report.conflicts.append(
                        f"{higher} exceeds {lower} by {gap:.2f}, which the nested thresholds forbid"
                    )
                before = values[higher]
                values[higher] = values[lower]
                _record(
                    report,
                    "monotone_threshold",
                    higher,
                    before,
                    values[higher],
                    f"clamped to {lower}",
                )

    for contained, container in CONTAINMENT_PAIRS:
        if contained in values and container in values:
            if values[contained] > values[container]:
                gap = values[contained] - values[container]
                if gap > conflict_tolerance:
                    report.conflicts.append(
                        f"{contained} exceeds {container} by {gap:.2f}, which the containment "
                        "hierarchy forbids"
                    )
                before = values[contained]
                values[contained] = values[container]
                _record(
                    report,
                    "containment",
                    contained,
                    before,
                    values[contained],
                    f"clamped to {container}",
                )

    for group_name in softmax_groups:
        members = [name for name in EXCLUSIVE_GROUPS.get(group_name, ()) if name in values]
        if len(members) < 2:
            continue
        total = sum(values[name] for name in members)
        if total <= 0:
            continue
        if abs(total - 1.0) > 1e-6:
            for name in members:
                before = values[name]
                values[name] = before / total
                _record(
                    report,
                    "softmax_renormalisation",
                    name,
                    before,
                    values[name],
                    f"group {group_name} summed to {total:.3f}",
                )

    report.probabilities = values
    return report
