"""Derive consensus evidence fields from model label probabilities.

The previous mapping looked each evidence field up against a tuple of candidate
label names and took whichever alias carried the *highest* probability. That had
two consequences, both of which cost accuracy.

First, taking the maximum over aliases is a one-sided error. ``hemorrhage_present``
listed ``intraventricular_blood`` among its aliases, so any frame the model
scored highly for intraventricular blood also raised hemorrhage presence, and
never lowered it. Every alias group biased its field toward "yes".

Second, ``confined_to_germinal_matrix`` had no label in the manifest contract at
all, so it resolved to "unknown" on every study. Step 2 of the consensus
algorithm could never be answered, which pinned every AI grade at
"Indeterminate GMH-IVH grade" no matter what the images showed.

This module replaces alias lookup with derivation. Fields that a network can
plausibly be trained to output are read directly. Fields that are logical
consequences of other fields are derived from the decisions already made, which
is how the consensus algorithm itself is written: Step 2 does not need its own
detector, it needs the answers to "is there hemorrhage" and "is there blood in
the ventricle".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .schemas import Answer, SideEvidence

DEFAULT_THRESHOLD = 0.50
DEFAULT_DECISION_MARGIN = 0.05


@dataclass(slots=True)
class Decision:
    """One trinary decision with the numbers that produced it."""

    field_name: str
    answer: Answer | str
    source: str
    label: str | None = None
    probability: float | None = None
    threshold: float | None = None
    margin: float | None = None
    confidence: float = 0.0
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "answer": self.answer,
            "source": self.source,
            "confidence": round(self.confidence, 4),
        }
        if self.label is not None:
            payload["label"] = self.label
        if self.probability is not None:
            payload["probability"] = round(self.probability, 6)
        if self.threshold is not None:
            payload["threshold"] = round(self.threshold, 6)
        if self.margin is not None:
            payload["decision_margin"] = round(self.margin, 6)
        if self.notes:
            payload["notes"] = list(self.notes)
        return payload


def _confidence(probability: float, threshold: float) -> float:
    """Distance from the decision boundary, scaled to the available headroom."""
    if probability >= threshold:
        headroom = max(1e-6, 1.0 - threshold)
        return min(1.0, (probability - threshold) / headroom)
    headroom = max(1e-6, threshold)
    return min(1.0, (threshold - probability) / headroom)


def decide(
    field_name: str,
    label: str,
    probabilities: dict[str, float],
    thresholds: dict[str, float],
    margin: float,
    decisions: dict[str, Decision],
    *,
    notes: list[str] | None = None,
) -> Answer:
    """Threshold one label into yes / no / unknown with a dead band."""
    if label not in probabilities:
        decisions[field_name] = Decision(
            field_name=field_name,
            answer="unknown",
            source="missing",
            label=label,
            notes=["model does not emit this label"],
        )
        return "unknown"
    value = float(probabilities[label])
    threshold = float(thresholds.get(label, DEFAULT_THRESHOLD))
    if value >= threshold + margin:
        answer: Answer = "yes"
    elif value <= threshold - margin:
        answer = "no"
    else:
        answer = "unknown"
    decisions[field_name] = Decision(
        field_name=field_name,
        answer=answer,
        source="model",
        label=label,
        probability=value,
        threshold=threshold,
        margin=margin,
        confidence=_confidence(value, threshold),
        notes=list(notes or []),
    )
    return answer


def _derive(
    field_name: str,
    answer: Answer | str,
    decisions: dict[str, Decision],
    detail: str,
    confidence: float,
) -> Answer | str:
    decisions[field_name] = Decision(
        field_name=field_name,
        answer=answer,
        source="derived",
        confidence=confidence,
        notes=[detail],
    )
    return answer


def _union(probabilities: dict[str, float], labels: tuple[str, ...]) -> float | None:
    """Lower bound on the probability that at least one member holds."""
    present = [float(probabilities[name]) for name in labels if name in probabilities]
    return max(present) if present else None


def side_evidence_from_probabilities(
    side: str,
    probabilities: dict[str, float],
    thresholds: dict[str, float],
    margin: float,
    decisions: dict[str, Decision],
) -> SideEvidence:
    """Build one hemisphere's consensus evidence from calibrated probabilities."""
    prefix = f"{side}_"

    # Step 1 needs "is there hemorrhage in or around the germinal matrix, or in
    # the lateral, third, or fourth ventricle". Where the model exposes germinal
    # matrix and intraventricular heads separately, the union of the two is the
    # question being asked, so the union is what gets thresholded.
    gmh_label = prefix + "germinal_matrix_hemorrhage"
    ivh_label = prefix + "intraventricular_blood"
    direct_label = prefix + "hemorrhage_present"

    working = dict(probabilities)
    if direct_label not in working:
        union_value = _union(working, (gmh_label, ivh_label))
        if union_value is not None:
            working[direct_label] = union_value

    hemorrhage = decide(
        prefix + "hemorrhage_present",
        direct_label,
        working,
        thresholds,
        margin,
        decisions,
        notes=(
            []
            if direct_label in probabilities
            else ["union of germinal matrix and intraventricular heads"]
        ),
    )
    intraventricular = decide(
        prefix + "intraventricular_blood", ivh_label, working, thresholds, margin, decisions
    )

    # Step 2 is a consequence of Step 1, not a separate detector. Hemorrhage
    # present with no intraventricular blood is confined to the germinal matrix.
    confined_key = prefix + "confined_to_germinal_matrix"
    if prefix + "confined_to_germinal_matrix" in probabilities:
        confined = decide(
            confined_key, confined_key, probabilities, thresholds, margin, decisions
        )
    elif hemorrhage == "yes" and intraventricular == "no":
        confined = _derive(
            confined_key,
            "yes",
            decisions,
            "hemorrhage present with no intraventricular blood",
            decisions[prefix + "intraventricular_blood"].confidence,
        )
    elif intraventricular == "yes":
        confined = _derive(
            confined_key,
            "no",
            decisions,
            "intraventricular blood places hemorrhage beyond the germinal matrix",
            decisions[prefix + "intraventricular_blood"].confidence,
        )
    elif hemorrhage == "no":
        confined = _derive(
            confined_key, "no", decisions, "no hemorrhage recorded on this side", 0.0
        )
    else:
        confined = _derive(
            confined_key,
            "unknown",
            decisions,
            "intraventricular blood is uncertain, so Step 2 cannot be answered",
            0.0,
        )

    distension = decide(
        prefix + "ventricular_distension",
        prefix + "ventricular_distension",
        probabilities,
        thresholds,
        margin,
        decisions,
    )
    ahw6 = decide(
        prefix + "ahw_above_6_mm", prefix + "ahw_above_6_mm", probabilities, thresholds, margin, decisions
    )
    ahw10 = decide(
        prefix + "ahw_above_10_mm", prefix + "ahw_above_10_mm", probabilities, thresholds, margin, decisions
    )

    # Periventricular echogenicity is a white matter finding. The old mapping
    # accepted a label literally named "pvhi" as a stand-in for it, which is a
    # category error: PVHI is the conclusion the rules are meant to reach, so
    # feeding it back in as an input short-circuits Step 4.
    echogenicity = decide(
        prefix + "focal_periventricular_echogenicity",
        prefix + "focal_periventricular_echogenicity",
        probabilities,
        thresholds,
        margin,
        decisions,
    )
    brighter = decide(
        prefix + "echogenicity_brighter_than_choroid",
        prefix + "echogenicity_brighter_than_choroid",
        probabilities,
        thresholds,
        margin,
        decisions,
    )

    cyst = decide(
        prefix + "porencephalic_cyst",
        prefix + "porencephalic_cyst",
        probabilities,
        thresholds,
        margin,
        decisions,
    )
    multiple = decide(
        prefix + "multiple_evolved_pvhi_cysts",
        prefix + "multiple_evolved_pvhi_cysts",
        probabilities,
        thresholds,
        margin,
        decisions,
    )
    if multiple == "yes":
        cystic_change = "multiple_evolved_pvhi"
    else:
        cystic_change = {"yes": "porencephalic", "no": "none", "unknown": "not_assessed"}[cyst]

    return SideEvidence(
        side=side,  # type: ignore[arg-type]
        hemorrhage_present=hemorrhage,
        confined_to_germinal_matrix=confined,  # type: ignore[arg-type]
        intraventricular_blood=intraventricular,
        ventricular_distension=distension,
        ahw_above_6_mm=ahw6,
        ahw_above_10_mm=ahw10,
        adjacent_periventricular_echogenicity=echogenicity,
        echogenicity_brighter_than_choroid=brighter,
        cystic_change=cystic_change,
        clinician_verified=False,
    )


def multiclass_choice(
    field_name: str,
    probabilities: dict[str, float],
    label_map: dict[str, str],
    thresholds: dict[str, float],
    margin: float,
    decisions: dict[str, Decision],
    default: str,
) -> str:
    """Pick one grade from a mutually exclusive family, or abstain."""
    present = [
        (name, float(probabilities[name]), value)
        for name, value in label_map.items()
        if name in probabilities
    ]
    if not present:
        decisions[field_name] = Decision(
            field_name=field_name,
            answer=default,
            source="missing",
            notes=["model emits no label for this family"],
        )
        return default
    present.sort(key=lambda item: item[1], reverse=True)
    label, probability, value = present[0]
    runner_up = present[1][1] if len(present) > 1 else 0.0
    threshold = float(thresholds.get(label, DEFAULT_THRESHOLD))
    separated = probability - runner_up
    accepted = probability >= threshold and separated >= margin
    notes: list[str] = []
    if not accepted:
        if probability < threshold:
            notes.append(f"top grade {label} sits below its threshold")
        if separated < margin:
            notes.append(f"top two grades are separated by only {separated:.3f}")
    decisions[field_name] = Decision(
        field_name=field_name,
        answer=value if accepted else default,
        source="model",
        label=label,
        probability=probability,
        threshold=threshold,
        margin=margin,
        confidence=min(1.0, separated / max(1e-6, margin)) if accepted else 0.0,
        notes=notes,
    )
    return value if accepted else default
