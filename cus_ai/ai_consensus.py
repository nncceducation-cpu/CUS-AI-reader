from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .clinical import classify_study
from .schemas import Answer, SideEvidence, StudyClassification, StudyEvidence


DEFAULT_THRESHOLD = 0.50
DEFAULT_DECISION_MARGIN = 0.05


@dataclass(slots=True)
class AIConsensusResult:
    evidence: StudyEvidence
    classification: StudyClassification
    abstained: bool
    abstention_reasons: list[str]
    missing_outputs: list[str]
    feature_decisions: dict[str, dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "evidence": self.evidence.to_dict(),
            "classification": self.classification.to_dict(),
            "abstained": self.abstained,
            "abstention_reasons": self.abstention_reasons,
            "missing_outputs": self.missing_outputs,
            "feature_decisions": self.feature_decisions,
        }


def _probability(
    probabilities: dict[str, float], aliases: tuple[str, ...]
) -> tuple[str | None, float | None]:
    found = [(name, float(probabilities[name])) for name in aliases if name in probabilities]
    if not found:
        return None, None
    return max(found, key=lambda item: item[1])


def _answer(
    probabilities: dict[str, float],
    aliases: tuple[str, ...],
    thresholds: dict[str, float],
    decision_margin: float,
    decisions: dict[str, dict[str, Any]],
    output_name: str,
) -> Answer:
    label, value = _probability(probabilities, aliases)
    if label is None or value is None:
        decisions[output_name] = {"answer": "unknown", "reason": "model output missing"}
        return "unknown"
    threshold = float(thresholds.get(label, DEFAULT_THRESHOLD))
    if value >= threshold + decision_margin:
        answer: Answer = "yes"
    elif value <= threshold - decision_margin:
        answer = "no"
    else:
        answer = "unknown"
    decisions[output_name] = {
        "answer": answer,
        "label": label,
        "probability": value,
        "threshold": threshold,
        "decision_margin": decision_margin,
    }
    return answer


def _multiclass_choice(
    probabilities: dict[str, float],
    labels: dict[str, str],
    thresholds: dict[str, float],
    decision_margin: float,
    decisions: dict[str, dict[str, Any]],
    output_name: str,
    default: str,
) -> str:
    found = [(name, probabilities[name], value) for name, value in labels.items() if name in probabilities]
    if not found:
        decisions[output_name] = {"answer": default, "reason": "model outputs missing"}
        return default
    found.sort(key=lambda item: item[1], reverse=True)
    label, probability, value = found[0]
    runner_up = found[1][1] if len(found) > 1 else 0.0
    threshold = float(thresholds.get(label, DEFAULT_THRESHOLD))
    accepted = probability >= threshold and probability - runner_up >= decision_margin
    answer = value if accepted else default
    decisions[output_name] = {
        "answer": answer,
        "label": label,
        "probability": float(probability),
        "runner_up_probability": float(runner_up),
        "threshold": threshold,
        "decision_margin": decision_margin,
    }
    return answer


def _side_from_prediction(
    side: str,
    probabilities: dict[str, float],
    thresholds: dict[str, float],
    decision_margin: float,
    decisions: dict[str, dict[str, Any]],
) -> SideEvidence:
    prefix = f"{side}_"
    cystic_answer = _answer(
        probabilities,
        (prefix + "porencephalic_cyst", prefix + "evolved_pvhi_cyst"),
        thresholds,
        decision_margin,
        decisions,
        prefix + "porencephalic_cyst",
    )
    return SideEvidence(
        side=side,  # type: ignore[arg-type]
        hemorrhage_present=_answer(
            probabilities,
            (prefix + "hemorrhage_present", prefix + "germinal_matrix_hemorrhage", prefix + "intraventricular_blood"),
            thresholds,
            decision_margin,
            decisions,
            prefix + "hemorrhage_present",
        ),
        confined_to_germinal_matrix=_answer(
            probabilities,
            (prefix + "confined_to_germinal_matrix", prefix + "germinal_matrix_only"),
            thresholds,
            decision_margin,
            decisions,
            prefix + "confined_to_germinal_matrix",
        ),
        intraventricular_blood=_answer(
            probabilities,
            (prefix + "intraventricular_blood",),
            thresholds,
            decision_margin,
            decisions,
            prefix + "intraventricular_blood",
        ),
        ventricular_distension=_answer(
            probabilities,
            (prefix + "ventricular_distension",),
            thresholds,
            decision_margin,
            decisions,
            prefix + "ventricular_distension",
        ),
        ahw_above_6_mm=_answer(
            probabilities,
            (prefix + "ahw_above_6_mm",),
            thresholds,
            decision_margin,
            decisions,
            prefix + "ahw_above_6_mm",
        ),
        ahw_above_10_mm=_answer(
            probabilities,
            (prefix + "ahw_above_10_mm",),
            thresholds,
            decision_margin,
            decisions,
            prefix + "ahw_above_10_mm",
        ),
        adjacent_periventricular_echogenicity=_answer(
            probabilities,
            (prefix + "focal_periventricular_echogenicity", prefix + "pvhi"),
            thresholds,
            decision_margin,
            decisions,
            prefix + "focal_periventricular_echogenicity",
        ),
        echogenicity_brighter_than_choroid=_answer(
            probabilities,
            (prefix + "echogenicity_brighter_than_choroid",),
            thresholds,
            decision_margin,
            decisions,
            prefix + "echogenicity_brighter_than_choroid",
        ),
        cystic_change={"yes": "porencephalic", "no": "none", "unknown": "not_assessed"}[cystic_answer],
        clinician_verified=False,
    )


def grade_prediction(
    prediction: dict[str, Any],
    *,
    thresholds: dict[str, float] | None = None,
    decision_margin: float = DEFAULT_DECISION_MARGIN,
    study_code: str = "",
    gestational_age_weeks: float | None = None,
    postnatal_age_days: float | None = None,
    all_frames_processed: bool = True,
    serial_study_available: bool = False,
) -> AIConsensusResult:
    probabilities = {name: float(value) for name, value in (prediction.get("probabilities") or {}).items()}
    threshold_map = thresholds or {}
    decisions: dict[str, dict[str, Any]] = {}
    left = _side_from_prediction("left", probabilities, threshold_map, decision_margin, decisions)
    right = _side_from_prediction("right", probabilities, threshold_map, decision_margin, decisions)

    wmi_pattern = _multiclass_choice(
        probabilities,
        {
            "wmi_none": "none",
            "wmi_pve_under_7_days": "pve_under_7_days",
            "wmi_grade_1": "grade_1",
            "wmi_grade_2": "grade_2",
            "wmi_grade_3": "grade_3",
            "wmi_grade_4": "grade_4",
        },
        threshold_map,
        decision_margin,
        decisions,
        "wmi_pattern",
        "not_assessed",
    )
    cbh = _multiclass_choice(
        probabilities,
        {
            "cbh_none": "none",
            "cbh_punctate": "punctate",
            "cbh_limited": "limited",
            "cbh_large": "large",
        },
        threshold_map,
        decision_margin,
        decisions,
        "cerebellar_hemorrhage",
        "not_assessed",
    )
    prior = "yes" if "yes" in {left.hemorrhage_present, right.hemorrhage_present} else (
        "no" if left.hemorrhage_present == right.hemorrhage_present == "no" else "unknown"
    )
    vi97 = _answer(
        probabilities,
        ("vi_above_97th",),
        threshold_map,
        decision_margin,
        decisions,
        "vi_above_97th",
    )
    vi97p4 = _answer(
        probabilities,
        ("vi_above_97th_plus_4_mm",),
        threshold_map,
        decision_margin,
        decisions,
        "vi_above_97th_plus_4_mm",
    )

    plane_counts = {name: int(value) for name, value in (prediction.get("plane_counts") or {}).items()}
    coronal = plane_counts.get("coronal", 0) > 0
    sagittal = plane_counts.get("sagittal", 0) > 0
    posterior = plane_counts.get("posterior_fossa", 0) > 0
    evidence = StudyEvidence(
        study_code=study_code,
        postnatal_age_days=postnatal_age_days,
        gestational_age_weeks=gestational_age_weeks,
        left=left,
        right=right,
        wmi_pattern=wmi_pattern,
        cerebellar_hemorrhage=cbh,
        prior_gmh_ivh=prior,  # type: ignore[arg-type]
        vi_above_97th=vi97,
        vi_above_97th_plus_4mm=vi97p4,
        coronal_views_complete=coronal,
        sagittal_views_complete=sagittal,
        posterior_fossa_views_complete=posterior,
        complete_required_views=coronal and sagittal and posterior,
        all_frames_processed=all_frames_processed,
        decoded_frame_count=int(prediction.get("processed_frame_count") or 0),
        serial_study_available=serial_study_available,
        model_id=prediction.get("model_id"),
        model_version=prediction.get("model_version"),
        model_processed_frame_count=int(prediction.get("processed_frame_count") or 0),
        model_plane_counts=plane_counts,
        evidence_source="ai",
    )
    classification = classify_study(evidence)

    required = [
        f"{side}_{feature}"
        for side in ("left", "right")
        for feature in (
            "hemorrhage_present",
            "confined_to_germinal_matrix",
            "intraventricular_blood",
            "ventricular_distension",
            "ahw_above_6_mm",
            "focal_periventricular_echogenicity",
            "porencephalic_cyst",
        )
    ]
    missing = [name for name in required if decisions.get(name, {}).get("reason") == "model output missing"]
    reasons = list(prediction.get("abstention_reasons") or [])
    if missing:
        reasons.append("required consensus feature outputs are missing")
    if not all_frames_processed:
        reasons.append("complete sequential frame processing was not confirmed")
    if not coronal:
        reasons.append("no accepted coronal frame")
    if not sagittal:
        reasons.append("no accepted sagittal frame")
    if not serial_study_available:
        reasons.append("serial evidence was not supplied for WMI evolution and PHVD trajectory")
    if any(value.get("answer") == "unknown" for value in decisions.values()):
        reasons.append("one or more AI feature decisions are uncertain")
    reasons = list(dict.fromkeys(reasons))
    return AIConsensusResult(
        evidence=evidence,
        classification=classification,
        abstained=bool(prediction.get("abstained")) or bool(reasons),
        abstention_reasons=reasons,
        missing_outputs=missing,
        feature_decisions=decisions,
    )
