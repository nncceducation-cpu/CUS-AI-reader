"""Turn calibrated model probabilities into a consensus classification.

The previous version abstained globally. Any single uncertain feature, and the
absence of a serial study, both appended a reason, and any reason at all set
``abstained`` for the whole study. Since ``serial_study_available`` defaults to
false and the manifest contract had no label for Step 2, the practical effect
was that every study abstained on every domain, including the domains the model
had answered confidently.

Abstention is now per domain. A study with a clean coronal sweep and no mastoid
window reports its GMH-IVH grades and withholds cerebellar hemorrhage, which is
what a reader would do. The study-level ``abstained`` flag fires only when no
domain survives, and the reasons that used to suppress everything are carried as
scoped caveats on the domains they actually affect.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import math

from .clinical import classify_study
from .evidence_mapping import (
    DEFAULT_DECISION_MARGIN,
    DEFAULT_THRESHOLD,
    Decision,
    multiclass_choice,
    side_evidence_from_probabilities,
)
from .schemas import StudyClassification, StudyEvidence

WMI_LABELS = {
    "wmi_none": "none",
    "wmi_pve_under_7_days": "pve_under_7_days",
    "wmi_grade_1": "grade_1",
    "wmi_grade_2": "grade_2",
    "wmi_grade_3": "grade_3",
    "wmi_grade_4": "grade_4",
}

CBH_LABELS = {
    "cbh_none": "none",
    "cbh_punctate": "punctate",
    "cbh_limited": "limited",
    "cbh_large": "large",
}

# Fields each reportable domain depends on. A domain is reportable when its
# required decisions were answered; optional decisions only lower confidence.
DOMAIN_REQUIREMENTS: dict[str, tuple[str, ...]] = {
    "left_gmh_ivh": (
        "left_hemorrhage_present",
        "left_intraventricular_blood",
        "left_confined_to_germinal_matrix",
    ),
    "right_gmh_ivh": (
        "right_hemorrhage_present",
        "right_intraventricular_blood",
        "right_confined_to_germinal_matrix",
    ),
    "left_pvhi": ("left_hemorrhage_present", "left_focal_periventricular_echogenicity"),
    "right_pvhi": ("right_hemorrhage_present", "right_focal_periventricular_echogenicity"),
    "wmi": ("wmi_pattern",),
    "cerebellar_hemorrhage": ("cerebellar_hemorrhage",),
    "phvd": ("vi_above_97th", "vi_above_97th_plus_4_mm"),
}

DOMAIN_PLANE_REQUIREMENTS: dict[str, tuple[str, ...]] = {
    "left_gmh_ivh": ("coronal", "sagittal"),
    "right_gmh_ivh": ("coronal", "sagittal"),
    "left_pvhi": ("coronal", "sagittal"),
    "right_pvhi": ("coronal", "sagittal"),
    "wmi": ("coronal", "sagittal"),
    "cerebellar_hemorrhage": ("posterior_fossa",),
    "phvd": ("coronal",),
}

# Domains that cannot be settled from one time point, whatever the model says.
SERIAL_DEPENDENT_DOMAINS = ("wmi", "phvd")

# Refinements a model may or may not emit. Their absence lowers detail, not
# validity, so they are not counted as missing consensus outputs.
OPTIONAL_FIELDS = (
    "left_multiple_evolved_pvhi_cysts",
    "right_multiple_evolved_pvhi_cysts",
    "left_confined_to_germinal_matrix",
    "right_confined_to_germinal_matrix",
)


@dataclass(slots=True)
class DomainStatus:
    """Whether one reportable domain can be released, and how confidently."""

    domain: str
    reportable: bool
    confidence: float
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "reportable": self.reportable,
            "confidence": round(self.confidence, 4),
            "reasons": list(self.reasons),
        }


@dataclass(slots=True)
class AIConsensusResult:
    evidence: StudyEvidence
    classification: StudyClassification
    abstained: bool
    abstention_reasons: list[str]
    missing_outputs: list[str]
    feature_decisions: dict[str, dict[str, Any]]
    domain_status: dict[str, DomainStatus] = field(default_factory=dict)
    study_caveats: list[str] = field(default_factory=list)

    @property
    def reportable_domains(self) -> list[str]:
        return sorted(name for name, status in self.domain_status.items() if status.reportable)

    def to_dict(self) -> dict[str, Any]:
        return {
            "evidence": self.evidence.to_dict(),
            "classification": self.classification.to_dict(),
            "abstained": self.abstained,
            "abstention_reasons": self.abstention_reasons,
            "missing_outputs": self.missing_outputs,
            "feature_decisions": self.feature_decisions,
            "domain_status": {name: status.to_dict() for name, status in self.domain_status.items()},
            "reportable_domains": self.reportable_domains,
            "study_caveats": self.study_caveats,
        }


def _assess_domain(
    domain: str,
    decisions: dict[str, Decision],
    plane_counts: dict[str, int],
    serial_study_available: bool,
    consistency_conflicts: list[str],
) -> DomainStatus:
    reasons: list[str] = []
    confidences: list[float] = []
    reportable = True

    for field_name in DOMAIN_REQUIREMENTS.get(domain, ()):
        decision = decisions.get(field_name)
        if decision is None or decision.source == "missing":
            reasons.append(f"{field_name} has no model output")
            reportable = False
            continue
        confidences.append(decision.confidence)
        if decision.answer in {"unknown", "not_assessed"}:
            reasons.append(f"{field_name} fell inside the decision dead band")
            reportable = False

    for plane in DOMAIN_PLANE_REQUIREMENTS.get(domain, ()):
        if plane_counts.get(plane, 0) == 0:
            reasons.append(f"no accepted {plane.replace('_', ' ')} frame")
            reportable = False

    if domain in SERIAL_DEPENDENT_DOMAINS and not serial_study_available:
        reasons.append(
            "this domain is defined by change over time and no serial study was supplied"
        )
        reportable = False

    if consistency_conflicts:
        reasons.append("model outputs required anatomic repair before grading")

    confidence = min(confidences) if confidences else 0.0
    if not reportable:
        confidence = min(confidence, 0.0) if not confidences else confidence * 0.5
    return DomainStatus(domain=domain, reportable=reportable, confidence=confidence, reasons=reasons)


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
    probabilities = {
        name: float(value) for name, value in (prediction.get("probabilities") or {}).items()
    }
    threshold_map = thresholds or {}
    if any(not math.isfinite(value) or value < 0 or value > 1 for value in probabilities.values()):
        raise ValueError("Consensus features must be finite probabilities between 0 and 1.")
    decisions: dict[str, Decision] = {}

    left = side_evidence_from_probabilities(
        "left", probabilities, threshold_map, decision_margin, decisions
    )
    right = side_evidence_from_probabilities(
        "right", probabilities, threshold_map, decision_margin, decisions
    )

    wmi_pattern = multiclass_choice(
        "wmi_pattern", probabilities, WMI_LABELS, threshold_map, decision_margin, decisions, "not_assessed"
    )
    cbh = multiclass_choice(
        "cerebellar_hemorrhage", probabilities, CBH_LABELS, threshold_map, decision_margin, decisions, "not_assessed"
    )

    # A current-frame hemorrhage head cannot establish preceding hemorrhage.
    # Only longitudinal, explicitly supplied history may answer this field.
    prior = prediction.get("prior_gmh_ivh", "unknown")
    if prior not in {"yes", "no", "unknown"}:
        prior = "unknown"
    vi97 = _decide_simple(
        "vi_above_97th", probabilities, threshold_map, decision_margin, decisions
    )
    vi97p4 = _decide_simple(
        "vi_above_97th_plus_4_mm", probabilities, threshold_map, decision_margin, decisions
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
        coronal_views_complete=bool(prediction.get("verified_view_coverage", {}).get("coronal", False)) and coronal,
        sagittal_views_complete=bool(prediction.get("verified_view_coverage", {}).get("sagittal", False)) and sagittal,
        posterior_fossa_views_complete=bool(prediction.get("verified_view_coverage", {}).get("posterior_fossa", False)) and posterior,
        complete_required_views=False,
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

    conflicts = list((prediction.get("consistency") or {}).get("conflicts") or [])
    domain_status = {
        domain: _assess_domain(domain, decisions, plane_counts, serial_study_available, conflicts)
        for domain in DOMAIN_REQUIREMENTS
    }

    global_reasons = []
    if not all_frames_processed:
        global_reasons.append("complete sequential frame processing was not confirmed")
    if prediction.get("abstained", False):
        global_reasons.extend(prediction.get("abstention_reasons") or ["upstream model abstained"])
    if prediction.get("model_type") == "pilot_similarity" or prediction.get("aggregation_mode") == "pilot_similarity_vote":
        global_reasons.append("reference similarity cannot establish consensus imaging features")

    for domain, status in domain_status.items():
        reasons = list(global_reasons)
        if domain.endswith("gmh_ivh"):
            side = left if domain.startswith("left") else right
            side_class = classification.left if domain.startswith("left") else classification.right
            if "Indeterminate" in side_class.gmh_ivh:
                reasons.append("GMH-IVH criteria remain indeterminate, including acute distension and AHW")
        elif domain.endswith("pvhi"):
            side_class = classification.left if domain.startswith("left") else classification.right
            if side_class.pvhi.startswith("Indeterminate"):
                reasons.append("PVHI criteria remain indeterminate")
        elif domain == "phvd" and (prior == "unknown" or postnatal_age_days is None):
            reasons.append("PHVD requires preceding hemorrhage history and postnatal age")
        if reasons:
            status.reportable = False
            status.confidence = 0.0
            status.reasons = list(dict.fromkeys(status.reasons + reasons))

    # Withheld results must remain withheld in the UI, exports and agreement
    # tables; hiding only the banner leaves a plausible grade visible below it.
    for side_name in ("left", "right"):
        side_class = getattr(classification, side_name)
        if not domain_status[f"{side_name}_gmh_ivh"].reportable:
            side_class.gmh_ivh = "Not assessed: AI withheld this domain"
        if not domain_status[f"{side_name}_pvhi"].reportable:
            side_class.pvhi = "Not assessed: AI withheld this domain"
        side_class.cystic_sequela = "Not assessed: requires verified serial evidence"
    for domain, attribute in (("wmi", "wmi"), ("cerebellar_hemorrhage", "cerebellar_hemorrhage"), ("phvd", "phvd")):
        if not domain_status[domain].reportable:
            setattr(classification, attribute, "Not assessed: AI withheld this domain")
    if global_reasons or not all(status.reportable for status in domain_status.values()):
        classification.severe_preterm_brain_injury_flag = "Not assessed: incomplete or withheld AI domains"

    missing = sorted(
        name
        for name, decision in decisions.items()
        if decision.source == "missing" and name not in OPTIONAL_FIELDS
    )

    caveats: list[str] = list(prediction.get("abstention_reasons") or [])
    if not all_frames_processed:
        caveats.append("complete sequential frame processing was not confirmed")
    if not prediction.get("calibration_applied", False):
        caveats.append(
            "no probability calibration was applied, so thresholds are being compared against "
            "raw network confidence"
        )
    if conflicts:
        caveats.extend(conflicts)
    if missing:
        caveats.append(f"{len(missing)} consensus feature outputs are absent from the model")
    caveats = list(dict.fromkeys(caveats))

    reportable = [name for name, status in domain_status.items() if status.reportable]
    abstained = not reportable

    abstention_reasons = (
        sorted({reason for status in domain_status.values() for reason in status.reasons})
        if abstained
        else []
    )

    return AIConsensusResult(
        evidence=evidence,
        classification=classification,
        abstained=abstained,
        abstention_reasons=abstention_reasons,
        missing_outputs=missing,
        feature_decisions={name: decision.to_dict() for name, decision in decisions.items()},
        domain_status=domain_status,
        study_caveats=caveats,
    )


def _decide_simple(
    label: str,
    probabilities: dict[str, float],
    thresholds: dict[str, float],
    margin: float,
    decisions: dict[str, Decision],
) -> str:
    from .evidence_mapping import decide

    return decide(label, label, probabilities, thresholds, margin, decisions)
