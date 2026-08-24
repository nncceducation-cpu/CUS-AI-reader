from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal


Answer = Literal["yes", "no", "unknown"]
Side = Literal["left", "right"]


@dataclass(slots=True)
class SideEvidence:
    side: Side
    hemorrhage_present: Answer = "unknown"
    confined_to_germinal_matrix: Answer = "unknown"
    intraventricular_blood: Answer = "unknown"
    ventricular_distension: Answer = "unknown"
    ahw_mm: float | None = None
    ahw_above_6_mm: Answer = "unknown"
    ahw_above_10_mm: Answer = "unknown"
    adjacent_periventricular_echogenicity: Answer = "unknown"
    echogenicity_brighter_than_choroid: Answer = "unknown"
    cystic_change: str = "not_assessed"
    clinician_verified: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class StudyEvidence:
    study_code: str
    expert_reader_code: str | None = None
    expert_review_round: str = "independent"
    postnatal_age_days: float | None = None
    gestational_age_weeks: float | None = None
    left: SideEvidence = field(default_factory=lambda: SideEvidence(side="left"))
    right: SideEvidence = field(default_factory=lambda: SideEvidence(side="right"))
    wmi_pattern: str = "not_assessed"
    cerebellar_hemorrhage: str = "not_assessed"
    prior_gmh_ivh: Answer = "unknown"
    vi_above_97th: Answer = "unknown"
    vi_above_97th_plus_4mm: Answer = "unknown"
    coronal_views_complete: bool = False
    sagittal_views_complete: bool = False
    posterior_fossa_views_complete: bool = False
    complete_required_views: bool = False
    all_frames_processed: bool = False
    decoded_frame_count: int = 0
    serial_study_available: bool = False
    model_id: str | None = None
    model_version: str | None = None
    model_processed_frame_count: int | None = None
    model_plane_counts: dict[str, int] = field(default_factory=dict)
    evidence_source: str = "expert"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class SideClassification:
    side: Side
    gmh_ivh: str
    pvhi: str
    cystic_sequela: str
    evidence_complete: bool
    reasoning: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class StudyClassification:
    left: SideClassification
    right: SideClassification
    wmi: str
    cerebellar_hemorrhage: str
    phvd: str
    classification_status: str
    view_coverage: dict[str, bool]
    severe_preterm_brain_injury_flag: str
    limitations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
