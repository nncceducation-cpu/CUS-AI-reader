from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from .schemas import StudyClassification, StudyEvidence


def build_report(
    evidence: StudyEvidence,
    classification: StudyClassification,
    media_summary: list[dict[str, Any]],
    model_prediction: dict[str, Any] | None = None,
    ai_consensus: dict[str, Any] | None = None,
    agreement: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": "0.4.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "intended_use": "Research and quality improvement prototype only",
        "diagnostic_status": "Not a diagnostic report",
        "consensus_source": {
            "citation": (
                "Mohammad K, Scott JN, Leijser LM, et al. Consensus Approach for Standardizing "
                "the Screening and Classification of Preterm Brain Injury Diagnosed With Cranial "
                "Ultrasound: A Canadian Perspective. Front Pediatr. 2021;9:618236."
            ),
            "doi": "10.3389/fped.2021.618236",
        },
        "study_evidence": evidence.to_dict(),
        "classification": classification.to_dict(),
        "media_summary": media_summary,
        "model_prediction": model_prediction,
        "ai_consensus": ai_consensus,
        "expert_ai_agreement": agreement,
        "required_human_review": True,
    }


def report_to_markdown(report: dict[str, Any]) -> str:
    c = report["classification"]
    e = report["study_evidence"]
    lines = [
        "# CUS AI Reader structured research report",
        "",
        f"Study code: {e['study_code'] or 'Not supplied'}",
        f"Generated: {report['generated_at_utc']}",
        "",
        "> Research and quality improvement prototype. This is not a diagnostic report.",
        "",
        "## Consensus classification",
        "",
        f"- Status: {c['classification_status']}",
        f"- Left: {c['left']['gmh_ivh']}; PVHI: {c['left']['pvhi']}",
        f"- Right: {c['right']['gmh_ivh']}; PVHI: {c['right']['pvhi']}",
        f"- White matter injury: {c['wmi']}",
        f"- Cerebellar hemorrhage: {c['cerebellar_hemorrhage']}",
        f"- PHVD: {c['phvd']}",
        f"- Severe-injury flag: {c['severe_preterm_brain_injury_flag']}",
        f"- Coronal coverage confirmed: {c['view_coverage']['coronal']}",
        f"- Sagittal or parasagittal coverage confirmed: {c['view_coverage']['sagittal_or_parasagittal']}",
        f"- Posterior fossa coverage confirmed: {c['view_coverage']['posterior_fossa']}",
    ]
    ai = report.get("ai_consensus")
    if ai:
        ai_c = ai["classification"]
        lines.extend(
            [
                "",
                "## AI grading",
                "",
                f"- Abstained: {ai['abstained']}",
                f"- Left: {ai_c['left']['gmh_ivh']}; PVHI: {ai_c['left']['pvhi']}",
                f"- Right: {ai_c['right']['gmh_ivh']}; PVHI: {ai_c['right']['pvhi']}",
                f"- White matter injury: {ai_c['wmi']}",
                f"- Cerebellar hemorrhage: {ai_c['cerebellar_hemorrhage']}",
                f"- PHVD: {ai_c['phvd']}",
            ]
        )
        if report.get("expert_ai_agreement"):
            agreement = report["expert_ai_agreement"]
            lines.extend(
                [
                    "",
                    "## Expert versus AI agreement",
                    "",
                    f"- Domains compared: {agreement['domains_compared']}",
                    f"- Domains agreeing: {agreement['domains_agreeing']}",
                    f"- Percent agreement: {agreement['percent_agreement']:.1f}%",
                ]
            )
    lines.extend(["", "## Limitations", ""])
    limitations = c.get("limitations") or ["No additional limitation recorded."]
    lines.extend(f"- {item}" for item in limitations)
    lines.extend(
        [
            "",
            "## Source",
            "",
            "Mohammad K, Scott JN, Leijser LM, et al. Front Pediatr. 2021;9:618236. "
            "doi:10.3389/fped.2021.618236.",
            "",
        ]
    )
    return "\n".join(lines)

