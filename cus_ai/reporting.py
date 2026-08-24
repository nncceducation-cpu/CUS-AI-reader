from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from .schemas import StudyClassification, StudyEvidence


def build_report(
    evidence: StudyEvidence,
    classification: StudyClassification,
    media_summary: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "schema_version": "0.1.0",
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
        f"- Left: {c['left']['gmh_ivh']}; PVHI: {c['left']['pvhi']}",
        f"- Right: {c['right']['gmh_ivh']}; PVHI: {c['right']['pvhi']}",
        f"- White matter injury: {c['wmi']}",
        f"- Cerebellar hemorrhage: {c['cerebellar_hemorrhage']}",
        f"- PHVD: {c['phvd']}",
        f"- Severe-injury flag: {c['severe_preterm_brain_injury_flag']}",
        "",
        "## Limitations",
        "",
    ]
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

