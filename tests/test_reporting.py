from cus_ai.clinical import classify_study
from cus_ai.reporting import build_report, report_to_markdown
from cus_ai.schemas import SideEvidence, StudyEvidence


def test_report_is_auditable_and_explicitly_non_diagnostic():
    evidence = StudyEvidence(
        study_code="CUS-TEST",
        left=SideEvidence(side="left", hemorrhage_present="no", adjacent_periventricular_echogenicity="no"),
        right=SideEvidence(side="right", hemorrhage_present="no", adjacent_periventricular_echogenicity="no"),
    )
    report = build_report(evidence, classify_study(evidence), [{"name": "test.png", "frames_loaded": 1}])
    markdown = report_to_markdown(report)
    assert report["schema_version"] == "0.2.0"
    assert report["required_human_review"] is True
    assert report["diagnostic_status"] == "Not a diagnostic report"
    assert "CUS-TEST" in markdown
    assert "Provisional consensus classification" in markdown
    assert "10.3389/fped.2021.618236" in markdown
