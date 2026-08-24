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
    assert report["schema_version"] == "0.4.0"
    assert report["required_human_review"] is True
    assert report["diagnostic_status"] == "Not a diagnostic report"
    assert "CUS-TEST" in markdown
    assert "Provisional consensus classification" in markdown
    assert "10.3389/fped.2021.618236" in markdown


def test_report_can_include_ai_and_agreement():
    evidence = StudyEvidence(study_code="CUS-2")
    classification = classify_study(evidence)
    ai = {
        "abstained": True,
        "classification": classification.to_dict(),
        "abstention_reasons": ["test"],
    }
    agreement = {"domains_compared": 8, "domains_agreeing": 8, "percent_agreement": 100.0}
    report = build_report(evidence, classification, [], ai_consensus=ai, agreement=agreement)
    markdown = report_to_markdown(report)
    assert "## AI grading" in markdown
    assert "Percent agreement: 100.0%" in markdown

