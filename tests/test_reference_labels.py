from pathlib import Path

from cus_ai.clinical import classify_study
from cus_ai.reference_labels import compare_to_reference, find_reference_label, load_reference_labels
from cus_ai.schemas import SideEvidence, StudyEvidence


LABEL_PATH = Path(__file__).parents[1] / "data" / "pilot_reference_labels_v1.csv"


def test_provisional_label_registry_contains_all_fifteen_examinations():
    rows = load_reference_labels(LABEL_PATH)
    assert len(rows) == 15
    assert len({row["study_code"] for row in rows}) == 15


def test_study_aliases_match_case_day_and_pma_labels():
    rows = load_reference_labels(LABEL_PATH)
    assert find_reference_label(rows, "case 6 day 14")["study_code"] == "CASE-06-DAY-14"
    assert find_reference_label(rows, "Case 3 at 36 weeks")["study_code"] == "CASE-03-PMA-36"
    assert find_reference_label(rows, "Case 7")["study_code"] == "CASE-07-SINGLE-EXAM"


def test_reference_comparison_excludes_unreported_domains():
    rows = load_reference_labels(LABEL_PATH)
    reference = find_reference_label(rows, "Case 2 day 20")
    evidence = StudyEvidence(
        study_code="Case 2 day 20",
        left=SideEvidence(side="left"),
        right=SideEvidence(
            side="right",
            hemorrhage_present="yes",
            confined_to_germinal_matrix="yes",
            intraventricular_blood="no",
            adjacent_periventricular_echogenicity="no",
            cystic_change="porencephalic",
            clinician_verified=True,
        ),
    )
    observed = classify_study(evidence).to_dict()
    comparisons = compare_to_reference(reference, observed, "expert")
    assert {row["domain"] for row in comparisons} == {
        "right_gmh_ivh",
        "right_pvhi",
        "right_porencephalic_cyst",
        "severe_preterm_brain_injury",
    }
    assert all(row["agreement"] for row in comparisons)
