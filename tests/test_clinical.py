from cus_ai.clinical import classify_study
from cus_ai.schemas import SideEvidence, StudyEvidence


def side(**kwargs):
    defaults = dict(
        side="left",
        hemorrhage_present="yes",
        confined_to_germinal_matrix="no",
        intraventricular_blood="yes",
        ventricular_distension="no",
        ahw_mm=4.0,
        adjacent_periventricular_echogenicity="no",
        clinician_verified=True,
    )
    defaults.update(kwargs)
    return SideEvidence(**defaults)


def blank_right():
    return SideEvidence(
        side="right",
        hemorrhage_present="no",
        adjacent_periventricular_echogenicity="no",
        clinician_verified=True,
    )


def test_grade_i_requires_germinal_matrix_only_and_no_ivh():
    evidence = StudyEvidence(
        study_code="A",
        left=side(confined_to_germinal_matrix="yes", intraventricular_blood="no"),
        right=blank_right(),
    )
    assert classify_study(evidence).left.gmh_ivh == "Grade I GMH-IVH"


def test_grade_ii_when_ivh_does_not_meet_both_grade_iii_criteria():
    evidence = StudyEvidence(study_code="B", left=side(ventricular_distension="no", ahw_mm=7.0), right=blank_right())
    assert classify_study(evidence).left.gmh_ivh == "Grade II GMH-IVH"


def test_ahw_exactly_six_is_not_grade_iii():
    evidence = StudyEvidence(study_code="C", left=side(ventricular_distension="yes", ahw_mm=6.0), right=blank_right())
    assert classify_study(evidence).left.gmh_ivh == "Grade II GMH-IVH"


def test_grade_iii_requires_distension_and_ahw_above_six():
    evidence = StudyEvidence(study_code="D", left=side(ventricular_distension="yes", ahw_mm=6.1), right=blank_right())
    result = classify_study(evidence)
    assert result.left.gmh_ivh == "Grade III GMH-IVH"
    assert "left" in result.severe_preterm_brain_injury_flag


def test_pvhi_requires_gmh_ivh_on_same_side():
    evidence = StudyEvidence(
        study_code="E",
        left=side(adjacent_periventricular_echogenicity="yes"),
        right=SideEvidence(
            side="right",
            hemorrhage_present="no",
            adjacent_periventricular_echogenicity="yes",
            echogenicity_brighter_than_choroid="yes",
            clinician_verified=True,
        ),
    )
    result = classify_study(evidence)
    assert result.left.pvhi == "Present"
    assert result.right.pvhi.startswith("No PVHI")


def test_wmi_and_phvd_thresholds_are_serial_domains():
    evidence = StudyEvidence(
        study_code="F",
        left=side(ahw_mm=11.0),
        right=blank_right(),
        wmi_pattern="grade_3",
        prior_gmh_ivh="yes",
        vi_above_97th="yes",
        vi_above_97th_plus_4mm="no",
    )
    result = classify_study(evidence)
    assert result.wmi.startswith("Grade 3 WMI")
    assert result.phvd == "Severe PHVD"

