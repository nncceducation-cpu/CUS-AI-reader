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


def test_final_status_requires_verified_planes_and_serial_domains():
    evidence = StudyEvidence(
        study_code="G",
        left=side(confined_to_germinal_matrix="yes", intraventricular_blood="no"),
        right=blank_right(),
        wmi_pattern="none",
        cerebellar_hemorrhage="none",
        prior_gmh_ivh="yes",
        vi_above_97th="no",
        vi_above_97th_plus_4mm="no",
        coronal_views_complete=True,
        sagittal_views_complete=True,
        posterior_fossa_views_complete=True,
        complete_required_views=True,
        all_frames_processed=True,
        decoded_frame_count=12,
        serial_study_available=True,
    )
    result = classify_study(evidence)
    assert result.classification_status.startswith("Final consensus classification")
    assert result.view_coverage["coronal"] is True


def test_porencephalic_cyst_is_recorded_as_evolved_pvhi_not_ischemic_wmi():
    evidence = StudyEvidence(
        study_code="H",
        left=SideEvidence(
            side="left",
            hemorrhage_present="no",
            adjacent_periventricular_echogenicity="no",
            cystic_change="porencephalic",
            clinician_verified=True,
        ),
        right=blank_right(),
        wmi_pattern="none",
    )
    result = classify_study(evidence)
    assert result.left.pvhi == "Evolved PVHI with porencephalic cyst"
    assert result.left.cystic_sequela == "Porencephalic cyst, consistent with evolved PVHI"
    assert result.wmi == "No ischemic WMI pattern recorded"
    assert "left" in result.severe_preterm_brain_injury_flag


def test_normal_ventricles_without_prior_hemorrhage_are_negative_for_phvd():
    evidence = StudyEvidence(
        study_code="I",
        left=SideEvidence(
            side="left",
            hemorrhage_present="no",
            adjacent_periventricular_echogenicity="no",
            clinician_verified=True,
        ),
        right=blank_right(),
        prior_gmh_ivh="no",
        vi_above_97th="no",
        vi_above_97th_plus_4mm="no",
    )
    assert classify_study(evidence).phvd == "No moderate or severe PHVD by recorded thresholds"
