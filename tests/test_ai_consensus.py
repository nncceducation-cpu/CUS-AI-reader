from cus_ai.ai_consensus import grade_prediction


def complete_prediction() -> dict:
    probabilities = {
        "left_germinal_matrix_hemorrhage": 0.90,
        "left_confined_to_germinal_matrix": 0.10,
        "left_intraventricular_blood": 0.90,
        "left_ventricular_distension": 0.90,
        "left_ahw_above_6_mm": 0.90,
        "left_ahw_above_10_mm": 0.10,
        "left_focal_periventricular_echogenicity": 0.90,
        "left_echogenicity_brighter_than_choroid": 0.90,
        "left_porencephalic_cyst": 0.10,
        "right_germinal_matrix_hemorrhage": 0.10,
        "right_confined_to_germinal_matrix": 0.10,
        "right_intraventricular_blood": 0.10,
        "right_ventricular_distension": 0.10,
        "right_ahw_above_6_mm": 0.10,
        "right_ahw_above_10_mm": 0.10,
        "right_focal_periventricular_echogenicity": 0.10,
        "right_echogenicity_brighter_than_choroid": 0.10,
        "right_porencephalic_cyst": 0.10,
        "wmi_none": 0.90,
        "wmi_pve_under_7_days": 0.10,
        "wmi_grade_1": 0.10,
        "wmi_grade_2": 0.10,
        "wmi_grade_3": 0.10,
        "wmi_grade_4": 0.10,
        "cbh_none": 0.90,
        "cbh_punctate": 0.10,
        "cbh_limited": 0.10,
        "cbh_large": 0.10,
        "vi_above_97th": 0.10,
        "vi_above_97th_plus_4_mm": 0.10,
    }
    return {
        "model_id": "test-model",
        "model_version": "1.0.0",
        "processed_frame_count": 20,
        "plane_counts": {"coronal": 8, "sagittal": 8, "posterior_fossa": 4},
        "probabilities": probabilities,
        "abstained": False,
        "abstention_reasons": [],
    }


def test_ai_probabilities_enter_same_consensus_rule_engine():
    result = grade_prediction(complete_prediction(), serial_study_available=True)
    assert result.classification.left.gmh_ivh == "Grade III GMH-IVH"
    assert result.classification.left.pvhi == "Present"
    assert result.classification.right.gmh_ivh == "Negative for GMH-IVH"
    assert result.missing_outputs == []
    assert not result.abstained


def test_step_two_is_derived_when_the_model_has_no_confinement_head():
    """The manifest contract has no confinement label, so it must be derived.

    Hemorrhage present with no intraventricular blood is grade I. The previous
    alias lookup left this field unknown on every study and pinned the grade at
    indeterminate.
    """
    prediction = complete_prediction()
    probabilities = prediction["probabilities"]
    probabilities.pop("left_confined_to_germinal_matrix")
    probabilities["left_intraventricular_blood"] = 0.05
    probabilities["left_ventricular_distension"] = 0.05
    probabilities["left_ahw_above_6_mm"] = 0.05
    probabilities["left_focal_periventricular_echogenicity"] = 0.05
    result = grade_prediction(prediction, serial_study_available=True)
    assert result.evidence.left.confined_to_germinal_matrix == "yes"
    assert result.classification.left.gmh_ivh == "Grade I GMH-IVH"
    assert result.feature_decisions["left_confined_to_germinal_matrix"]["source"] == "derived"


def test_uncertain_feature_suppresses_only_its_own_domain():
    """Uncertainty is scoped, not global.

    An intraventricular blood probability sitting on the threshold blocks the
    left grade, but the right hemisphere and the cerebellum were answered
    confidently and are still released.
    """
    prediction = complete_prediction()
    prediction["probabilities"]["left_intraventricular_blood"] = 0.50
    result = grade_prediction(prediction, serial_study_available=True)
    assert result.evidence.left.intraventricular_blood == "unknown"
    assert not result.domain_status["left_gmh_ivh"].reportable
    assert result.domain_status["right_gmh_ivh"].reportable
    assert result.domain_status["cerebellar_hemorrhage"].reportable
    assert not result.abstained
    assert any("dead band" in reason for reason in result.domain_status["left_gmh_ivh"].reasons)


def test_serial_dependent_domains_are_withheld_without_a_serial_study():
    """WMI grade and PHVD are defined by change over time.

    Without a second time point they are withheld, while the hemorrhage grades
    from this single study are still reported.
    """
    result = grade_prediction(complete_prediction(), serial_study_available=False)
    assert not result.domain_status["wmi"].reportable
    assert not result.domain_status["phvd"].reportable
    assert result.domain_status["left_gmh_ivh"].reportable
    assert not result.abstained


def test_absent_posterior_fossa_withholds_only_cerebellar_hemorrhage():
    prediction = complete_prediction()
    prediction["plane_counts"] = {"coronal": 8, "sagittal": 8, "posterior_fossa": 0}
    result = grade_prediction(prediction, serial_study_available=True)
    assert not result.domain_status["cerebellar_hemorrhage"].reportable
    assert result.domain_status["left_gmh_ivh"].reportable


def test_study_abstains_only_when_no_domain_survives():
    prediction = complete_prediction()
    prediction["probabilities"] = {}
    prediction["plane_counts"] = {"coronal": 0, "sagittal": 0, "posterior_fossa": 0}
    result = grade_prediction(prediction, serial_study_available=False)
    assert result.abstained
    assert result.reportable_domains == []
    assert result.abstention_reasons
