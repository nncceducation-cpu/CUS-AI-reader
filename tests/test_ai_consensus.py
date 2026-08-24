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
        "right_germinal_matrix_hemorrhage": 0.10,
        "right_confined_to_germinal_matrix": 0.10,
        "right_intraventricular_blood": 0.10,
        "right_ventricular_distension": 0.10,
        "right_ahw_above_6_mm": 0.10,
        "right_ahw_above_10_mm": 0.10,
        "right_focal_periventricular_echogenicity": 0.10,
        "right_echogenicity_brighter_than_choroid": 0.10,
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


def test_uncertain_probability_forces_abstention():
    prediction = complete_prediction()
    prediction["probabilities"]["left_intraventricular_blood"] = 0.50
    result = grade_prediction(prediction, serial_study_available=True)
    assert result.evidence.left.intraventricular_blood == "unknown"
    assert result.abstained
    assert "one or more AI feature decisions are uncertain" in result.abstention_reasons


def test_missing_serial_evidence_forces_final_grade_abstention():
    result = grade_prediction(complete_prediction(), serial_study_available=False)
    assert result.abstained
    assert "serial evidence was not supplied for WMI evolution and PHVD trajectory" in result.abstention_reasons

