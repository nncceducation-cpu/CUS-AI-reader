from cus_ai.agreement import agreement_summary, compare_classifications, frame_csv, study_csv


def classification(left_grade: str) -> dict:
    return {
        "left": {"gmh_ivh": left_grade, "pvhi": "Not present"},
        "right": {"gmh_ivh": "Negative for GMH-IVH", "pvhi": "Not present"},
        "wmi": "No ischemic WMI pattern recorded",
        "cerebellar_hemorrhage": "No cerebellar hemorrhage recorded",
        "phvd": "No moderate or severe PHVD by recorded thresholds",
        "severe_preterm_brain_injury_flag": "No severe-injury criterion established from the recorded evidence",
    }


def test_agreement_rows_and_raw_csv_exports():
    expert = classification("Grade I GMH-IVH")
    ai = classification("Grade II GMH-IVH")
    rows = compare_classifications(expert, ai)
    summary = agreement_summary(rows)
    assert summary["domains_compared"] == 8
    assert summary["domains_agreeing"] == 7
    prediction = {
        "model_id": "test",
        "frame_predictions": [
            {
                "source_name": "clip.avi",
                "frame_index": 0,
                "plane": "coronal",
                "plane_confidence": 0.9,
                "ambiguous_plane": False,
                "probabilities": {"left_ivh": 0.8},
            }
        ],
    }
    study = study_csv({"study_code": "CUS-1"}, expert, {"classification": ai}, prediction, rows)
    frames = frame_csv(prediction)
    assert "expert_evidence.study_code" in study
    assert "agreement.percent_agreement" in study
    assert "probability.left_ivh" in frames
    assert "clip.avi" in frames

