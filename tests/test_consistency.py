from cus_ai.consistency import enforce_consistency


def test_nested_thresholds_cannot_invert():
    """Every ventricle wider than 10 mm is also wider than 6 mm."""
    report = enforce_consistency(
        {"left_ahw_above_10_mm": 0.85, "left_ahw_above_6_mm": 0.30}
    )
    assert report.probabilities["left_ahw_above_10_mm"] == 0.30
    assert report.conflicts


def test_containment_hierarchy_is_enforced():
    report = enforce_consistency(
        {"left_intraventricular_blood": 0.90, "left_hemorrhage_present": 0.20}
    )
    assert report.probabilities["left_intraventricular_blood"] == 0.20


def test_consistent_outputs_are_left_alone():
    report = enforce_consistency(
        {"left_ahw_above_10_mm": 0.20, "left_ahw_above_6_mm": 0.80}
    )
    assert report.repairs == []
    assert report.conflicts == []


def test_exclusive_grade_family_is_renormalised():
    report = enforce_consistency(
        {"cbh_none": 0.6, "cbh_punctate": 0.2, "cbh_limited": 0.1, "cbh_large": 0.1},
        softmax_groups=("cerebellar_hemorrhage",),
    )
    total = sum(report.probabilities[name] for name in report.probabilities)
    assert total == 1.0
