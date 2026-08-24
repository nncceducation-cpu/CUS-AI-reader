import pytest

from cus_ai.calibration import CalibrationSet, LabelCalibrator


def test_identity_is_a_no_op():
    calibrators = CalibrationSet()
    assert calibrators.is_identity
    assert calibrators.calibrate("anything", 0.73) == pytest.approx(0.73)


def test_temperature_above_one_reduces_over_confidence():
    calibrators = CalibrationSet(temperature=2.0)
    assert calibrators.calibrate("x", 0.95) < 0.95
    assert calibrators.calibrate("x", 0.05) > 0.05


def test_platt_preserves_ranking():
    calibrator = LabelCalibrator(method="platt", slope=1.8, intercept=-0.9)
    values = [calibrator.apply(p) for p in (0.1, 0.3, 0.5, 0.7, 0.9)]
    assert values == sorted(values)


def test_isotonic_interpolates_between_knots():
    calibrator = LabelCalibrator(method="isotonic", knots_x=[0.0, 0.5, 1.0], knots_y=[0.0, 0.2, 1.0])
    assert calibrator.apply(0.25) == pytest.approx(0.10)
    assert calibrator.apply(0.75) == pytest.approx(0.60)


def test_rank_inverting_calibration_is_rejected():
    with pytest.raises(ValueError):
        LabelCalibrator(method="platt", slope=-1.0)
    with pytest.raises(ValueError):
        LabelCalibrator(method="isotonic", knots_x=[0.0, 1.0], knots_y=[1.0, 0.0])
