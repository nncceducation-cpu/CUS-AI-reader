"""The learning loop must improve on unseen studies, and must refuse to move on noise."""

import random

from cus_ai.evaluation import binary_metrics
from cus_ai.learning import (
    Correction,
    CorrectionStore,
    LearnedParameters,
    ParameterHistory,
    fit_from_corrections,
    training_export,
    truth_from_evidence,
)
from cus_ai.schemas import SideEvidence, StudyEvidence

LABEL = "left_germinal_matrix_hemorrhage"


def stream(n_infants, *, signal=True, seed=5, start=0, blinded=True):
    rng = random.Random(seed)
    out = []
    for index in range(n_infants):
        infant = f"INF-{start + index:03d}"
        for study in range(2):
            truth = 1 if rng.random() < 0.4 else 0
            if signal:
                probability = min(0.999, max(0.001, (0.90 if truth else 0.55) + rng.gauss(0, 0.10)))
            else:
                probability = rng.random()
            out.append(
                Correction(
                    study_code=f"{infant}-S{study}",
                    infant_code=infant,
                    reader_code="R1",
                    recorded_at_utc="2026-01-01T00:00:00+00:00",
                    model_id="m",
                    model_version="1",
                    blinded_to_ai=blinded,
                    probabilities={LABEL: probability},
                    truth={LABEL: truth},
                )
            )
    return out


def score_with(records, learned):
    """Apply learned parameters the way inference does: calibrate, then threshold.

    Thresholds are fitted on calibrated scores. Comparing one against a raw
    probability mixes scales and makes a good fit look like a bad one.
    """
    from cus_ai.calibration import CalibrationSet

    thresholds, calibration = learned.apply_to({LABEL: 0.50}, CalibrationSet())
    return binary_metrics(
        [calibration.calibrate(LABEL, r.probabilities[LABEL]) for r in records],
        [r.truth[LABEL] for r in records],
        thresholds[LABEL],
    )


def score(records, threshold):
    return binary_metrics(
        [r.probabilities[LABEL] for r in records], [r.truth[LABEL] for r in records], threshold
    )


def test_corrections_improve_accuracy_on_studies_never_fitted_on():
    future = stream(40, seed=99, start=900)
    before = score(future, 0.50)
    learned = fit_from_corrections(stream(60), current_thresholds={LABEL: 0.50})
    fit = next(item for item in learned.fits if item["label"] == LABEL)
    assert fit["promoted"] is True
    after = score_with(future, learned)
    assert after.false_positive < before.false_positive
    assert after.specificity > before.specificity + 0.3
    # The floor the screen is required to hold, checked on studies it never saw.
    assert after.sensitivity >= learned.min_sensitivity - 0.05


def test_sensitivity_floor_blocks_a_specificity_grab():
    """A rule that buys specificity by dropping below the floor is refused."""
    learned = fit_from_corrections(
        stream(60), current_thresholds={LABEL: 0.50}, min_sensitivity=0.999
    )
    fit = next(item for item in learned.fits if item["label"] == LABEL)
    assert fit["promoted"] is False
    assert "floor" in fit["reason"]
    assert LABEL not in learned.thresholds


def test_sensitivity_is_not_judged_on_a_handful_of_positives():
    learned = fit_from_corrections(stream(9), current_thresholds={LABEL: 0.50})
    fit = next(item for item in learned.fits if item["label"] == LABEL)
    assert fit["promoted"] is False


def test_every_fit_reports_the_uncertainty_in_its_sensitivity():
    learned = fit_from_corrections(stream(60), current_thresholds={LABEL: 0.50})
    fit = next(item for item in learned.fits if item["label"] == LABEL)
    assert fit["held_out_positives"] >= 12
    assert fit["candidate_sensitivity_lower_bound"] is not None
    # The bound must never be sold as the point estimate.
    assert fit["candidate_sensitivity_lower_bound"] <= fit["candidate_sensitivity"]


def test_learned_threshold_belongs_to_the_calibrated_scale():
    """Guard the mistake that a caller is most likely to make."""
    from cus_ai.calibration import CalibrationSet

    learned = fit_from_corrections(stream(60), current_thresholds={LABEL: 0.50})
    raw = 0.62
    calibrated = learned.score(LABEL, raw, CalibrationSet())
    thresholds, calibration = learned.apply_to({LABEL: 0.50}, CalibrationSet())
    assert calibrated == calibration.calibrate(LABEL, raw)
    assert LABEL in thresholds


def test_noise_is_never_promoted():
    """A threshold search on signal-free scores will find a degenerate rule that
    scores well under an asymmetric cost. The gate has to catch that."""
    for infants in (25, 40, 60):
        learned = fit_from_corrections(
            stream(infants, signal=False, seed=3), current_thresholds={LABEL: 0.50}
        )
        fit = next(item for item in learned.fits if item["label"] == LABEL)
        assert fit["promoted"] is False
        assert LABEL not in learned.thresholds


def test_too_few_corrections_declines_to_fit():
    learned = fit_from_corrections(stream(2), current_thresholds={LABEL: 0.50})
    fit = next(item for item in learned.fits if item["label"] == LABEL)
    assert fit["tier"] == "none"
    assert "needs" in fit["reason"]


def test_unblinded_corrections_are_stored_but_never_fitted():
    """A reader who saw the AI grade first is not an independent label."""
    learned = fit_from_corrections(stream(60, blinded=False), current_thresholds={LABEL: 0.50})
    assert learned.corrections_used == 0
    assert learned.thresholds == {}


def test_store_is_append_only_and_separates_eligible_records(tmp_path):
    store = CorrectionStore(tmp_path / "corrections.jsonl")
    for record in stream(3):
        store.append(record)
    for record in stream(2, blinded=False, start=500):
        store.append(record)
    summary = store.summary()
    assert summary["total_corrections"] == 10
    assert summary["eligible_for_fitting"] == 6
    assert summary["excluded_not_blinded"] == 4
    assert len(store.eligible()) == 6


def test_history_supports_rollback(tmp_path):
    history = ParameterHistory(tmp_path / "learned.json")
    first = LearnedParameters(version=1, thresholds={LABEL: 0.60})
    second = LearnedParameters(version=2, thresholds={LABEL: 0.80})
    history.save(first)
    history.save(second)
    assert history.load().thresholds[LABEL] == 0.80
    restored = history.rollback()
    assert restored is not None and restored.thresholds[LABEL] == 0.60
    assert history.load().thresholds[LABEL] == 0.60


def test_learned_parameters_overlay_the_manifest_without_replacing_it():
    from cus_ai.calibration import CalibrationSet

    learned = LearnedParameters(
        thresholds={LABEL: 0.72},
        calibration={"per_label": {LABEL: {"method": "platt", "slope": 1.4, "intercept": -0.3}}},
    )
    thresholds, calibration = learned.apply_to(
        {LABEL: 0.50, "other_label": 0.45}, CalibrationSet()
    )
    assert thresholds[LABEL] == 0.72
    assert thresholds["other_label"] == 0.45
    assert calibration.per_label[LABEL].method == "platt"


def test_truth_is_read_from_verified_evidence_and_unknowns_are_not_negatives():
    evidence = StudyEvidence(
        study_code="X",
        left=SideEvidence(
            side="left",
            hemorrhage_present="yes",
            intraventricular_blood="no",
            ventricular_distension="unknown",
            clinician_verified=True,
        ),
        right=SideEvidence(side="right", hemorrhage_present="no", clinician_verified=True),
    )
    truth = truth_from_evidence(evidence)
    assert truth["left_germinal_matrix_hemorrhage"] == 1
    assert truth["left_intraventricular_blood"] == 0
    assert truth["right_germinal_matrix_hemorrhage"] == 0
    assert "left_ventricular_distension" not in truth


def test_training_export_flattens_every_correction():
    rows = training_export(stream(2))
    assert len(rows) == 4
    assert {"study_code", "infant_code", "label", "probability", "truth"} <= set(rows[0])
