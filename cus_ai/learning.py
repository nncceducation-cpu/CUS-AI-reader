"""Learn from expert corrections, without learning the wrong thing.

The loop the clinician sees is simple: the AI proposes a grade, the reader
corrects it, and the next study is graded better. Making that loop actually
improve accuracy, rather than appear to, takes four constraints. Each one exists
because the obvious implementation fails in a specific way.

**Fit only what the data can support.** Corrections arrive a handful at a time.
A decision threshold is one number and moves sensibly on a dozen examples. A
Platt calibration is two and needs a few dozen. Network weights are millions and
need thousands, so this module never touches them; it accumulates a labelled
training set for that job instead. Parameters unlock in tiers as the evidence
arrives, and the tier is stated rather than assumed.

**Never score a fit on the data it was fitted to.** Refitting a threshold on ten
corrections and reporting the accuracy on those same ten will show near-perfect
agreement every time, and the tool will get worse while its own numbers improve.
Every candidate is scored by leave-one-infant-out cross-validation, grouped by
infant, because the same infant contributes several studies and several hundred
correlated frames.

**Change nothing unless the change is earned.** A candidate replaces the
incumbent only when it beats it on held-out folds by a margin, with minimum
counts in both classes. Most of the time, early on, the honest answer is that
there is not enough evidence yet, and the module says so and keeps the current
settings.

**Do not learn the reader's anchoring.** A reader who sees the AI grade first and
then adjusts it is not producing an independent label; they are producing a
lightly edited copy of the model's own output. Fitting on that teaches the model
to agree with itself. Corrections are only eligible for fitting when the reader
recorded their grade blind to the AI. Non-blinded corrections are kept for audit
and excluded from every fit.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

from .calibration import CalibrationSet, LabelCalibrator, logit, sigmoid
from .evaluation import binary_metrics, roc_auc
from .schemas import StudyEvidence

SCHEMA_VERSION = "1.0.0"

# Minimum corrections in the smaller class before a parameter family unlocks.
TIER_THRESHOLD_ONLY = 8
TIER_WITH_CALIBRATION = 20
TIER_WITH_AGGREGATION = 50

# A candidate must beat the incumbent by more than this on the held-out
# objective. Anything smaller is noise on a set this size.
PROMOTION_MARGIN = 0.02

# Beating the incumbent is not sufficient. A threshold search on scores that
# carry no signal will happily degenerate to a constant classifier, calling every
# study positive or every study negative, and under an asymmetric cost that can
# score better than an honest threshold. Three further conditions have to hold.
#
#   1. the candidate must also beat the best constant predictor, so "always say
#      yes" cannot be sold as learning
#   2. the out-of-fold ranking must be better than chance by a permutation test,
#      which is what actually establishes that the probabilities carry signal
#   3. the fitted operating point must not be degenerate on the fitting data
PERMUTATION_ITERATIONS = 400
PERMUTATION_ALPHA = 0.05
MIN_HOLDOUT_AUC = 0.60

# A cost function will trade sensitivity for specificity whenever the arithmetic
# favours it, and for a screen whose purpose is to not miss severe preterm brain
# injury that trade needs a limit.
#
# The limit is an absolute floor, not a rule that sensitivity may never fall. A
# threshold that flags every study has perfect sensitivity and no clinical value,
# and forbidding any decrease from it would forbid all learning. What the
# clinical lead actually owns is the question "how much sensitivity is this
# screen required to hold", and the answer belongs in one place they can set.
#
# Below the floor nothing is promoted, whatever it buys in specificity. If the
# incumbent already sits below the floor, a candidate that moves sensitivity back
# toward it is allowed, because staying put is the worse option.
DEFAULT_MIN_SENSITIVITY = 0.95

# Sensitivity is checked as an out-of-fold point estimate against the floor, and
# separately required to rest on enough held-out positives to mean anything. It
# is deliberately not required to clear the floor by its lower confidence bound:
# a 95% lower bound above 0.95 needs on the order of seventy consecutive correct
# held-out positives, so that rule would keep the gate shut for years and the
# tool would never learn anything. The bound is computed and reported anyway, on
# every fit and in the interface, so that nobody reads "sensitivity 1.00 on
# fourteen positives" as a guarantee it is not.
MIN_HELD_OUT_POSITIVES = 12

# A missed grade III bleed is not the same error as a second look at a normal
# sweep. The objective weights them accordingly.
FALSE_NEGATIVE_COST = 4.0


def _answer_to_truth(answer: str) -> int | None:
    if answer == "yes":
        return 1
    if answer == "no":
        return 0
    return None


def truth_from_evidence(evidence: StudyEvidence) -> dict[str, int]:
    """Read the expert's verified evidence as per-label ground truth.

    Only fields the expert actually settled become labels. An unknown is not a
    negative, and recording it as one would teach the model that uncertainty
    means absence.
    """
    truth: dict[str, int] = {}
    for side_name, side in (("left", evidence.left), ("right", evidence.right)):
        pairs = {
            f"{side_name}_germinal_matrix_hemorrhage": side.hemorrhage_present,
            f"{side_name}_hemorrhage_present": side.hemorrhage_present,
            f"{side_name}_intraventricular_blood": side.intraventricular_blood,
            f"{side_name}_ventricular_distension": side.ventricular_distension,
            f"{side_name}_ahw_above_6_mm": side.ahw_above_6_mm,
            f"{side_name}_ahw_above_10_mm": side.ahw_above_10_mm,
            f"{side_name}_focal_periventricular_echogenicity": side.adjacent_periventricular_echogenicity,
            f"{side_name}_echogenicity_brighter_than_choroid": side.echogenicity_brighter_than_choroid,
        }
        for label, answer in pairs.items():
            value = _answer_to_truth(answer)
            if value is not None:
                truth[label] = value
        if side.cystic_change != "not_assessed":
            truth[f"{side_name}_porencephalic_cyst"] = int(
                side.cystic_change in {"porencephalic", "multiple_evolved_pvhi"}
            )
    for label, answer in (
        ("vi_above_97th", evidence.vi_above_97th),
        ("vi_above_97th_plus_4_mm", evidence.vi_above_97th_plus_4mm),
    ):
        value = _answer_to_truth(answer)
        if value is not None:
            truth[label] = value
    if evidence.wmi_pattern != "not_assessed":
        for label, name in (
            ("wmi_none", "none"),
            ("wmi_pve_under_7_days", "pve_under_7_days"),
            ("wmi_grade_1", "grade_1"),
            ("wmi_grade_2", "grade_2"),
            ("wmi_grade_3", "grade_3"),
            ("wmi_grade_4", "grade_4"),
        ):
            truth[label] = int(evidence.wmi_pattern == name)
    if evidence.cerebellar_hemorrhage != "not_assessed":
        for label, name in (
            ("cbh_none", "none"),
            ("cbh_punctate", "punctate"),
            ("cbh_limited", "limited"),
            ("cbh_large", "large"),
        ):
            truth[label] = int(evidence.cerebellar_hemorrhage == name)
    return truth


@dataclass(slots=True)
class Correction:
    """One study where an expert settled what the AI had guessed."""

    study_code: str
    infant_code: str
    reader_code: str
    recorded_at_utc: str
    model_id: str
    model_version: str
    blinded_to_ai: bool
    probabilities: dict[str, float]
    truth: dict[str, int]
    ai_classification: dict[str, Any] = field(default_factory=dict)
    expert_classification: dict[str, Any] = field(default_factory=dict)
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "Correction":
        return cls(
            study_code=str(raw["study_code"]),
            infant_code=str(raw.get("infant_code") or raw["study_code"]),
            reader_code=str(raw.get("reader_code", "")),
            recorded_at_utc=str(raw.get("recorded_at_utc", "")),
            model_id=str(raw.get("model_id", "")),
            model_version=str(raw.get("model_version", "")),
            blinded_to_ai=bool(raw.get("blinded_to_ai", False)),
            probabilities={k: float(v) for k, v in (raw.get("probabilities") or {}).items()},
            truth={k: int(v) for k, v in (raw.get("truth") or {}).items()},
            ai_classification=raw.get("ai_classification") or {},
            expert_classification=raw.get("expert_classification") or {},
            note=str(raw.get("note", "")),
        )


class CorrectionStore:
    """Append-only JSONL log of corrections.

    Append-only on purpose. A learning loop whose history can be quietly edited
    cannot be audited, and the record of what the model was taught is part of the
    evidence for any claim about how well it performs.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def append(self, correction: Correction) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"schema_version": SCHEMA_VERSION, **correction.to_dict()}
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, separators=(",", ":")) + "\n")

    def load(self) -> list[Correction]:
        if not self.path.exists():
            return []
        records: list[Correction] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                records.append(Correction.from_dict(json.loads(line)))
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                continue
        return records

    def eligible(self) -> list[Correction]:
        """Corrections that may be fitted on: blind reads only."""
        return [item for item in self.load() if item.blinded_to_ai]

    def summary(self) -> dict[str, Any]:
        records = self.load()
        eligible = [item for item in records if item.blinded_to_ai]
        return {
            "total_corrections": len(records),
            "eligible_for_fitting": len(eligible),
            "excluded_not_blinded": len(records) - len(eligible),
            "infants": len({item.infant_code for item in eligible}),
            "readers": sorted({item.reader_code for item in records if item.reader_code}),
            "labels_seen": len({label for item in eligible for label in item.truth}),
        }


@dataclass(slots=True)
class LabelFit:
    label: str
    tier: str
    positives: int
    negatives: int
    threshold: float
    calibration: dict[str, Any] | None
    incumbent_score: float
    candidate_score: float
    promoted: bool
    reason: str
    holdout_auc: float | None = None
    incumbent_sensitivity: float | None = None
    candidate_sensitivity: float | None = None
    candidate_sensitivity_lower_bound: float | None = None
    held_out_positives: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class LearnedParameters:
    """The overlay applied on top of the shipped manifest.

    Kept in its own file. The manifest that came with the model is never
    rewritten, so the vendor's stated operating point stays visible next to
    whatever this site has learned on its own data.
    """

    version: int = 0
    updated_at_utc: str = ""
    corrections_used: int = 0
    infants_used: int = 0
    min_sensitivity: float = DEFAULT_MIN_SENSITIVITY
    thresholds: dict[str, float] = field(default_factory=dict)
    calibration: dict[str, Any] = field(default_factory=dict)
    fits: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "LearnedParameters":
        return cls(
            version=int(raw.get("version", 0)),
            updated_at_utc=str(raw.get("updated_at_utc", "")),
            corrections_used=int(raw.get("corrections_used", 0)),
            infants_used=int(raw.get("infants_used", 0)),
            min_sensitivity=float(raw.get("min_sensitivity", DEFAULT_MIN_SENSITIVITY)),
            thresholds={k: float(v) for k, v in (raw.get("thresholds") or {}).items()},
            calibration=raw.get("calibration") or {},
            fits=list(raw.get("fits") or []),
        )

    def calibration_set(self) -> CalibrationSet:
        return CalibrationSet.from_dict(self.calibration)

    def score(
        self, label: str, raw_probability: float, calibration: CalibrationSet | None = None
    ) -> float:
        """Calibrated probability for one label.

        Thresholds are fitted on calibrated scores, so a caller that compares a
        learned threshold against a raw probability is comparing two different
        scales and will silently get a worse answer than doing nothing. Always
        come through here, or through :meth:`apply_to`, rather than reaching into
        ``thresholds`` alone.
        """
        base = calibration or CalibrationSet()
        _, merged = self.apply_to({}, base)
        return merged.calibrate(label, raw_probability)

    def apply_to(self, thresholds: dict[str, float], calibration: CalibrationSet) -> tuple[
        dict[str, float], CalibrationSet
    ]:
        merged_thresholds = {**thresholds, **self.thresholds}
        merged = CalibrationSet(
            temperature=calibration.temperature, per_label=dict(calibration.per_label)
        )
        for label, entry in (self.calibration.get("per_label") or {}).items():
            merged.per_label[label] = LabelCalibrator.from_dict(entry)
        return merged_thresholds, merged


def _cost(metrics) -> float:
    """Lower is better. Normalised so it can be compared across fold sizes."""
    total = metrics.true_positive + metrics.false_positive + metrics.true_negative + metrics.false_negative
    if total == 0:
        return float("inf")
    return (FALSE_NEGATIVE_COST * metrics.false_negative + metrics.false_positive) / total


def _fit_platt(scores: Sequence[float], truth: Sequence[int], iterations: int = 300) -> tuple[float, float]:
    n_pos = sum(truth)
    n_neg = len(truth) - n_pos
    high = (n_pos + 1.0) / (n_pos + 2.0) if n_pos else 0.5
    low = 1.0 / (n_neg + 2.0) if n_neg else 0.5
    targets = [high if y else low for y in truth]
    features = [logit(s) for s in scores]
    slope, intercept = 1.0, 0.0
    for _ in range(iterations):
        gs = gi = 0.0
        for x, t in zip(features, targets):
            error = sigmoid(slope * x + intercept) - t
            gs += error * x
            gi += error
        slope -= 0.15 * gs / len(features)
        intercept -= 0.15 * gi / len(features)
    return max(1e-3, slope), intercept


def _best_threshold(scores: Sequence[float], truth: Sequence[int]) -> float:
    """Pick the most central threshold among the near-optimal ones.

    Cost as a function of threshold is a step curve, and on a few dozen studies
    long stretches of it are tied. Taking the first minimum lands on whichever
    edge of a plateau the sample happened to produce, and that edge sits right
    next to a training point, so it moves as soon as new data arrives. The middle
    of the plateau is the same cost on the fitting data and considerably steadier
    on the next study.
    """
    candidates = sorted({0.05, 0.5, 0.95, *scores})
    scored = [(candidate, _cost(binary_metrics(scores, truth, candidate))) for candidate in candidates]
    best_value = min(value for _, value in scored)
    plateau = [candidate for candidate, value in scored if value <= best_value + 1e-9]
    return float(plateau[len(plateau) // 2])


def _constant_baseline_cost(truth: Sequence[int]) -> float:
    """Cost of the best classifier that ignores the image entirely."""
    n = len(truth)
    if n == 0:
        return float("inf")
    positives = sum(truth)
    negatives = n - positives
    always_positive = negatives / n
    always_negative = FALSE_NEGATIVE_COST * positives / n
    return min(always_positive, always_negative)


def _out_of_fold_scores(
    scores: Sequence[float], truth: Sequence[int], groups: Sequence[str], use_calibration: bool
) -> tuple[list[float], list[int]]:
    """Predictions for each infant made by a model that never saw that infant."""
    pooled_scores: list[float] = []
    pooled_truth: list[int] = []
    for train, test in _grouped_folds(groups):
        train_scores = [scores[i] for i in train]
        train_truth = [truth[i] for i in train]
        if len(set(train_truth)) < 2:
            continue
        if use_calibration:
            slope, intercept = _fit_platt(train_scores, train_truth)
            pooled_scores.extend(sigmoid(slope * logit(scores[i]) + intercept) for i in test)
        else:
            pooled_scores.extend(scores[i] for i in test)
        pooled_truth.extend(truth[i] for i in test)
    return pooled_scores, pooled_truth


def _permutation_p_value(
    scores: Sequence[float], truth: Sequence[int], observed_auc: float, seed: int = 7
) -> float:
    """How often does shuffled truth reach this ranking performance by luck?"""
    import random as _random

    rng = _random.Random(seed)
    shuffled = list(truth)
    hits = 0
    for _ in range(PERMUTATION_ITERATIONS):
        rng.shuffle(shuffled)
        value = roc_auc(scores, shuffled)
        if value is not None and value >= observed_auc:
            hits += 1
    return (hits + 1) / (PERMUTATION_ITERATIONS + 1)


def _wilson_lower_bound(successes: int, total: int, z: float = 1.96) -> float:
    """Lower 95% bound on a proportion.

    A point estimate of sensitivity from nine held-out positives is not evidence
    that the floor is held; nine out of nine is entirely consistent with a true
    sensitivity of 0.70. Requiring the lower bound to clear the floor is what
    makes the guarantee mean something, and it is why the gate stays shut until
    enough corrected positives have actually accumulated.
    """
    if total <= 0:
        return 0.0
    proportion = successes / total
    denominator = 1 + z * z / total
    centre = proportion + z * z / (2 * total)
    spread = z * ((proportion * (1 - proportion) / total + z * z / (4 * total * total)) ** 0.5)
    return max(0.0, (centre - spread) / denominator)


def _out_of_fold_sensitivity(
    scores: Sequence[float],
    truth: Sequence[int],
    groups: Sequence[str],
    *,
    use_calibration: bool,
    incumbent_threshold: float | None = None,
) -> float | None:
    """Sensitivity of the whole rule, measured only on infants it never saw."""
    true_positive = false_negative = 0
    for train, test in _grouped_folds(groups):
        train_scores = [scores[i] for i in train]
        train_truth = [truth[i] for i in train]
        test_scores = [scores[i] for i in test]
        test_truth = [truth[i] for i in test]
        if len(set(train_truth)) < 2:
            continue
        if incumbent_threshold is not None:
            threshold = incumbent_threshold
        else:
            if use_calibration:
                slope, intercept = _fit_platt(train_scores, train_truth)
                train_scores = [sigmoid(slope * logit(s) + intercept) for s in train_scores]
                test_scores = [sigmoid(slope * logit(s) + intercept) for s in test_scores]
            threshold = _best_threshold(train_scores, train_truth)
        for value, label in zip(test_scores, test_truth):
            if label:
                if value >= threshold:
                    true_positive += 1
                else:
                    false_negative += 1
    total = true_positive + false_negative
    if not total:
        return None
    return (true_positive / total, true_positive, total)


def _is_degenerate(scores: Sequence[float], threshold: float) -> bool:
    """Does this threshold give every study the same answer?"""
    positives = sum(1 for s in scores if s >= threshold)
    return positives == 0 or positives == len(scores)


def _grouped_folds(groups: Sequence[str]) -> list[tuple[list[int], list[int]]]:
    """Leave one infant out. Frames and studies are not independent; infants are."""
    unique = sorted(set(groups))
    folds = []
    for held in unique:
        test = [i for i, g in enumerate(groups) if g == held]
        train = [i for i, g in enumerate(groups) if g != held]
        if train and test:
            folds.append((train, test))
    return folds


def _evaluate(
    scores: Sequence[float],
    truth: Sequence[int],
    groups: Sequence[str],
    *,
    use_calibration: bool,
    incumbent_threshold: float | None = None,
) -> float:
    """Held-out cost. If incumbent_threshold is given, the fixed rule is scored."""
    folds = _grouped_folds(groups)
    if not folds:
        return float("inf")
    total, weight = 0.0, 0
    for train, test in folds:
        train_scores = [scores[i] for i in train]
        train_truth = [truth[i] for i in train]
        test_scores = [scores[i] for i in test]
        test_truth = [truth[i] for i in test]
        if len(set(train_truth)) < 2:
            continue
        if incumbent_threshold is not None:
            fold_cost = _cost(binary_metrics(test_scores, test_truth, incumbent_threshold))
        else:
            if use_calibration:
                slope, intercept = _fit_platt(train_scores, train_truth)
                train_scores = [sigmoid(slope * logit(s) + intercept) for s in train_scores]
                test_scores = [sigmoid(slope * logit(s) + intercept) for s in test_scores]
            threshold = _best_threshold(train_scores, train_truth)
            fold_cost = _cost(binary_metrics(test_scores, test_truth, threshold))
        if fold_cost == float("inf"):
            continue
        total += fold_cost * len(test)
        weight += len(test)
    return total / weight if weight else float("inf")


def fit_from_corrections(
    corrections: Iterable[Correction],
    *,
    current_thresholds: dict[str, float] | None = None,
    current_version: int = 0,
    default_threshold: float = 0.50,
    min_sensitivity: float = DEFAULT_MIN_SENSITIVITY,
) -> LearnedParameters:
    """Refit thresholds and calibration, promoting only what is earned.

    ``min_sensitivity`` is the floor this screen must hold, measured out of fold.
    It is a clinical setting rather than a technical one and should be decided by
    whoever owns the screening programme.
    """
    records = [item for item in corrections if item.blinded_to_ai]
    current_thresholds = current_thresholds or {}
    learned = LearnedParameters(
        version=current_version + 1,
        updated_at_utc=datetime.now(timezone.utc).isoformat(),
        corrections_used=len(records),
        infants_used=len({item.infant_code for item in records}),
        min_sensitivity=min_sensitivity,
    )
    if not records:
        learned.fits.append(
            {"label": "*", "promoted": False, "reason": "no blind corrections have been recorded yet"}
        )
        return learned

    labels = sorted({label for item in records for label in item.truth})
    per_label_calibration: dict[str, Any] = {}

    for label in labels:
        rows = [
            (item.probabilities[label], item.truth[label], item.infant_code)
            for item in records
            if label in item.probabilities and label in item.truth
        ]
        if not rows:
            continue
        scores = [r[0] for r in rows]
        truth = [r[1] for r in rows]
        groups = [r[2] for r in rows]
        positives, negatives = sum(truth), len(truth) - sum(truth)
        smaller = min(positives, negatives)
        infants = len(set(groups))

        if smaller < TIER_THRESHOLD_ONLY or infants < 3:
            learned.fits.append(
                LabelFit(
                    label=label, tier="none", positives=positives, negatives=negatives,
                    threshold=float(current_thresholds.get(label, default_threshold)),
                    calibration=None, incumbent_score=float("nan"), candidate_score=float("nan"),
                    promoted=False,
                    reason=(
                        f"needs {TIER_THRESHOLD_ONLY} corrections in the smaller class across at "
                        f"least 3 infants, has {smaller} across {infants}"
                    ),
                ).to_dict()
            )
            continue

        use_calibration = smaller >= TIER_WITH_CALIBRATION
        tier = "threshold_and_calibration" if use_calibration else "threshold_only"
        incumbent_threshold = float(current_thresholds.get(label, default_threshold))
        incumbent = _evaluate(scores, truth, groups, use_calibration=False,
                              incumbent_threshold=incumbent_threshold)
        candidate = _evaluate(scores, truth, groups, use_calibration=use_calibration)

        constant = _constant_baseline_cost(truth)
        pooled_scores, pooled_truth = _out_of_fold_scores(scores, truth, groups, use_calibration)
        holdout_auc = roc_auc(pooled_scores, pooled_truth) if pooled_scores else None
        p_value = (
            _permutation_p_value(pooled_scores, pooled_truth, holdout_auc)
            if holdout_auc is not None
            else 1.0
        )

        incumbent_result = _out_of_fold_sensitivity(
            scores, truth, groups, use_calibration=False, incumbent_threshold=incumbent_threshold
        )
        candidate_result = _out_of_fold_sensitivity(
            scores, truth, groups, use_calibration=use_calibration
        )
        incumbent_sensitivity = incumbent_result[0] if incumbent_result else None
        candidate_sensitivity = candidate_result[0] if candidate_result else None
        candidate_lower_bound = (
            _wilson_lower_bound(candidate_result[1], candidate_result[2]) if candidate_result else None
        )

        if candidate_result is None or candidate_lower_bound is None:
            sensitivity_safe = False
        elif candidate_result[2] < MIN_HELD_OUT_POSITIVES:
            sensitivity_safe = False
        elif candidate_sensitivity >= min_sensitivity:
            sensitivity_safe = True
        elif incumbent_sensitivity is not None and incumbent_sensitivity < min_sensitivity:
            # Already below the floor: allow anything that climbs back toward it.
            sensitivity_safe = candidate_sensitivity >= incumbent_sensitivity
        else:
            sensitivity_safe = False

        beats_incumbent = candidate + PROMOTION_MARGIN < incumbent
        beats_constant = candidate + PROMOTION_MARGIN < constant
        has_signal = (
            holdout_auc is not None
            and holdout_auc >= MIN_HOLDOUT_AUC
            and p_value < PERMUTATION_ALPHA
        )
        promoted = beats_incumbent and beats_constant and has_signal and sensitivity_safe

        blockers: list[str] = []
        if not sensitivity_safe:
            if candidate_result is None:
                blockers.append("no held-out positive studies, so sensitivity cannot be checked")
            elif candidate_result[2] < MIN_HELD_OUT_POSITIVES:
                blockers.append(
                    f"only {candidate_result[2]} held-out positive studies, and sensitivity is not "
                    f"checked against the floor on fewer than {MIN_HELD_OUT_POSITIVES}"
                )
            else:
                blockers.append(
                    f"out-of-fold sensitivity is {candidate_sensitivity:.3f} on "
                    f"{candidate_result[2]} held-out positives, lower bound "
                    f"{candidate_lower_bound:.3f}, which does not clear the {min_sensitivity:.2f} "
                    "floor this screen is required to hold"
                )
        if not beats_incumbent:
            blockers.append(
                f"held-out cost {candidate:.3f} did not beat the current {incumbent:.3f}"
            )
        if not beats_constant:
            blockers.append(
                f"held-out cost {candidate:.3f} did not beat simply guessing the same answer "
                f"every time, which costs {constant:.3f}"
            )
        if holdout_auc is None or holdout_auc < MIN_HOLDOUT_AUC:
            blockers.append(
                f"out-of-fold ranking {holdout_auc if holdout_auc is None else round(holdout_auc, 3)} "
                f"is below the {MIN_HOLDOUT_AUC} floor, so these probabilities carry too little signal"
            )
        elif p_value >= PERMUTATION_ALPHA:
            blockers.append(
                f"shuffled labels reach the same ranking {p_value:.3f} of the time, so the "
                "apparent improvement is within chance"
            )

        fitted_threshold = incumbent_threshold
        calibration_entry = None
        if promoted:
            fit_scores = list(scores)
            if use_calibration:
                slope, intercept = _fit_platt(scores, truth)
                calibration_entry = {
                    "method": "platt",
                    "slope": round(slope, 6),
                    "intercept": round(intercept, 6),
                }
                fit_scores = [sigmoid(slope * logit(s) + intercept) for s in scores]
            fitted_threshold = _best_threshold(fit_scores, truth)
            if _is_degenerate(fit_scores, fitted_threshold):
                promoted = False
                fitted_threshold = incumbent_threshold
                calibration_entry = None
                blockers.append(
                    "the fitted threshold gives every study the same answer, which is a "
                    "degenerate rule rather than a learned one"
                )
            else:
                learned.thresholds[label] = round(fitted_threshold, 4)
                if calibration_entry:
                    per_label_calibration[label] = calibration_entry

        learned.fits.append(
            LabelFit(
                label=label, tier=tier, positives=positives, negatives=negatives,
                threshold=round(fitted_threshold, 4), calibration=calibration_entry,
                incumbent_score=round(incumbent, 4), candidate_score=round(candidate, 4),
                promoted=promoted,
                reason=(
                    f"held-out cost improved from {incumbent:.3f} to {candidate:.3f}, beating the "
                    f"constant baseline of {constant:.3f}, out-of-fold ranking "
                    f"{holdout_auc:.3f} with permutation p={p_value:.3f}"
                    if promoted
                    else "; ".join(blockers) + ". Nothing changed."
                ),
                holdout_auc=round(holdout_auc, 4) if holdout_auc is not None else None,
                incumbent_sensitivity=(
                    round(incumbent_sensitivity, 4) if incumbent_sensitivity is not None else None
                ),
                candidate_sensitivity=(
                    round(candidate_sensitivity, 4) if candidate_sensitivity is not None else None
                ),
                candidate_sensitivity_lower_bound=(
                    round(candidate_lower_bound, 4) if candidate_lower_bound is not None else None
                ),
                held_out_positives=candidate_result[2] if candidate_result else 0,
            ).to_dict()
        )

    if per_label_calibration:
        learned.calibration = {"temperature": 1.0, "per_label": per_label_calibration}
    return learned


class ParameterHistory:
    """Versioned learned parameters, with rollback."""

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def load(self) -> LearnedParameters:
        if not self.path.exists():
            return LearnedParameters()
        raw = json.loads(self.path.read_text(encoding="utf-8"))
        return LearnedParameters.from_dict(raw.get("current") or raw)

    def versions(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        raw = json.loads(self.path.read_text(encoding="utf-8"))
        return list(raw.get("history") or [])

    def save(self, learned: LearnedParameters) -> None:
        history = self.versions()
        current = self.load()
        if current.version:
            history.append(current.to_dict())
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps({"current": learned.to_dict(), "history": history[-20:]}, indent=2),
            encoding="utf-8",
        )

    def rollback(self) -> LearnedParameters | None:
        history = self.versions()
        if not history:
            return None
        previous = LearnedParameters.from_dict(history[-1])
        self.path.write_text(
            json.dumps({"current": previous.to_dict(), "history": history[:-1]}, indent=2),
            encoding="utf-8",
        )
        return previous


def training_export(corrections: Iterable[Correction]) -> list[dict[str, Any]]:
    """Flatten the correction log into rows for training a real model later.

    This is the durable value of the loop. Threshold tuning helps at the margin;
    a few hundred of these rows is what eventually lets a network be trained or
    fine-tuned on Canadian consensus targets.
    """
    rows: list[dict[str, Any]] = []
    for item in corrections:
        for label, value in sorted(item.truth.items()):
            rows.append(
                {
                    "study_code": item.study_code,
                    "infant_code": item.infant_code,
                    "reader_code": item.reader_code,
                    "recorded_at_utc": item.recorded_at_utc,
                    "blinded_to_ai": item.blinded_to_ai,
                    "label": label,
                    "probability": item.probabilities.get(label),
                    "truth": value,
                }
            )
    return rows
