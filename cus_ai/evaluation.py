"""Metrics for grading agreement and for threshold selection.

Everything here is deliberately dependency-light: numpy only, no scikit-learn,
so the offline Windows build keeps working. Confidence intervals come from the
percentile bootstrap, resampled at the level of the study rather than the frame,
because frames within a sweep are not independent observations and treating them
as such inflates precision.
"""

from __future__ import annotations

import math
import random
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Sequence


@dataclass(slots=True)
class BinaryMetrics:
    threshold: float
    true_positive: int
    false_positive: int
    true_negative: int
    false_negative: int

    @property
    def sensitivity(self) -> float | None:
        denominator = self.true_positive + self.false_negative
        return self.true_positive / denominator if denominator else None

    @property
    def specificity(self) -> float | None:
        denominator = self.true_negative + self.false_positive
        return self.true_negative / denominator if denominator else None

    @property
    def precision(self) -> float | None:
        denominator = self.true_positive + self.false_positive
        return self.true_positive / denominator if denominator else None

    @property
    def f1(self) -> float | None:
        precision, recall = self.precision, self.sensitivity
        if not precision or not recall or (precision + recall) == 0:
            return None
        return 2 * precision * recall / (precision + recall)

    @property
    def youden_j(self) -> float | None:
        if self.sensitivity is None or self.specificity is None:
            return None
        return self.sensitivity + self.specificity - 1.0

    def to_dict(self) -> dict[str, Any]:
        def rounded(value: float | None) -> float | None:
            return round(value, 4) if value is not None else None

        return {
            "threshold": round(self.threshold, 4),
            "tp": self.true_positive,
            "fp": self.false_positive,
            "tn": self.true_negative,
            "fn": self.false_negative,
            "sensitivity": rounded(self.sensitivity),
            "specificity": rounded(self.specificity),
            "precision": rounded(self.precision),
            "f1": rounded(self.f1),
            "youden_j": rounded(self.youden_j),
        }


def binary_metrics(
    scores: Sequence[float], truth: Sequence[int], threshold: float
) -> BinaryMetrics:
    tp = fp = tn = fn = 0
    for score, label in zip(scores, truth):
        predicted = score >= threshold
        if predicted and label:
            tp += 1
        elif predicted and not label:
            fp += 1
        elif not predicted and label:
            fn += 1
        else:
            tn += 1
    return BinaryMetrics(threshold=threshold, true_positive=tp, false_positive=fp, true_negative=tn, false_negative=fn)


def roc_auc(scores: Sequence[float], truth: Sequence[int]) -> float | None:
    """Area under the ROC curve by the rank-sum identity, ties handled."""
    positives = [s for s, y in zip(scores, truth) if y]
    negatives = [s for s, y in zip(scores, truth) if not y]
    if not positives or not negatives:
        return None
    ordered = sorted(range(len(scores)), key=lambda i: scores[i])
    ranks = [0.0] * len(scores)
    index = 0
    while index < len(ordered):
        stop = index
        while stop + 1 < len(ordered) and scores[ordered[stop + 1]] == scores[ordered[index]]:
            stop += 1
        average_rank = (index + stop) / 2.0 + 1.0
        for position in range(index, stop + 1):
            ranks[ordered[position]] = average_rank
        index = stop + 1
    positive_rank_sum = sum(rank for rank, y in zip(ranks, truth) if y)
    n_pos, n_neg = len(positives), len(negatives)
    return (positive_rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def select_threshold(
    scores: Sequence[float],
    truth: Sequence[int],
    *,
    objective: str = "youden",
    false_negative_cost: float = 4.0,
    min_sensitivity: float | None = None,
) -> tuple[float, BinaryMetrics]:
    """Choose an operating point on labelled data.

    Screening for preterm brain injury is asymmetric. A missed grade III bleed
    costs far more than a second look at a normal sweep, so ``cost`` weights
    false negatives above false positives by default. ``min_sensitivity`` imposes
    a hard floor and picks the most specific threshold that clears it.
    """
    if not scores:
        raise ValueError("Threshold selection needs at least one scored study.")
    candidates = sorted({0.0, 1.0, *scores})
    best_threshold = 0.5
    best_metrics = binary_metrics(scores, truth, 0.5)
    best_value = -math.inf

    for threshold in candidates:
        metrics = binary_metrics(scores, truth, threshold)
        if min_sensitivity is not None:
            sensitivity = metrics.sensitivity
            if sensitivity is None or sensitivity < min_sensitivity:
                continue
            value = metrics.specificity or 0.0
        elif objective == "youden":
            value = metrics.youden_j if metrics.youden_j is not None else -math.inf
        elif objective == "f1":
            value = metrics.f1 if metrics.f1 is not None else -math.inf
        elif objective == "cost":
            value = -(false_negative_cost * metrics.false_negative + metrics.false_positive)
        else:
            raise ValueError(f"Unknown objective: {objective}")
        if value > best_value:
            best_value, best_threshold, best_metrics = value, threshold, metrics
    return best_threshold, best_metrics


def cohens_kappa(a: Sequence[str], b: Sequence[str]) -> float | None:
    if len(a) != len(b) or len(a) < 2:
        return None
    n = len(a)
    observed = sum(x == y for x, y in zip(a, b)) / n
    counts_a, counts_b = Counter(a), Counter(b)
    categories = set(counts_a) | set(counts_b)
    expected = sum((counts_a[c] / n) * (counts_b[c] / n) for c in categories)
    if expected >= 1.0:
        return 1.0 if observed >= 1.0 else None
    return (observed - expected) / (1.0 - expected)


def quadratic_weighted_kappa(a: Sequence[str], b: Sequence[str], order: Sequence[str]) -> float | None:
    """Ordered-grade agreement: confusing grade I with II costs less than I with III."""
    rank = {value: index for index, value in enumerate(order)}
    pairs = [(x, y) for x, y in zip(a, b) if x in rank and y in rank]
    if len(pairs) < 2:
        return None
    n = len(pairs)
    size = len(order)
    if size < 2:
        return None
    observed = [[0.0] * size for _ in range(size)]
    for x, y in pairs:
        observed[rank[x]][rank[y]] += 1
    counts_a = [sum(row) for row in observed]
    counts_b = [sum(observed[i][j] for i in range(size)) for j in range(size)]
    denominator = (size - 1) ** 2
    numerator_o = sum(
        ((i - j) ** 2 / denominator) * observed[i][j] for i in range(size) for j in range(size)
    )
    numerator_e = sum(
        ((i - j) ** 2 / denominator) * counts_a[i] * counts_b[j] / n
        for i in range(size)
        for j in range(size)
    )
    if numerator_e == 0:
        return 1.0 if numerator_o == 0 else None
    return 1.0 - numerator_o / numerator_e


def bootstrap_interval(
    values: Sequence[Any],
    statistic,
    *,
    iterations: int = 2000,
    alpha: float = 0.05,
    seed: int = 20240101,
) -> tuple[float, float] | None:
    """Percentile bootstrap over study-level units."""
    items = list(values)
    if len(items) < 2:
        return None
    rng = random.Random(seed)
    samples: list[float] = []
    for _ in range(iterations):
        resampled = [items[rng.randrange(len(items))] for _ in range(len(items))]
        result = statistic(resampled)
        if result is not None and math.isfinite(result):
            samples.append(float(result))
    if len(samples) < 20:
        return None
    samples.sort()
    low = samples[max(0, int(math.floor((alpha / 2) * len(samples))))]
    high = samples[min(len(samples) - 1, int(math.ceil((1 - alpha / 2) * len(samples))) - 1)]
    return low, high


@dataclass(slots=True)
class DomainAgreement:
    domain: str
    compared: int
    agreeing: int
    kappa: float | None = None
    weighted_kappa: float | None = None
    interval: tuple[float, float] | None = None
    withheld: int = 0
    disagreements: list[dict[str, str]] = field(default_factory=list)

    @property
    def percent_agreement(self) -> float | None:
        return 100.0 * self.agreeing / self.compared if self.compared else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "domain": self.domain,
            "compared": self.compared,
            "agreeing": self.agreeing,
            "withheld": self.withheld,
            "percent_agreement": round(self.percent_agreement, 2) if self.compared else None,
            "cohens_kappa": round(self.kappa, 4) if self.kappa is not None else None,
            "quadratic_weighted_kappa": (
                round(self.weighted_kappa, 4) if self.weighted_kappa is not None else None
            ),
            "kappa_95_ci": (
                [round(self.interval[0], 4), round(self.interval[1], 4)] if self.interval else None
            ),
            "disagreements": list(self.disagreements),
        }


def domain_agreement(
    domain: str,
    pairs: Sequence[tuple[str, str]],
    *,
    withheld: int = 0,
    grade_order: Sequence[str] | None = None,
    study_codes: Sequence[str] | None = None,
) -> DomainAgreement:
    """Agreement for one domain, with a bootstrap interval around kappa.

    Studies the AI withheld are counted separately rather than scored as
    disagreements. A tool that declines to grade an unreadable sweep should not
    be penalised as though it had graded it wrongly.
    """
    reference = [item[0] for item in pairs]
    candidate = [item[1] for item in pairs]
    agreeing = sum(a == b for a, b in pairs)
    codes = list(study_codes or [f"study_{i}" for i in range(len(pairs))])
    disagreements = [
        {"study": codes[index] if index < len(codes) else f"study_{index}", "reference": a, "candidate": b}
        for index, (a, b) in enumerate(pairs)
        if a != b
    ]
    return DomainAgreement(
        domain=domain,
        compared=len(pairs),
        agreeing=agreeing,
        withheld=withheld,
        kappa=cohens_kappa(reference, candidate),
        weighted_kappa=(
            quadratic_weighted_kappa(reference, candidate, grade_order) if grade_order else None
        ),
        interval=bootstrap_interval(
            list(pairs), lambda sample: cohens_kappa([x for x, _ in sample], [y for _, y in sample])
        ),
        disagreements=disagreements,
    )
