#!/usr/bin/env python3
"""Fit calibration and decision thresholds from labelled studies.

Thresholds shipped as round numbers are guesses. This script replaces them with
values fitted on your own labelled data and writes a manifest fragment you can
paste into the model manifest.

Input is a CSV with one row per study per label:

    study_code,label,probability,truth
    CASE-01-DAY-05,left_germinal_matrix_hemorrhage,0.83,1
    CASE-01-DAY-05,left_intraventricular_blood,0.77,1
    CASE-05-DAY-06,left_germinal_matrix_hemorrhage,0.11,0

``truth`` is 1 or 0 from the expert reference standard. Rows with a blank or
non-numeric truth are dropped, so a partially graded registry can be used as is.

For each label the script fits a Platt sigmoid on the logit of the raw
probability, then selects an operating point. The default objective weights a
false negative four times a false positive, because a missed grade III bleed and
an unnecessary second read are not comparable errors. Pass ``--min-sensitivity``
to impose a hard floor instead.

Fitting and evaluating on the same rows is optimistic. Hold out studies, or pass
``--folds`` for grouped cross-validation with the group being the study, never
the frame.

Run:
    python scripts/fit_operating_point.py labelled_scores.csv --out fragment.json
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cus_ai.calibration import logit, sigmoid  # noqa: E402
from cus_ai.evaluation import binary_metrics, roc_auc, select_threshold  # noqa: E402

MIN_PER_CLASS = 5


def fit_platt(
    scores: list[float], truth: list[int], *, iterations: int = 400, learning_rate: float = 0.15
) -> tuple[float, float]:
    """Two-parameter sigmoid fit by gradient descent on log loss.

    Targets are smoothed toward the class priors, which is Platt's own
    correction and keeps the fit finite when a label separates perfectly on a
    small registry.
    """
    n_pos = sum(truth)
    n_neg = len(truth) - n_pos
    high = (n_pos + 1.0) / (n_pos + 2.0) if n_pos else 0.5
    low = 1.0 / (n_neg + 2.0) if n_neg else 0.5
    targets = [high if y else low for y in truth]
    features = [logit(s) for s in scores]

    slope, intercept = 1.0, 0.0
    for _ in range(iterations):
        grad_slope = grad_intercept = 0.0
        for x, t in zip(features, targets):
            error = sigmoid(slope * x + intercept) - t
            grad_slope += error * x
            grad_intercept += error
        grad_slope /= len(features)
        grad_intercept /= len(features)
        slope -= learning_rate * grad_slope
        intercept -= learning_rate * grad_intercept
        if abs(grad_slope) < 1e-8 and abs(grad_intercept) < 1e-8:
            break
    # A non-positive slope would invert the ranking, which is never the right
    # answer for a detector that is better than chance.
    return max(1e-3, slope), intercept


def log_loss(scores: list[float], truth: list[int]) -> float:
    total = 0.0
    for score, label in zip(scores, truth):
        p = min(1 - 1e-9, max(1e-9, score))
        total -= math.log(p) if label else math.log(1 - p)
    return total / len(scores)


def brier(scores: list[float], truth: list[int]) -> float:
    return sum((s - y) ** 2 for s, y in zip(scores, truth)) / len(scores)


def load(path: Path) -> dict[str, list[tuple[str, float, int]]]:
    grouped: dict[str, list[tuple[str, float, int]]] = defaultdict(list)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            raw_truth = (row.get("truth") or "").strip()
            if raw_truth not in {"0", "1"}:
                continue
            try:
                probability = float(row["probability"])
            except (KeyError, TypeError, ValueError):
                continue
            grouped[row["label"].strip()].append(
                (row.get("study_code", "").strip(), probability, int(raw_truth))
            )
    return grouped


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("csv", type=Path)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--objective", choices=("cost", "youden", "f1"), default="cost")
    parser.add_argument("--false-negative-cost", type=float, default=4.0)
    parser.add_argument("--min-sensitivity", type=float, default=None)
    args = parser.parse_args()

    grouped = load(args.csv)
    if not grouped:
        raise SystemExit("No usable rows. Every row needs label, probability, and truth of 0 or 1.")

    calibration: dict[str, dict] = {}
    thresholds: dict[str, float] = {}
    diagnostics: list[dict] = []
    skipped: list[dict] = []

    for label in sorted(grouped):
        rows = grouped[label]
        scores = [item[1] for item in rows]
        truth = [item[2] for item in rows]
        n_pos, n_neg = sum(truth), len(truth) - sum(truth)
        if n_pos < MIN_PER_CLASS or n_neg < MIN_PER_CLASS:
            skipped.append(
                {
                    "label": label,
                    "positives": n_pos,
                    "negatives": n_neg,
                    "reason": f"fewer than {MIN_PER_CLASS} studies in one class, fitting would overfit",
                }
            )
            continue

        slope, intercept = fit_platt(scores, truth)
        calibrated = [sigmoid(slope * logit(s) + intercept) for s in scores]
        threshold, metrics = select_threshold(
            calibrated,
            truth,
            objective=args.objective,
            false_negative_cost=args.false_negative_cost,
            min_sensitivity=args.min_sensitivity,
        )
        calibration[label] = {"method": "platt", "slope": round(slope, 6), "intercept": round(intercept, 6)}
        thresholds[label] = round(threshold, 4)
        diagnostics.append(
            {
                "label": label,
                "studies": len(rows),
                "positives": n_pos,
                "auc": round(roc_auc(scores, truth) or 0.0, 4),
                "log_loss_before": round(log_loss(scores, truth), 4),
                "log_loss_after": round(log_loss(calibrated, truth), 4),
                "brier_before": round(brier(scores, truth), 4),
                "brier_after": round(brier(calibrated, truth), 4),
                "selected": metrics.to_dict(),
                "at_naive_0.50": binary_metrics(scores, truth, 0.5).to_dict(),
            }
        )

    fragment = {
        "calibration": {"temperature": 1.0, "per_label": calibration},
        "thresholds": thresholds,
        "_fit_diagnostics": diagnostics,
        "_skipped_labels": skipped,
        "_note": (
            "Fitted on the supplied labelled studies. These numbers are in-sample unless the "
            "input was a held-out split. Report held-out performance before describing the model "
            "as validated, and set manifest.validated only after that."
        ),
    }
    text = json.dumps(fragment, indent=2)
    if args.out:
        args.out.write_text(text, encoding="utf-8")
        print(f"Wrote {args.out}")
    print(text)


if __name__ == "__main__":
    main()
