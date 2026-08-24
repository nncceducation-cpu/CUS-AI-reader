#!/usr/bin/env python3
"""Quantify what the frame-to-study aggregation change does to error rates.

This is a simulation, not a clinical validation. It does not tell you how well
any model reads an ultrasound. It isolates one question that can be answered
without labelled images: given per-frame detector outputs of a stated quality,
how often does each pooling rule turn them into the wrong study-level answer?

The generative model reflects how sweeps actually behave:

* a positive study shows the lesion on a contiguous run of frames, because the
  probe passes through the lesion once per sweep
* a negative study still produces isolated high-scoring frames, because speckle,
  the choroid plexus, and the caudothalamic notch all mimic clot on single frames
* both are corrupted by per-frame detector noise

Two rules are compared at matched sensitivity so the comparison is not simply a
threshold shift:

    legacy       mean of the k highest frame probabilities (k = 3)
    persistence  min(k-th largest, highest level sustained over a run)

Run:  python scripts/benchmark_aggregation.py
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cus_ai.aggregation import AggregationConfig, aggregate_labels  # noqa: E402
from cus_ai.evaluation import binary_metrics, roc_auc, select_threshold  # noqa: E402

LABEL = "left_germinal_matrix_hemorrhage"


def synthesise_sweep(
    rng: random.Random,
    *,
    positive: bool,
    frames: int,
    lesion_run: int,
    detector_sensitivity: float,
    detector_noise: float,
    speckle_rate: float,
    subtle_fraction: float,
    speckle_burst_max: int,
) -> list[float]:
    """One sweep of per-frame probabilities for a single label.

    Two sources of difficulty are modelled deliberately, because without them
    the comparison is trivial and the result would not mean anything. A share of
    positive studies carry a subtle lesion the detector barely registers, which
    is where persistence rules risk losing sensitivity. Artifact arrives in short
    bursts of consecutive frames rather than as single spikes, which is the
    regime persistence rules find hardest.
    """
    subtle = positive and rng.random() < subtle_fraction
    per_study_sensitivity = (
        rng.uniform(0.30, 0.50) if subtle else rng.uniform(detector_sensitivity - 0.10, min(0.97, detector_sensitivity + 0.18))
    )

    lesion_frames: set[int] = set()
    if positive:
        start = rng.randrange(0, max(1, frames - lesion_run))
        lesion_frames = set(range(start, min(frames, start + lesion_run)))

    speckle_frames: set[int] = set()
    index = 0
    while index < frames:
        if rng.random() < speckle_rate:
            burst = rng.randint(1, max(1, speckle_burst_max))
            speckle_frames.update(range(index, min(frames, index + burst)))
            index += burst
        else:
            index += 1

    values = []
    for position in range(frames):
        if position in lesion_frames:
            base = per_study_sensitivity
        elif position in speckle_frames:
            base = rng.uniform(0.70, 0.97)
        else:
            base = rng.uniform(0.03, 0.15)
        value = base + rng.gauss(0.0, detector_noise)
        values.append(min(0.999, max(0.001, value)))
    return values


def legacy_top_k(values: list[float], k: int = 3) -> float:
    ordered = sorted(values, reverse=True)[:k]
    return float(statistics.fmean(ordered)) if ordered else 0.0


def persistence_score(values: list[float], config: AggregationConfig) -> float:
    rows = [
        {
            "plane": "coronal",
            "source_name": "sweep",
            "frame_index": index,
            "weight": 1.0,
            "probabilities": {LABEL: value},
        }
        for index, value in enumerate(values)
    ]
    return aggregate_labels(rows, [LABEL], config)[LABEL].probability


def run(args: argparse.Namespace) -> dict:
    rng = random.Random(args.seed)
    config = AggregationConfig(
        min_frames=args.min_frames,
        persistence_fraction=args.persistence_fraction,
        min_run=args.min_run,
        require_two_planes=False,
    )

    truth: list[int] = []
    legacy_scores: list[float] = []
    new_scores: list[float] = []

    for index in range(args.studies):
        positive = index < int(args.studies * args.prevalence)
        values = synthesise_sweep(
            rng,
            positive=positive,
            frames=rng.randint(args.min_frames_per_sweep, args.max_frames_per_sweep),
            lesion_run=rng.randint(args.lesion_run_min, args.lesion_run_max),
            detector_sensitivity=args.detector_sensitivity,
            detector_noise=args.detector_noise,
            speckle_rate=args.speckle_rate,
            subtle_fraction=args.subtle_fraction,
            speckle_burst_max=args.speckle_burst_max,
        )
        truth.append(int(positive))
        legacy_scores.append(legacy_top_k(values, args.legacy_k))
        new_scores.append(persistence_score(values, config))

    report: dict = {
        "simulation": {
            "studies": args.studies,
            "prevalence": args.prevalence,
            "frames_per_sweep": [args.min_frames_per_sweep, args.max_frames_per_sweep],
            "lesion_run_frames": [args.lesion_run_min, args.lesion_run_max],
            "detector_sensitivity": args.detector_sensitivity,
            "detector_noise": args.detector_noise,
            "speckle_rate_per_frame": args.speckle_rate,
            "speckle_burst_max_frames": args.speckle_burst_max,
            "subtle_lesion_fraction": args.subtle_fraction,
            "seed": args.seed,
        },
        "rules": {},
    }

    for name, scores in (("legacy_top_k_mean", legacy_scores), ("persistence", new_scores)):
        auc = roc_auc(scores, truth)
        at_half = binary_metrics(scores, truth, 0.5)
        matched_threshold, matched = select_threshold(
            scores, truth, min_sensitivity=args.matched_sensitivity
        )
        report["rules"][name] = {
            "auc": round(auc, 4) if auc is not None else None,
            "at_threshold_0.50": at_half.to_dict(),
            f"at_matched_sensitivity_{args.matched_sensitivity}": matched.to_dict(),
            "matched_threshold": round(matched_threshold, 4),
        }

    legacy_fp = report["rules"]["legacy_top_k_mean"]["at_threshold_0.50"]["fp"]
    new_fp = report["rules"]["persistence"]["at_threshold_0.50"]["fp"]
    legacy_key = f"at_matched_sensitivity_{args.matched_sensitivity}"
    report["headline"] = {
        "false_positives_at_0.50": {"legacy": legacy_fp, "persistence": new_fp},
        "false_positive_reduction_at_0.50": (
            round(100.0 * (legacy_fp - new_fp) / legacy_fp, 1) if legacy_fp else None
        ),
        "specificity_at_matched_sensitivity": {
            "legacy": report["rules"]["legacy_top_k_mean"][legacy_key]["specificity"],
            "persistence": report["rules"]["persistence"][legacy_key]["specificity"],
        },
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--studies", type=int, default=1000)
    parser.add_argument("--prevalence", type=float, default=0.30)
    parser.add_argument("--min-frames-per-sweep", type=int, default=60)
    parser.add_argument("--max-frames-per-sweep", type=int, default=400)
    parser.add_argument("--lesion-run-min", type=int, default=6)
    parser.add_argument("--lesion-run-max", type=int, default=30)
    parser.add_argument("--detector-sensitivity", type=float, default=0.72)
    parser.add_argument("--detector-noise", type=float, default=0.12)
    parser.add_argument("--speckle-rate", type=float, default=0.005)
    parser.add_argument("--speckle-burst-max", type=int, default=3)
    parser.add_argument("--subtle-fraction", type=float, default=0.30)
    parser.add_argument("--legacy-k", type=int, default=3)
    parser.add_argument("--min-frames", type=int, default=2)
    parser.add_argument("--min-run", type=int, default=3)
    parser.add_argument("--persistence-fraction", type=float, default=0.02)
    parser.add_argument("--matched-sensitivity", type=float, default=0.90)
    parser.add_argument("--seed", type=int, default=20240101)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="report across a range of artifact rates instead of a single setting, "
        "so the result is not read as one tuned number",
    )
    args = parser.parse_args()

    if args.sweep:
        rows = []
        for rate in (0.002, 0.005, 0.010, 0.020):
            for burst in (2, 3):
                args.speckle_rate = rate
                args.speckle_burst_max = burst
                single = run(args)
                key = f"at_matched_sensitivity_{args.matched_sensitivity}"
                rows.append(
                    {
                        "artifact_rate_per_frame": rate,
                        "artifact_burst_max_frames": burst,
                        "auc_legacy": single["rules"]["legacy_top_k_mean"]["auc"],
                        "auc_persistence": single["rules"]["persistence"]["auc"],
                        "specificity_legacy": single["rules"]["legacy_top_k_mean"][key]["specificity"],
                        "specificity_persistence": single["rules"]["persistence"][key]["specificity"],
                    }
                )
        report = {"matched_sensitivity": args.matched_sensitivity, "sweep": rows}
        text = json.dumps(report, indent=2)
        if args.out:
            args.out.write_text(text, encoding="utf-8")
        print(text)
        return

    report = run(args)
    text = json.dumps(report, indent=2)
    if args.out:
        args.out.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
