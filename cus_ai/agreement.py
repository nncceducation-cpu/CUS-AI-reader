from __future__ import annotations

import csv
import io
import json
from collections import Counter
from typing import Any


AGREEMENT_FIELDS = {
    "Left GMH-IVH": ("left", "gmh_ivh"),
    "Left PVHI": ("left", "pvhi"),
    "Left cystic sequela": ("left", "cystic_sequela"),
    "Right GMH-IVH": ("right", "gmh_ivh"),
    "Right PVHI": ("right", "pvhi"),
    "Right cystic sequela": ("right", "cystic_sequela"),
    "White matter injury": ("wmi",),
    "Cerebellar hemorrhage": ("cerebellar_hemorrhage",),
    "PHVD": ("phvd",),
    "Severe injury flag": ("severe_preterm_brain_injury_flag",),
}


def _get(record: dict[str, Any], path: tuple[str, ...]) -> Any:
    value: Any = record
    for item in path:
        value = value[item]
    return value


def compare_classifications(expert: dict[str, Any], ai: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for domain, path in AGREEMENT_FIELDS.items():
        expert_value = _get(expert, path)
        ai_value = _get(ai, path)
        rows.append(
            {
                "domain": domain,
                "expert": expert_value,
                "ai": ai_value,
                "agreement": expert_value == ai_value,
            }
        )
    return rows


def agreement_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    agreements = sum(bool(row["agreement"]) for row in rows)
    return {
        "domains_compared": total,
        "domains_agreeing": agreements,
        "percent_agreement": 100.0 * agreements / total if total else None,
    }


def cohens_kappa(expert_values: list[str], ai_values: list[str]) -> float | None:
    if len(expert_values) != len(ai_values) or len(expert_values) < 2:
        return None
    n = len(expert_values)
    observed = sum(a == b for a, b in zip(expert_values, ai_values)) / n
    expert_counts = Counter(expert_values)
    ai_counts = Counter(ai_values)
    categories = set(expert_counts) | set(ai_counts)
    expected = sum((expert_counts[c] / n) * (ai_counts[c] / n) for c in categories)
    if expected == 1.0:
        return 1.0 if observed == 1.0 else None
    return (observed - expected) / (1.0 - expected)


def _flatten(prefix: str, value: Any, output: dict[str, Any]) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            _flatten(f"{prefix}.{key}" if prefix else key, child, output)
    elif isinstance(value, list):
        output[prefix] = json.dumps(value, separators=(",", ":"))
    else:
        output[prefix] = value


def study_csv(
    expert_evidence: dict[str, Any],
    expert_classification: dict[str, Any],
    ai_result: dict[str, Any] | None,
    model_prediction: dict[str, Any] | None,
    agreement_rows: list[dict[str, Any]],
    reference_label: dict[str, Any] | None = None,
    expert_reference_rows: list[dict[str, Any]] | None = None,
    ai_reference_rows: list[dict[str, Any]] | None = None,
) -> str:
    row: dict[str, Any] = {}
    _flatten("expert_evidence", expert_evidence, row)
    _flatten("expert_classification", expert_classification, row)
    if ai_result:
        _flatten("ai_result", ai_result, row)
    if model_prediction:
        model_study = {key: value for key, value in model_prediction.items() if key != "frame_predictions"}
        _flatten("model", model_study, row)
    _flatten("agreement", agreement_summary(agreement_rows), row)
    if reference_label:
        _flatten("pilot_reference", reference_label, row)
    if expert_reference_rows is not None:
        _flatten("expert_reference_agreement", agreement_summary(expert_reference_rows), row)
    if ai_reference_rows is not None:
        _flatten("ai_reference_agreement", agreement_summary(ai_reference_rows), row)
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(row))
    writer.writeheader()
    writer.writerow(row)
    return buffer.getvalue()


def frame_csv(model_prediction: dict[str, Any] | None) -> str:
    frame_predictions = (model_prediction or {}).get("frame_predictions") or []
    rows: list[dict[str, Any]] = []
    for frame in frame_predictions:
        row = {key: value for key, value in frame.items() if key != "probabilities"}
        for label, probability in (frame.get("probabilities") or {}).items():
            row[f"probability.{label}"] = probability
        rows.append(row)
    if not rows:
        return "source_name,frame_index,plane,plane_confidence,ambiguous_plane\n"
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()
