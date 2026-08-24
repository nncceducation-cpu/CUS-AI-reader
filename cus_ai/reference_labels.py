from __future__ import annotations

import csv
import io
import re
from pathlib import Path
from typing import Any


REFERENCE_FIELDS = (
    "left_gmh_ivh",
    "left_pvhi",
    "left_porencephalic_cyst",
    "right_gmh_ivh",
    "right_pvhi",
    "right_porencephalic_cyst",
    "wmi",
    "cerebellar_hemorrhage",
    "phvd",
    "severe_preterm_brain_injury",
)

ALLOWED_VALUES = {
    "left_gmh_ivh": {"negative", "grade_1", "grade_2", "grade_3", "indeterminate", "not_reported"},
    "right_gmh_ivh": {"negative", "grade_1", "grade_2", "grade_3", "indeterminate", "not_reported"},
    "left_pvhi": {"absent", "present", "evolved", "indeterminate", "not_reported"},
    "right_pvhi": {"absent", "present", "evolved", "indeterminate", "not_reported"},
    "left_porencephalic_cyst": {"absent", "present", "indeterminate", "not_reported"},
    "right_porencephalic_cyst": {"absent", "present", "indeterminate", "not_reported"},
    "wmi": {"none", "pve_under_7_days", "grade_1", "grade_2", "grade_3", "grade_4", "indeterminate", "not_reported"},
    "cerebellar_hemorrhage": {"none", "punctate", "limited", "large", "indeterminate", "not_reported"},
    "phvd": {"none", "moderate", "severe", "not_phvd", "indeterminate", "not_reported"},
    "severe_preterm_brain_injury": {"yes", "no", "indeterminate"},
}

DISPLAY_VALUES = {
    "negative": "Negative",
    "grade_1": "Grade I",
    "grade_2": "Grade II",
    "grade_3": "Grade III",
    "absent": "Absent",
    "present": "Present",
    "evolved": "Evolved to porencephalic cyst",
    "none": "None",
    "pve_under_7_days": "PVE under 7 days or duration not established",
    "grade_4": "Grade 4",
    "punctate": "Punctate",
    "limited": "Limited",
    "large": "Large",
    "moderate": "Moderate",
    "severe": "Severe",
    "not_phvd": "Not PHVD",
    "yes": "Yes",
    "no": "No",
    "indeterminate": "Indeterminate",
    "not_reported": "Not reported",
}


def normalize_study_code(value: str) -> str:
    text = re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()
    match = re.search(r"case\s*0*(\d+).*?day\s*0*(\d+)", text)
    if match:
        return f"CASE-{int(match.group(1)):02d}-DAY-{int(match.group(2)):02d}"
    match = re.search(r"case\s*0*(\d+).*?(?:week|wk)\s*0*(\d+)", text)
    if match:
        case_number, week = int(match.group(1)), int(match.group(2))
        if case_number == 3 and week == 36:
            return "CASE-03-PMA-36"
        return f"CASE-{case_number:02d}-WEEK-{week:02d}"
    match = re.search(r"case\s*0*3.*?36", text)
    if match:
        return "CASE-03-PMA-36"
    match = re.search(r"case\s*0*7(?:\D|$)", text)
    if match:
        return "CASE-07-SINGLE-EXAM"
    return re.sub(r"[^A-Z0-9]+", "-", value.upper()).strip("-")


def load_reference_labels(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("Reference label file is empty.")
    missing_columns = set(REFERENCE_FIELDS) - set(rows[0])
    if missing_columns:
        raise ValueError("Reference label file is missing columns: " + ", ".join(sorted(missing_columns)))
    study_codes: set[str] = set()
    for row_number, row in enumerate(rows, start=2):
        study_code = row.get("study_code", "").strip()
        if not study_code:
            raise ValueError(f"Row {row_number} has no study_code.")
        if study_code in study_codes:
            raise ValueError(f"Duplicate study_code: {study_code}")
        study_codes.add(study_code)
        for field in REFERENCE_FIELDS:
            value = row.get(field, "").strip()
            if value not in ALLOWED_VALUES[field]:
                raise ValueError(f"Row {row_number}, {field}: invalid value {value!r}")
    return rows


def find_reference_label(rows: list[dict[str, str]], study_code: str) -> dict[str, str] | None:
    target = normalize_study_code(study_code)
    for row in rows:
        aliases = [row["study_code"], *(row.get("study_aliases", "").split("|"))]
        if target in {normalize_study_code(alias) for alias in aliases if alias.strip()}:
            return row
    return None


def reference_display_row(row: dict[str, str]) -> dict[str, str]:
    output = {
        "Study": row["study_code"],
        "Provided label": row["raw_label"],
        "Reference status": row["reference_status"],
    }
    for field in REFERENCE_FIELDS:
        output[field] = DISPLAY_VALUES.get(row[field], row[field])
    output["Normalization notes"] = row.get("normalization_notes", "")
    return output


def reference_csv(rows: list[dict[str, str]]) -> str:
    if not rows:
        return ""
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def _observed_codes(classification: dict[str, Any]) -> dict[str, str]:
    gmh = {
        "Negative for GMH-IVH": "negative",
        "Grade I GMH-IVH": "grade_1",
        "Grade II GMH-IVH": "grade_2",
        "Grade III GMH-IVH": "grade_3",
    }
    pvhi = {
        "Not present": "absent",
        "Present": "present",
        "Evolved PVHI with porencephalic cyst": "evolved",
        "PVHI with porencephalic cyst": "present",
    }
    cyst = {
        "No porencephalic cyst recorded": "absent",
        "Porencephalic cyst, consistent with evolved PVHI": "present",
    }
    wmi = {
        "No ischemic WMI pattern recorded": "none",
        "PVE present, duration below 7 days or not yet established": "pve_under_7_days",
    }
    for grade in range(1, 5):
        wmi[f"Grade {grade} WMI"] = f"grade_{grade}"
    cbh = {
        "No cerebellar hemorrhage recorded": "none",
        "Punctate CBH, 4 mm or smaller, usually MRI detected": "punctate",
        "Limited CBH, above 4 mm and below one third of a cerebellar hemisphere": "limited",
        "Large CBH, one third or more of a cerebellar hemisphere": "large",
    }
    phvd_value = classification.get("phvd", "")
    if phvd_value == "Severe PHVD":
        phvd_code = "severe"
    elif phvd_value == "Moderate PHVD":
        phvd_code = "moderate"
    elif phvd_value.startswith("No moderate or severe PHVD"):
        phvd_code = "none"
    elif phvd_value.startswith("Not PHVD"):
        phvd_code = "not_phvd"
    else:
        phvd_code = "indeterminate"
    severe_text = classification.get("severe_preterm_brain_injury_flag", "")
    severe_code = "yes" if severe_text.startswith("Criteria recorded") else (
        "no" if severe_text.startswith("No severe-injury criterion") else "indeterminate"
    )
    output: dict[str, str] = {
        "left_gmh_ivh": gmh.get(classification["left"].get("gmh_ivh", ""), "indeterminate"),
        "left_pvhi": pvhi.get(classification["left"].get("pvhi", ""), "indeterminate"),
        "left_porencephalic_cyst": cyst.get(classification["left"].get("cystic_sequela", ""), "indeterminate"),
        "right_gmh_ivh": gmh.get(classification["right"].get("gmh_ivh", ""), "indeterminate"),
        "right_pvhi": pvhi.get(classification["right"].get("pvhi", ""), "indeterminate"),
        "right_porencephalic_cyst": cyst.get(classification["right"].get("cystic_sequela", ""), "indeterminate"),
        "wmi": "indeterminate",
        "cerebellar_hemorrhage": cbh.get(classification.get("cerebellar_hemorrhage", ""), "indeterminate"),
        "phvd": phvd_code,
        "severe_preterm_brain_injury": severe_code,
    }
    wmi_text = classification.get("wmi", "")
    for prefix, code in wmi.items():
        if wmi_text.startswith(prefix):
            output["wmi"] = code
            break
    return output


def compare_to_reference(
    reference: dict[str, str], classification: dict[str, Any], observer: str
) -> list[dict[str, Any]]:
    observed = _observed_codes(classification)
    rows: list[dict[str, Any]] = []
    for field in REFERENCE_FIELDS:
        expected = reference[field]
        if expected in {"not_reported", "indeterminate"}:
            continue
        actual = observed[field]
        rows.append(
            {
                "domain": field,
                "reference": expected,
                observer: actual,
                "agreement": expected == actual,
            }
        )
    return rows
