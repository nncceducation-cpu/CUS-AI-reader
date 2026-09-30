"""Audit proposed training partitions without reading or uploading pixels."""
from collections import Counter, defaultdict


def audit_records(records: list[dict]) -> dict:
    errors, warnings = [], []
    infants, hashes, centers = defaultdict(set), defaultdict(set), defaultdict(set)
    seen_studies = set()
    split_counts = Counter()
    label_counts = defaultdict(Counter)
    for index, record in enumerate(records):
        required = ("infant_code", "study_code", "center", "split")
        missing = [key for key in required if not record.get(key)]
        if missing:
            errors.append(f"Record {index}: missing {', '.join(missing)}")
            continue
        split = record["split"]
        if split not in {"train", "calibration", "internal_test", "external_test"}:
            errors.append(f"Record {index}: unrecognized split {split}")
        study = record["study_code"]
        if study in seen_studies:
            errors.append(f"Duplicate examination: {study}")
        seen_studies.add(study)
        infants[record["infant_code"]].add(split)
        centers[record["center"]].add(split)
        split_counts[split] += 1
        if record.get("adjudicated") is not True:
            errors.append(f"{study}: independent expert adjudication is not confirmed")
        digests = record.get("source_sha256", [])
        if not digests:
            errors.append(f"{study}: source hashes are required to detect duplicate-media leakage")
        for digest in digests:
            if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdefABCDEF" for c in digest):
                errors.append(f"{study}: invalid source SHA256")
                continue
            hashes[digest.lower()].add(split)
        for domain, value in record.get("labels", {}).items():
            if value not in (None, "", "unknown", "not_assessed", "indeterminate"):
                label_counts[domain][str(value)] += 1
    for infant, splits in infants.items():
        if len(splits) > 1:
            errors.append(f"Infant leakage: {infant} appears in {sorted(splits)}")
    for digest, splits in hashes.items():
        if len(splits) > 1:
            errors.append(f"Duplicate-media leakage: {digest} appears in {sorted(splits)}")
    for center, splits in centers.items():
        if "external_test" in splits and len(splits) > 1:
            errors.append(f"External-center leakage: {center} appears in development and external testing")
    if not records:
        errors.append("No examination records supplied")
    for split in ("train", "calibration", "internal_test", "external_test"):
        if not split_counts[split]:
            warnings.append(f"No examinations in {split}")
    warnings.append("Passing this audit does not establish sample-size adequacy, label correctness or model accuracy")
    return {"valid": not errors, "examinations": len(records), "infants": len(infants),
            "centers": len(centers), "split_counts": dict(split_counts),
            "label_counts": {key: dict(value) for key, value in label_counts.items()},
            "errors": errors, "warnings": warnings}
