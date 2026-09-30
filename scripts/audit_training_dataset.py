"""python scripts/audit_training_dataset.py records.json --output audit.json"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cus_ai.dataset_audit import audit_records

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check infant and external-center independence before model fitting")
    parser.add_argument("records", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    records = json.loads(args.records.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        parser.error("records must be a JSON list of examination records")
    result = audit_records(records)
    output = json.dumps(result, indent=2)
    if args.output:
        args.output.write_text(output, encoding="utf-8")
    print(output)
    sys.exit(0 if result["valid"] else 1)
