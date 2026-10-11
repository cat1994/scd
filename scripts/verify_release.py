#!/usr/bin/env python3
"""Verify release file integrity and, optionally, complete archived tournaments."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true", help="Also validate all archived pair/round/metric records")
    args = parser.parse_args()
    expected = {}
    for line in (ROOT / "SHA256SUMS").read_text().splitlines():
        digest, name = line.split("  ", 1)
        path = (ROOT / name).resolve()
        if not path.is_relative_to(ROOT) or not path.is_file():
            raise SystemExit(f"Missing or unsafe release path: {name}")
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise SystemExit(f"Checksum mismatch: {name}")
        expected[name] = digest
    provenance = json.loads((ROOT / "data/provenance.json").read_text())
    runs = provenance["runs"]
    if len(runs) != 28 or sum(x["family"] == "main" for x in runs) != 7:
        raise SystemExit("Expected 7 main and 21 ablation archives")
    for run in runs:
        if expected.get(run["path"]) != run["sha256"]:
            raise SystemExit(f"Provenance mismatch: {run['path']}")
        if args.full:
            from hr_edl_data.tournament_records import validate_tournament_result
            valid, reason = validate_tournament_result(ROOT / run["path"], run["algorithm_pool"], run["iterations"])
            if not valid:
                raise SystemExit(f"{run['path']}: {reason}")
    print(f"Verified {len(expected)} release files and {len(runs)} unchanged raw archives" + ("; all tournament matrices complete" if args.full else ""))

if __name__ == "__main__":
    main()
