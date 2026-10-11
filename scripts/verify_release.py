#!/usr/bin/env python3
"""Verify release file integrity and, optionally, complete archived tournaments."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true", help="Also validate archived metadata, label migration and complete tournaments")
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
            from hr_edl_data.tournament_records import read_metadata, validate_tournament_result
            path = ROOT / run["path"]
            metadata = read_metadata(path)
            if metadata.get("status") != "complete" or metadata.get("output_sha256") != run["sha256"]:
                raise SystemExit(f"Archive metadata/fingerprint mismatch: {run['path']}")
            if metadata.get("original_output_sha256") != run["original_sha256"] or path.stat().st_size != run["bytes"]:
                raise SystemExit(f"Archive original fingerprint/size mismatch: {run['path']}")
            # Restore historical labels only for the migration integrity check.
            # The original checksum proves tuple text, rounds and timing are preserved.
            original = re.sub(rb"(\d+)-SCD(_UNCOMPRESSED)?",
                              lambda match: match[1] + (b"-QBC-EFR" if match[2] else b"-QB-EFR"),
                              path.read_bytes())
            if hashlib.sha256(original).hexdigest() != run["original_sha256"]:
                raise SystemExit(f"Archived numerical data changed during label migration: {run['path']}")
            valid, reason = validate_tournament_result(ROOT / run["path"], run["algorithm_pool"], run["iterations"])
            if not valid:
                raise SystemExit(f"{run['path']}: {reason}")
    print(f"Verified {len(expected)} release files and {len(runs)} archived runs" + ("; all tournament matrices complete" if args.full else ""))

if __name__ == "__main__":
    main()
