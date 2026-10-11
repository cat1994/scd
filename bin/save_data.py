#!/usr/bin/env python3
"""Validate paper SSV records and save a reusable NumPy analysis dataset."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
from hr_edl_data.scd_analysis import load_raw_data


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=("main", "ablation"), required=True)
    parser.add_argument("--input-dir", type=Path, help="Default: data/raw/FAMILY")
    parser.add_argument("--output", "--o", type=Path, help="Default: outputs/data/FAMILY.npy")
    args = parser.parse_args(argv)
    source = args.input_dir or Path("data/raw") / args.family
    destination = args.output or Path("outputs/data") / f"{args.family}.npy"
    if destination.suffix != ".npy":
        destination = Path(str(destination) + ".npy")
    archive = Path(__file__).resolve().parent.parent / "data" / "raw"
    if archive.resolve() == destination.resolve() or archive.resolve() in destination.resolve().parents:
        parser.error("Derived data must not be written into the paper archive")
    if destination.exists():
        parser.error(f"Output exists; choose a new --output: {destination}")
    try:
        frame = load_raw_data(source)
        expected = "SCD_main" if args.family == "main" else "SCD_ablation"
        if set(frame.experiment_family) != {expected}:
            raise ValueError("Save SCD main and ablation datasets separately")
        destination.parent.mkdir(parents=True, exist_ok=True)
        # The exclusive file mode protects an existing analysis dataset.
        with destination.open("xb") as handle:
            np.save(handle, frame.to_dict("records"))
        print(f"Saved {len(frame):,} validated records to {destination}")
    except (OSError, ValueError, KeyError) as error:
        print(str(error), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
