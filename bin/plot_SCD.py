#!/usr/bin/env python3
"""Render SCD main and/or ablation figures after checking complete matrices."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import matplotlib
matplotlib.use("Agg")
from hr_edl_data.scd_analysis import (GAMES, load_data, plot_ablation, plot_main,
                                     assert_main_ablation_compatible)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ablation-data", type=Path)
    parser.add_argument("--main-data", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/figures"))
    parser.add_argument("--games", nargs="+", default=list(GAMES))
    parser.add_argument("--metric", choices=("osr", "retained_local_regret"), default="osr")
    parser.add_argument("--smoke", action="store_true", help="Label partial validation figures explicitly")
    args = parser.parse_args(argv)
    if not args.ablation_data and not args.main_data:
        parser.error("provide --ablation-data and/or --main-data")
    if tuple(args.games) != GAMES and not args.smoke:
        parser.error("a subset of the default plotted games requires --smoke")
    try:
        ablation = load_data(args.ablation_data) if args.ablation_data else None
        main_frame = load_data(args.main_data) if args.main_data else None
        if ablation is not None and main_frame is not None:
            assert_main_ablation_compatible(main_frame, ablation)
        if ablation is not None:
            for path in plot_ablation(ablation, args.output_dir, games=args.games, smoke=args.smoke): print(path)
        if main_frame is not None:
            for path in plot_main(main_frame, args.output_dir, games=args.games, metric=args.metric, smoke=args.smoke): print(path)
    except ValueError as error:
        print(str(error), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
