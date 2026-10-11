#!/usr/bin/env python3
"""Run the seven required SCD main methods (group 11) with metadata."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from bin.run_SCD_ablation import execute_matrix, parse_args

if __name__ == "__main__":
    raise SystemExit(execute_matrix(parse_args(main_experiment=True)))
