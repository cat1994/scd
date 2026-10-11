"""Read tournament tuples and optional reproducibility sidecars.

The legacy ``osr`` name is retained for compatibility. It denotes the maximum
maintained local solver regret, not an oracle for the sparse message protocol.
"""

import json
from pathlib import Path
import re

import numpy as np

from hr_edl_data.experiment_parameters import ExperimentParameters


def metadata_path(path):
    return Path(str(path) + ".meta.json")


def read_metadata(path):
    sidecar = metadata_path(path)
    return json.loads(sidecar.read_text()) if sidecar.exists() else {}


def evaluation_set(row):
    sets = {
        "CFR": "local_EX_empty_history",
        "A-EFR_IN": "local_IN_on_identity_action_sequence",
        "CSPS-EFR": "local_EX_causal_partial_histories_and_IN_identity_history",
        "BEHAV-EFR": "local_IN_all_behavioral_recommendation_histories",
        "BR": "unavailable",
    }
    if row in sets:
        return sets[row]
    if re.fullmatch(r"\d+-SCD", row):
        return "local_EX_length_k_if_present_and_IN_length_min(k-1,d_I)"
    if re.fullmatch(r"\d+-SCD_UNCOMPRESSED", row):
        return "local_EX_all_lengths_le_k_and_IN_all_lengths_lt_k"
    return "maintained_local_EX_IN_experts"


def tournament_data_entries(file_name):
    """Accept two through five component tuples without inventing metrics.

    New fixed outputs have T updates for T records. Historical three component
    outputs keep their original osr normalization; their actual update count is
    unknown, and the new raw metric remains NaN.
    """
    source = Path(file_name).resolve()
    source_label = portable_source_path(source)
    params = ExperimentParameters(str(source))
    metadata = read_metadata(source)
    config = metadata.get("config", {})
    for field, expected in (("game", params.game_tag), ("mode", params.mode),
                            ("scd_budget", params.scd_budget)):
        if field in config and config[field] != expected:
            raise ValueError(f"{source}: metadata/filename mismatch for {field}")
    cfr_plus = config.get("cfr_plus")
    if cfr_plus is None and "no_cfr_plus" in source.parts:
        cfr_plus = False
    family = config.get("experiment_family", "SCD_main" if params.alg_group_str == "scd_main" else
                        "SCD_ablation" if params.alg_group_str == "scd_ablation" else params.alg_group_str)
    columns = None
    t = None
    seen = set()
    explicit_updates = False
    with source.open() as file:
        for line_number, line in enumerate(file, 1):
            line = line.strip()
            if not line:
                continue
            if line.startswith("#"):
                explicit_updates |= "each row learner has t+1 updates" in line
                if line.startswith("# cfr_plus = "):
                    actual = line.split("=", 1)[1].strip() == "true"
                    if cfr_plus is not None and actual != cfr_plus:
                        raise ValueError(f"{source}: CFR+ header/metadata mismatch")
                    cfr_plus = actual
                continue
            if columns is None:
                fields = re.split(r"\s{2,}", line)
                if fields[0] != "learner/env" or len(fields) < 2:
                    raise ValueError(f"{source}:{line_number}: missing learner/env header")
                columns = fields[1:]
                if len(set(columns)) != len(columns):
                    raise ValueError(f"{source}: duplicate column labels")
                continue
            if line.startswith("t = "):
                t = int(line[4:])
                if t < 0:
                    raise ValueError(f"{source}: negative t")
                continue
            if t is None:
                raise ValueError(f"{source}:{line_number}: row before t")
            fields = re.split(r"\s{2,}", line)
            if len(fields) != len(columns) + 1:
                raise ValueError(f"{source}:{line_number}: incomplete row {fields[0]}")
            row = fields[0]
            for col, token in zip(columns, fields[1:]):
                key = (t, row, col)
                if key in seen:
                    raise ValueError(f"{source}: duplicate record {key}")
                seen.add(key)
                if not (token.startswith("(") and token.endswith(")")):
                    raise ValueError(f"{source}:{line_number}: invalid tuple {token}")
                values = [float(x.strip()) for x in token[1:-1].split(",")]
                if len(values) not in (2, 3, 4, 5):
                    raise ValueError(f"{source}:{line_number}: invalid tuple {token}")
                solver = values[2] if len(values) >= 3 and row != "BR" else np.nan
                raw = values[3] if len(values) >= 4 and row != "BR" else np.nan
                known_updates = explicit_updates or config.get("round_semantics") == "pre_update_ev_then_update_every_round"
                updates = t + 1 if known_updates and row != "BR" else np.nan
                yield {
                    "game_tag": params.game_tag, "sampler": config.get("sampler", "null"),
                    "seed": None, "mode": params.mode, "row_alg": row, "col_alg": col,
                    "value": values[0], "t": t, "milliseconds": values[1],
                    "diagnostic_milliseconds": values[4] if len(values) == 5 else np.nan,
                    "max_full_phi_regret_fus": solver, "osr": solver / (t + 1),
                    "regret_updates": updates,
                    "retained_local_regret_numerator": raw,
                    "retained_local_regret": raw / updates if np.isfinite(updates) else np.nan,
                    "full_local_regret_bound": np.nan, "full_local_regret_exact": np.nan,
                    "scd_budget": config.get("scd_budget", params.scd_budget),
                    "experiment_family": family, "opponent_pool": tuple(columns),
                    "analysis_opponent_pool": tuple(config.get("analysis_opponent_pool", columns)),
                    "source_file": source_label,
                    "run_id": metadata.get("run_id", source_label),
                    "run_config": config, "run_status": metadata.get("status", "legacy"),
                    "cfr_plus": cfr_plus,
                    "solver_regret_truncated": cfr_plus,
                    "metric_evaluation_set": evaluation_set(row),
                    "record_format": f"tournament_tuple_{len(values)}",
                    "round_semantics": config.get("round_semantics", "pre_update_ev_then_update_every_round" if known_updates else "legacy_unknown"),
                    "source_config_provenance": "sidecar" if metadata else "filename_and_header",
                    "osr_definition": "local_solver_max" if known_updates else "legacy_local_solver_max_t_plus_1",
                }
    if columns is None or not seen:
        raise ValueError(f"{source}: empty tournament")


def validate_tournament_result(path, algorithms, iterations, *, entries=None):
    """Validate exact labels, every pair/round, and finite learner metrics."""
    algorithms = tuple(algorithms)
    try:
        entries = list(tournament_data_entries(path)) if entries is None else entries
        expected = {(t, row, col) for t in range(iterations)
                    for row in (*algorithms, "BR") for col in algorithms}
        actual = {(x["t"], x["row_alg"], x["col_alg"]) for x in entries}
        if actual != expected:
            missing, extra = sorted(expected - actual), sorted(actual - expected)
            return False, f"incomplete matrix: missing={missing[:8]}, extra={extra[:8]}"
        if any(x["opponent_pool"] != algorithms for x in entries):
            return False, "unexpected algorithm header"
        for x in entries:
            declared_format = x["run_config"].get("record_format")
            if declared_format and x["record_format"] != declared_format:
                return False, "tuple format does not match run metadata"
            if not np.isfinite(x["value"]) or not np.isfinite(x["milliseconds"]) or x["milliseconds"] < 0:
                return False, f"invalid value/time at {(x['t'], x['row_alg'], x['col_alg'])}"
            if x["row_alg"] != "BR" and (not np.isfinite(x["osr"]) or
                                          x["regret_updates"] != x["t"] + 1 or
                                          not np.isfinite(x["retained_local_regret"])):
                return False, "missing local metric or explicit update count"
            if x["row_alg"] != "BR" and x["record_format"] == "tournament_tuple_5" and (
                not np.isfinite(x["diagnostic_milliseconds"]) or x["diagnostic_milliseconds"] < 0
            ):
                return False, "invalid diagnostic time"
        return True, "complete"
    except (OSError, ValueError, KeyError, TypeError) as error:
        return False, str(error)


def validate_ablation_result(path, budget, iterations):
    """Validate all four learner pairs and both BR pairs at every round."""
    return validate_tournament_result(path, (f"{budget}-SCD", f"{budget}-SCD_UNCOMPRESSED"), iterations)


def portable_source_path(path):
    """Avoid embedding the local machine path in saved analysis data."""
    root = Path(__file__).resolve().parent.parent
    try:
        return Path(path).resolve().relative_to(root).as_posix()
    except ValueError:
        return Path(path).name
