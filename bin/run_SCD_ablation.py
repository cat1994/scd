#!/usr/bin/env python3
"""Run independent fixed 2x2 SCD ablations for k=1,2,4.

Commands and scheduling are shared with run_all_qbce. Existing results are
never overwritten. --skip-existing requires complete records and matching
configuration, code fingerprints, and executable bytes.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from bin import run_all_qbce as launcher
from hr_edl_data.experiment_parameters import SCD_MAIN_ALGORITHMS
from hr_edl_data.tournament_records import (
    metadata_path, read_metadata, validate_tournament_result,
)

DEFAULT_GAMES = ('kuhn_3p', 'leduc', 'random_goofspiel', 'goofspiel_3p', 'goofspiel_ascending_3p', 'sheriff', 'battleship')


def parse_args(argv=None, *, main_experiment=False):
    parser = argparse.ArgumentParser(description=(
        "Run the seven-algorithm fixed SCD main experiment (group 11)."
        if main_experiment else __doc__
    ))
    parser.add_argument("--games", nargs="+", choices=sorted(launcher.GAME_MAP), default=list(DEFAULT_GAMES))
    parser.add_argument("--query-budgets", "--k", dest="query_budgets", nargs="+", type=int, choices=(1, 2, 4), default=[4] if main_experiment else [1, 2, 4])
    parser.add_argument("--iterations", type=launcher.positive_int, default=1000)
    parser.add_argument("--exe-dir", type=Path, default=Path("hr_edl/build.optimized/bin"))
    parser.add_argument("--data-dir", type=Path, default=Path("outputs/raw/main" if main_experiment else "outputs/raw/ablation"))
    parser.add_argument("--cfr-plus", type=launcher.parse_bool, default=False, metavar="{true,false}")
    parser.add_argument("--threads-per-run", type=launcher.positive_int, default=1)
    parser.add_argument("--max-parallel-jobs", type=launcher.positive_int, default=1)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if len(set(args.games)) != len(args.games) or len(set(args.query_budgets)) != len(args.query_budgets):
        parser.error("games and budgets must not contain duplicates")
    if main_experiment and args.query_budgets != [4]:
        parser.error("the seven-algorithm main pool requires k=4")
    args.algorithms, args.run_modes = ["qbce_scd_main" if main_experiment else "qbce_test"], ["fixed"]
    return args


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def repository_snapshot():
    """Content fingerprints work before the first commit and expose no Git identity."""
    root = launcher.PROJECT_ROOT
    files = sorted({p for directory in ("bin", "hr_edl_data", "hr_edl/src/bin", "hr_edl/src/hr_edl")
                    for p in (root / directory).rglob("*") if p.suffix in (".py", ".cc", ".h")})
    hashes = {p.relative_to(root).as_posix(): sha256(p) for p in files}
    digest = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    return {"source_sha256": digest, "source_hashes": hashes}


def portable_path(path):
    try:
        return Path(path).resolve().relative_to(launcher.PROJECT_ROOT).as_posix()
    except ValueError:
        return Path(path).name


def portable_command(command):
    return [portable_path(command[0])] + [
        "--file_dir=" + portable_path(token.split("=", 1)[1]) if token.startswith("--file_dir=") else token
        for token in command[1:]]


def task_config(experiment, args, snapshot):
    executable = Path(experiment.command[0])
    stat = executable.stat()
    algorithms = expected_algorithms(experiment)
    return {
        "experiment_family": "SCD_ablation" if experiment.algorithm == "qbce_test" else "SCD_main",
        "algorithm_group": experiment.algorithm, "alg_group": launcher.ALG_GROUP_MAP[experiment.algorithm],
        "algorithms": algorithms, "opponent_pool": algorithms,
        "game": experiment.game, "game_definition": launcher.GAME_MAP[experiment.game],
        "query_budget": experiment.query_budget, "iterations": args.iterations,
        "mode": "fixed", "sampler": "null", "cfr_plus": args.cfr_plus,
        "threads_per_run": args.threads_per_run, "max_parallel_jobs": args.max_parallel_jobs,
        "epsilon": 0.0,
        "sequential": False if experiment.algorithm == "qbce_test" else None,
        "causal": False if experiment.algorithm == "qbce_test" else None,
        "algorithm_settings": {name: algorithm_settings(name, args.cfr_plus) for name in algorithms},
        "analysis_opponent_pool": algorithms if experiment.algorithm == "qbce_test" else
            list(SCD_MAIN_ALGORITHMS),
        "round_semantics": "pre_update_ev_then_update_every_round",
        "runtime_scope": "row value tree evaluation and policy update; excludes column self-play, metric reads, output and audit",
        "diagnostic_scope": "PhiRegrets and untruncated retained local regret reads, timed separately",
        "record_format": "tournament_tuple_5",
        "regret_definition": "maximum positive maintained local EX/IN cumulative regret, maximized over seats and information sets",
        "raw_regret_accumulator": "stage_original, initialized at construction and never reset in fixed tournament",
        "code_head": snapshot["source_sha256"], "source_hashes": snapshot["source_hashes"],
        "executable": {"path": portable_path(executable), "sha256": sha256(executable),
                       "size": stat.st_size},
        "command": portable_command(experiment.command),
    }


def algorithm_settings(name, cfr_plus):
    settings = {"cfr_plus": cfr_plus, "epsilon": 0.0}
    if match := re.fullmatch(r"(\d+)-(QBC?)-EFR(.*)", name):
        budget, kind, suffix = match.groups()
        settings.update(query_budget=int(budget), sequential="SEQ" in suffix,
                        causal="CS" in suffix,
                        predecessors="QueryPredecessors_" if kind == "QBC" else "QueryPredecessors")
    else:
        settings["predecessors"] = {"CFR": "ImmediateExternalSequencePredecessors",
                                    "CFR_IN": "ImmediateInternalSequencePredecessors",
                                    "A-EFR_IN": "InformedActionSequencePredecessors",
                                    "CSPS-EFR": "CausalPartialSequencePredecessors",
                                    "BEHAV-EFR": "BehavioralPredecessors"}[name]
    return settings


def expected_algorithms(experiment):
    if experiment.algorithm == "qbce_test":
        return [f"{experiment.query_budget}-QB-EFR", f"{experiment.query_budget}-QBC-EFR"]
    if experiment.algorithm == "qbce_scd_main":
        return list(SCD_MAIN_ALGORITHMS)
    raise ValueError(f"Unsupported SCD algorithm group: {experiment.algorithm}")


def write_json(path, data):
    temporary = Path(str(path) + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def execute_matrix(args):
    data_dir = launcher.resolve_project_path(args.data_dir).resolve()
    archive_dir = (launcher.PROJECT_ROOT / "data" / "raw").resolve()
    if data_dir == archive_dir or archive_dir in data_dir.parents:
        raise ValueError("Archived paper data is read-only; choose a directory under outputs/")
    experiments = launcher.build_experiments(args)
    print(f"SCD matrix: {len(experiments)} tasks; group={args.algorithms[0]}, mode=fixed")
    # A dry run performs no executable validation, metadata writes or experiments.
    if args.dry_run:
        for experiment in experiments:
            print(f"[DRY-RUN] {experiment.label}")
            print(f"          {launcher.command_text(experiment)}")
        return 0
    launcher.validate_executables(experiments)
    snapshot = repository_snapshot()
    pending, configs, refused, skipped = [], {}, [], []
    for experiment in experiments:
        config = task_config(experiment, args, snapshot)
        configs[experiment.output_path] = config
        if any(p.exists() for p in (experiment.output_path, experiment.log_path, metadata_path(experiment.output_path))):
            try:
                metadata = read_metadata(experiment.output_path)
            except (ValueError, OSError) as error:
                metadata = {"metadata_error": str(error)}
            valid, reason = validate_tournament_result(experiment.output_path, expected_algorithms(experiment), args.iterations)
            compatible = metadata.get("config") == config
            if args.skip_existing and valid and compatible and metadata.get("status") == "complete" and metadata.get("output_sha256") == sha256(experiment.output_path):
                print(f"[SKIP] {experiment.label}")
                skipped.append(portable_path(experiment.output_path))
                continue
            if valid and compatible and metadata.get("status") == "complete" and metadata.get("output_sha256") == sha256(experiment.output_path):
                reason = "complete result exists; --skip-existing can reuse it"
            elif valid:
                reason = "configuration, metadata status or output fingerprint mismatch"
            print(f"[REFUSE] {experiment.label}: existing files preserved ({reason}); use a new --data-dir")
            refused.append({"task": experiment.label, "reason": reason})
        else:
            pending.append(experiment)
    if refused:
        return 1
    if not pending:
        return 0
    output_dir = experiments[0].output_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    report_id = str(uuid.uuid4())
    report = {"report_id": report_id, "started_at_utc": datetime.now(timezone.utc).isoformat(),
              "repository": snapshot, "skipped": skipped, "refused": refused, "tasks": []}
    report_path = output_dir / f"run_report.{report_id}.json"
    metadata_by_path = {}
    for experiment in pending:
        metadata = {"schema_version": 1, "run_id": str(uuid.uuid4()), "status": "pending",
                    "config": configs[experiment.output_path], "repository": snapshot,
                    "started_at_utc": report["started_at_utc"]}
        metadata_by_path[experiment.output_path] = metadata
        with metadata_path(experiment.output_path).open("x") as handle:
            json.dump(metadata, handle, indent=2)
            handle.write("\n")

    def checker(experiment):
        return validate_tournament_result(experiment.output_path, expected_algorithms(experiment), args.iterations)

    def finish(experiment, return_code, reason):
        metadata = metadata_by_path[experiment.output_path]
        if reason:
            reason = reason.replace(str(launcher.PROJECT_ROOT), ".")
            reason = reason.replace(str(experiment.output_path), portable_path(experiment.output_path))
            reason = reason.replace(experiment.command[0], portable_path(experiment.command[0]))
        metadata.update(status="failed" if reason else "complete", return_code=return_code,
                        failure_reason=reason, finished_at_utc=datetime.now(timezone.utc).isoformat())
        if experiment.output_path.exists():
            metadata["output_sha256"] = sha256(experiment.output_path)
        write_json(metadata_path(experiment.output_path), metadata)
        report["tasks"].append({"task": experiment.label, "status": metadata["status"],
                                "return_code": return_code, "reason": reason,
                                "metadata": portable_path(metadata_path(experiment.output_path))})
        write_json(report_path, report)

    write_json(report_path, report)
    try:
        failures = launcher.run_experiments(pending, args.max_parallel_jobs, checker, finish)
    except KeyboardInterrupt:
        for experiment in pending:
            if metadata_by_path[experiment.output_path]["status"] == "pending":
                finish(experiment, 130, "interrupted")
        return 130
    print(f"Run report: {report_path}")
    return 1 if failures or refused else 0


def main(argv=None):
    return execute_matrix(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
