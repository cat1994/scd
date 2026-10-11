"""Scheduling helpers for the two fixed SCD experiment matrices."""
from __future__ import annotations
import argparse
import os
import shlex
import signal
import subprocess
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Callable, Sequence
from hr_edl_data.experiment_parameters import ALG_GROUP_MAP, GAME_MAP
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCD_BUDGET_ALGORITHMS = frozenset({"scd_ablation"})
EXECUTABLE_BY_MODE = {"fixed": "run_fixed_ltbr"}

@dataclass(frozen=True)
class Experiment:
    """One SCD process and the files associated with it."""

    game: str
    algorithm: str
    scd_budget: int
    run_mode: str
    output_path: Path
    log_path: Path
    command: tuple[str, ...]

    @property
    def label(self) -> str:
        return (
            f"algorithm={self.algorithm} game={self.game} "
            f"mode={self.run_mode} k={self.scd_budget}"
        )


@dataclass
class RunningExperiment:
    """Resources owned by a currently running experiment."""

    experiment: Experiment
    process: subprocess.Popen[bytes]
    log_file: BinaryIO


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def parse_bool(value: str) -> bool:
    normalized = value.lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise argparse.ArgumentTypeError("expected true or false")


def resolve_project_path(path: Path) -> Path:
    """Resolve a CLI path consistently without changing the process directory."""
    return path if path.is_absolute() else PROJECT_ROOT / path


def bool_flag(value: bool) -> str:
    return "true" if value else "false"


def build_experiments(args: argparse.Namespace) -> list[Experiment]:
    """Expand CLI selections into concrete commands and output paths."""
    executable_dir = resolve_project_path(args.exe_dir)
    data_dir = resolve_project_path(args.data_dir)
    experiments = []

    for game in args.games:
        iterations = args.iterations
        for algorithm in args.algorithms:
            scd_budgets = (
                args.scd_budgets
                if algorithm in SCD_BUDGET_ALGORITHMS
                else args.scd_budgets[:1]
            )
            for scd_budget in scd_budgets:
                for run_mode in args.run_modes:
                    stem = f"{algorithm}.{game}.null.{run_mode}.gen.{scd_budget}"
                    output_path = data_dir / f"{stem}.ssv"
                    log_path = data_dir / f"{stem}.log"
                    executable = executable_dir / EXECUTABLE_BY_MODE[run_mode]
                    command = (
                        str(executable),
                        f"--game={GAME_MAP[game]}",
                        "--sampler=null",
                        f"--t={iterations}",
                        f"--alg_group={ALG_GROUP_MAP[algorithm]}",
                        "--rt_cfr_weight=0.0",
                        "--rt_cfr_T=1",
                        "--epsilon=0.0",
                        f"--file_dir={output_path}",
                        f"--scd_budget={scd_budget}",
                        f"--cfr_plus={bool_flag(args.cfr_plus)}",
                        f"--threads={args.threads_per_run}",
                    )
                    experiments.append(
                        Experiment(
                            game=game,
                            algorithm=algorithm,
                            scd_budget=scd_budget,
                            run_mode=run_mode,
                            output_path=output_path,
                            log_path=log_path,
                            command=command,
                        )
                    )
    return experiments


def command_text(experiment: Experiment) -> str:
    return (
        f"{shlex.join(experiment.command)} > "
        f"{shlex.quote(str(experiment.log_path))} 2>&1"
    )


def validate_executables(experiments: Sequence[Experiment]) -> None:
    missing = sorted(
        {
            experiment.command[0]
            for experiment in experiments
            if not os.access(experiment.command[0], os.X_OK)
        }
    )
    if missing:
        formatted = "\n  ".join(missing)
        raise SystemExit(
            f"SCD executable(s) not found or not executable:\n  {formatted}"
        )


def is_complete_result(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


def terminate_all(running: Sequence[RunningExperiment]) -> None:
    """Stop all process groups and close their log files after interruption."""
    for item in running:
        if item.process.poll() is None:
            try:
                os.killpg(item.process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass

    deadline = time.monotonic() + 5
    for item in running:
        timeout = max(0, deadline - time.monotonic())
        try:
            item.process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(item.process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        finally:
            item.log_file.close()


def run_experiments(
    experiments: Sequence[Experiment], max_parallel_jobs: int,
    result_checker: Callable[[Experiment], tuple[bool, str]] | None = None,
    on_finish: Callable[[Experiment, int, str | None], None] | None = None,
) -> list[tuple[Experiment, int, str]]:
    """Run the matrix and return each failed experiment with its reason."""
    pending = deque(experiments)
    running: list[RunningExperiment] = []
    failures = []

    try:
        while pending or running:
            while pending and len(running) < max_parallel_jobs:
                experiment = pending.popleft()
                log_file = experiment.log_path.open("xb")
                print(f"[RUN] {experiment.label}", flush=True)
                print(f"      {command_text(experiment)}", flush=True)
                try:
                    process = subprocess.Popen(
                        experiment.command,
                        stdout=log_file,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                    )
                except OSError as error:
                    log_file.close()
                    failures.append((experiment, -1, str(error)))
                    if on_finish is not None:
                        on_finish(experiment, -1, str(error))
                    print(f"[FAIL] {experiment.label}: {error}", flush=True)
                    continue
                running.append(RunningExperiment(experiment, process, log_file))

            still_running = []
            for item in running:
                return_code = item.process.poll()
                if return_code is None:
                    still_running.append(item)
                    continue

                item.log_file.close()
                experiment = item.experiment
                valid = False
                reason = None
                if return_code != 0:
                    reason = f"exit status {return_code}"
                    failures.append((experiment, return_code, reason))
                    print(f"[FAIL] {experiment.label}: {reason}", flush=True)
                else:
                    valid, reason = (
                        result_checker(experiment) if result_checker is not None
                        else (is_complete_result(experiment.output_path),
                              "result file is missing or empty")
                    )
                if return_code == 0 and not valid:
                    failures.append((experiment, return_code, reason))
                    print(f"[FAIL] {experiment.label}: {reason}", flush=True)
                elif return_code == 0:
                    print(f"[DONE] {experiment.label}", flush=True)
                if on_finish is not None:
                    on_finish(experiment, return_code,
                              reason if return_code != 0 or not valid else None)
            running = still_running

            if running:
                time.sleep(0.5)
    except KeyboardInterrupt:
        print(
            "\n[STOP] Interrupt received; terminating running experiments.",
            flush=True,
        )
        terminate_all(running)
        raise

    return failures
