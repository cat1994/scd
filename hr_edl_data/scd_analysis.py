"""Complete opponent pools and equal-opponent learning curves for SCD.

Runtime curves pair average cumulative learner time with average payoff at the
same round. They do not stop every matchup at a common wall-clock budget.
"""

from pathlib import Path
import hashlib
import json
import re
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from hr_edl_data.plot import set_plt_params
from hr_edl_data.experiment_parameters import SCD_MAIN_ALGORITHMS
from hr_edl_data.tournament import game_label
from hr_edl_data.tournament_records import read_metadata, tournament_data_entries, validate_tournament_result

# Games shown in figures; the experiment launchers keep their full game lists.
GAMES = ("kuhn_3p", "leduc", "random_goofspiel", "goofspiel_3p",
         "goofspiel_ascending_3p", "sheriff", "battleship")
MAIN_POOL = SCD_MAIN_ALGORITHMS
LABELS = dict(zip(MAIN_POOL, ("SCD_0", "SCD_1", "SCD_2", "SCD_4", "ACT_IN", "CSPS", "BHV")))
COLORS = dict(zip(MAIN_POOL, ("#7f7f7f", "#d62728", "#ff7f0e", "#1f77b4", "#9467bd", "#2ca02c", "#111111")))
for _k in (1, 2, 4):
    LABELS[f"{_k}-QBC-EFR"] = f"SCD_<={_k}"
    COLORS[f"{_k}-QBC-EFR"] = COLORS[f"{_k}-QB-EFR"]

PLOT_STYLE = {
    "font.family": "serif", "font.size": 29,
    "mathtext.fontset": "cm",
    "axes.labelsize": 24, "axes.titlesize": 29,
    "axes.edgecolor": "#545859", "axes.labelcolor": "black",
    "axes.linewidth": .75, "text.color": "black",
    "axes.titlecolor": "black",
    "xtick.color": "black", "ytick.color": "black",
    "xtick.labelcolor": "black", "ytick.labelcolor": "black",
    "legend.labelcolor": "black",
    "xtick.labelsize": 24, "ytick.labelsize": 24,
    "figure.facecolor": "white", "axes.facecolor": "white",
    "pdf.fonttype": 42, "ps.fonttype": 42,
}


def _game_title(game):
    title = game_label(game).replace("$(", " $(")
    if not plt.rcParams["text.usetex"]:
        title = re.sub(r"\\num\{([^}]+)\}", r"\1", title)
    return title


def _style_axis(ax, ylabel, *, runtime=False, show_xlabel=True):
    ax.set_xlabel(("avg. cumulative time (s)" if runtime else "round") if show_xlabel else "")
    if plt.rcParams["text.usetex"]:
        ylabel = ylabel.replace("%", r"\%")
    ax.set_ylabel(ylabel, labelpad=9)
    ax.set_axisbelow(True)
    ax.grid(color="#d9dde1", linewidth=.6, alpha=.65)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(direction="out", length=3, width=.7)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5, integer=not runtime))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.margins(x=.025, y=.07)


def _row_legend(ax, *, loc="upper right"):
    ax.legend(loc=loc, ncol=2, fontsize=20,
              frameon=False, handlelength=1.9,
              columnspacing=.9, labelspacing=.35,
              handletextpad=.45, borderaxespad=.6)


def ablation_pool(k):
    return (f"{k}-QB-EFR", f"{k}-QBC-EFR")


def load_data(path):
    path = Path(path)
    if path.is_dir():
        return load_raw_data(path)
    if not path.is_file():
        raise ValueError(f"Data file missing: {path}; generate/save results first")
    frame = pd.DataFrame(list(np.load(path, allow_pickle=True)))
    if frame.empty:
        raise ValueError(f"Empty data file: {path}")
    return frame


def load_raw_data(directory, *, require_metadata=True):
    """Read SSV files, verify hashes and complete matrices, and preserve raw values."""
    files = sorted(Path(directory).rglob("*.ssv"))
    if not files:
        raise ValueError(f"No SCD result files found: {directory}")
    records, run_ids = [], set()
    for path in files:
        metadata = read_metadata(path)
        if require_metadata and not metadata:
            raise ValueError(f"Missing run metadata: {path}")
        if metadata:
            if metadata.get("status") != "complete":
                raise ValueError(f"Incomplete run metadata: {path}")
            if hashlib.sha256(path.read_bytes()).hexdigest() != metadata.get("output_sha256"):
                raise ValueError(f"Output fingerprint mismatch: {path}")
        entries = list(tournament_data_entries(path))
        if metadata:
            config = metadata["config"]
            valid, reason = validate_tournament_result(path, config["algorithms"], config["iterations"], entries=entries)
            if not valid:
                raise ValueError(f"Invalid tournament {path}: {reason}")
        run_id = entries[0]["run_id"]
        if run_id in run_ids:
            raise ValueError(f"Duplicate run_id across files: {run_id}")
        run_ids.add(run_id)
        records.extend(entries)
    return pd.DataFrame(records)


def _configurations(frame):
    # Parsing a file shares one immutable config dict across all of its records.
    # Cache by object identity instead of repeatedly hashing per-round metadata.
    seen = set()
    for config, cfr_plus in zip(frame.get("run_config", pd.Series([{}] * len(frame))),
                                frame.get("cfr_plus", pd.Series([None] * len(frame)))):
        config = config or {}
        key = (id(config), cfr_plus)
        if key not in seen:
            seen.add(key)
            yield config, cfr_plus


def _configuration_signature(config, cfr_plus):
    # All repeats in a runtime curve must use the same execution configuration.
    keys = ("cfr_plus", "threads_per_run", "max_parallel_jobs", "sampler", "epsilon",
            "round_semantics", "code_head", "iterations")
    return tuple(config.get(key, cfr_plus if key == "cfr_plus" else None) for key in keys) + (
        config.get("executable", {}).get("sha256"),
        tuple(sorted(config.get("source_hashes", {}).items())),
    )


def curve_data(frame, game, row_pool, opponent_pool, *, budget=None,
               metrics=(), cumulative=True):
    """Check each repeat's complete matrix, average columns, then repeats.

    Rows from BR or outside the explicit pools are excluded before checking.
    A missing opponent, duplicated round, unequal trajectory length, or mixed
    execution configuration raises an error instead of reducing a denominator.
    """
    required = {"game_tag", "mode", "row_alg", "col_alg", "t", "value", "milliseconds"}
    if missing := required - set(frame):
        raise ValueError(f"Missing fields: {sorted(missing)}")
    df = frame[(frame.game_tag == game) & (frame["mode"] == "fixed") &
               frame.row_alg.isin(row_pool) & frame.col_alg.isin(opponent_pool)].copy()
    if budget is not None:
        if "query_budget" not in df:
            raise ValueError(f"{game}/k={budget}: query_budget missing")
        df = df[df.query_budget == budget]
    context = f"{game}" + (f"/k={budget}" if budget is not None else "")
    if df.empty:
        raise ValueError(f"{context}: missing all requested matchups")
    if "run_status" in df and not df.run_status.isin(["complete", "legacy"]).all():
        raise ValueError(f"{context}: incomplete run metadata")
    if "run_id" not in df:
        df["run_id"] = df["source_file"] if "source_file" in df else "legacy"
    signatures = {_configuration_signature(config, plus) for config, plus in _configurations(df)}
    if len(signatures) != 1:
        raise ValueError(f"{context}: mixed runtime/algorithm configurations")
    if any(value is None for value in next(iter(signatures))[:3]):
        warnings.warn(f"{context}: legacy timing configuration is unknown", stacklevel=2)
    all_runs = []
    expected_rounds = None
    missing_pairs = []
    for run, subset in df.groupby("run_id", sort=False):
        pairs = {(row, col) for row in row_pool for col in opponent_pool}
        present = set(zip(subset.row_alg, subset.col_alg))
        if absent := pairs - present:
            missing_pairs.append(f"run={run}: {sorted(absent)}")
            continue
        if subset.duplicated(["row_alg", "col_alg", "t"]).any():
            raise ValueError(f"{context}/run={run}: duplicate matchup/round")
        trajectories = {}
        for (row, col), trajectory in subset.groupby(["row_alg", "col_alg"], sort=False):
            trajectory = trajectory.sort_values("t")
            rounds = trajectory.t.to_numpy()
            declared_iterations = (trajectory.iloc[0].get("run_config", {}) or {}).get("iterations")
            if declared_iterations is not None and len(rounds) != declared_iterations:
                raise ValueError(f"{context}/{row} vs {col}: expected {declared_iterations} rounds, found {len(rounds)}")
            if not np.array_equal(rounds, np.arange(len(rounds))):
                raise ValueError(f"{context}/{row} vs {col}: missing or invalid rounds {rounds[:8]}")
            if expected_rounds is None:
                expected_rounds = rounds
            elif not np.array_equal(rounds, expected_rounds):
                raise ValueError(f"{context}: inconsistent trajectory lengths")
            if not np.isfinite(trajectory.value).all():
                raise ValueError(f"{context}/{row} vs {col}: nonfinite payoff")
            ms = trajectory.milliseconds.to_numpy(dtype=float)
            if not np.isfinite(ms).all() or (ms < 0).any() or not (ms > 0).any():
                raise ValueError(f"{context}/{row} vs {col}: invalid or entirely zero runtime")
            values = trajectory.value.to_numpy(dtype=float)
            columns = {"value": np.cumsum(values) / np.arange(1, len(values) + 1) if cumulative else values,
                       "runtime_seconds": np.cumsum(ms) / 1000}
            for metric in metrics:
                if metric not in trajectory or not np.isfinite(trajectory[metric]).all():
                    raise ValueError(f"{context}/{row} vs {col}: unavailable metric {metric}")
                columns[metric] = trajectory[metric].to_numpy(dtype=float)
            trajectories[row, col] = pd.DataFrame(columns)
        # Average complete columns inside each repeat, then average repeats.
        all_runs.append({row: sum((trajectories[row, col] for col in opponent_pool)) / len(opponent_pool)
                         for row in row_pool})
    if missing_pairs:
        raise ValueError(f"{context}: missing pairs: {'; '.join(missing_pairs)}")
    return {row: sum(run[row] for run in all_runs) / len(all_runs) for row in row_pool}


def _check_comparison_config(frame, games, pools):
    selected = frame[(frame.game_tag.isin(games)) & (frame["mode"] == "fixed") &
                     frame.row_alg.isin(pools) & frame.col_alg.isin(pools)]
    signatures = {_configuration_signature(config, plus) for config, plus in _configurations(selected)}
    if len(signatures) > 1:
        raise ValueError("Compared games/budgets use different algorithm/runtime configurations")


def payoff_units(game, values):
    if game == "leduc" or game.startswith("kuhn_"):
        # Preserve the existing x1000 scale and the paper's mbb notation.
        return values * 1000, "avg. payoff (mbb)"
    if game.startswith("goofspiel") or game == "random_goofspiel":
        # Match the original notebook's percentage scale. For multiplayer
        # win_loss returns, this affine score is not a literal win probability.
        return (values + 1.0) / 2.0 * 100, "avg. win %"
    return values, "avg. payoff"


def _curves_for_all(frame, games, pools, metric=()):
    curves, errors = {}, []
    for game in games:
        for budget, pool in pools:
            try:
                curves[game, budget] = curve_data(frame, game, pool, pool, budget=budget, metrics=metric)
            except ValueError as error:
                errors.append(str(error))
    if errors:
        raise ValueError("Requested figure is incomplete:\n" + "\n".join(errors))
    return curves


def plot_ablation(frame, output_dir, *, games=GAMES, budgets=(1, 2, 4), smoke=False):
    _check_comparison_config(frame, games, [alg for k in budgets for alg in ablation_pool(k)])
    curves = _curves_for_all(frame, games, [(k, ablation_pool(k)) for k in budgets])
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    with plt.rc_context():
        set_plt_params(font_size=29)
        plt.rcParams.update(PLOT_STYLE)
        for runtime in (False, True):
            fig, axes = plt.subplots(len(games), len(budgets),
                                     figsize=(7 * len(budgets), 3.6 * len(games)),
                                     squeeze=False)
            for i, game in enumerate(games):
                # Common payoff limits across budgets within this game;
                # include every curve and preserve an autoscale margin.
                lows, highs = [], []
                for j, k in enumerate(budgets):
                    ax = axes[i, j]
                    for row, trajectory in curves[game, k].items():
                        y, ylabel = payoff_units(game, trajectory.value)
                        x = trajectory.runtime_seconds if runtime else np.arange(1, len(y) + 1)
                        full = "QBC" in row
                        label = (rf"$\mathrm{{SCD}}_{{\leq {k}}}$" if full else
                                 rf"$\mathrm{{SCD}}_{{{k}}}$")
                        ax.plot(x, y, label=label, color=COLORS[row],
                                linestyle=(0, (5, 2)) if full else "-",
                                linewidth=2.1, solid_capstyle="round",
                                dash_capstyle="round")
                    _style_axis(ax, ylabel, runtime=runtime,
                                show_xlabel=i == len(games) - 1)
                    _row_legend(ax, loc="lower right" if game == "sheriff" else "upper right")
                    lows.append(ax.get_ylim()[0]); highs.append(ax.get_ylim()[1])
                for ax in axes[i]:
                    ax.set_ylim(min(lows), max(highs))
                axes[i, len(budgets) // 2].set_title(_game_title(game), pad=12)
            if smoke:
                fig.suptitle("SMOKE / validation only", fontsize=29)
            fig.tight_layout(pad=.8, h_pad=.8, w_pad=.9,
                             rect=(0, 0, 1, .98 if smoke else 1))
            stem = "SCD_ablation_runtime" if runtime else "SCD_ablation_round"
            if smoke: stem += "_smoke"
            for extension in ("pdf", "png"):
                path = output_dir / f"{stem}.{extension}"
                fig.savefig(path, dpi=200, bbox_inches="tight")
                outputs.append(path)
            plt.close(fig)
    return outputs


def plot_main(frame, output_dir, *, games=GAMES, metric="osr", smoke=False):
    _check_comparison_config(frame, games, MAIN_POOL)
    curves = _curves_for_all(frame, games, [(None, MAIN_POOL)], metric=(metric,))
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    legacy = "regret_updates" not in frame or frame.loc[
        frame.row_alg.isin(MAIN_POOL), "regret_updates"].isna().any()
    line_styles = dict(zip(MAIN_POOL, (
        "-", (0, (5, 2)), (0, (3, 1, 1, 1)), "-", ":",
        (0, (4, 2)), (0, (6, 3)),
    )))
    plot_labels = dict(LABELS)
    for k in (0, 1, 2, 4):
        plot_labels["CFR" if k == 0 else f"{k}-QB-EFR"] = rf"$\mathrm{{SCD}}_{{{k}}}$"
    plot_labels["A-EFR_IN"] = r"$\mathrm{ACT}_{\mathrm{IN}}$"
    outputs = []
    with plt.rc_context():
        # Match the learning-curve font setup in efr_data.ipynb, including
        # its LaTeX serif rendering when LaTeX is available.
        set_plt_params(font_size=29)
        plt.rcParams.update(PLOT_STYLE)
        fig, axes = plt.subplots(len(games), 3,
                                 figsize=(21, 3.6 * len(games)), squeeze=False)
        for i, game in enumerate(games):
            # Share only payoff limits; regret and runtime use separate scales.
            axes[i, 2].sharey(axes[i, 1])
            for row, trajectory in curves[game, None].items():
                rounds = np.arange(1, len(trajectory) + 1)
                y, ylabel = payoff_units(game, trajectory.value)
                options = dict(label=plot_labels[row], color=COLORS[row],
                               linestyle=line_styles[row], linewidth=2.1,
                               solid_capstyle="round", dash_capstyle="round")
                axes[i, 0].plot(rounds, trajectory[metric], **options)
                axes[i, 1].plot(rounds, y, **options)
                axes[i, 2].plot(trajectory.runtime_seconds, y, **options)
            axes[i, 1].set_title(_game_title(game), pad=12)
            for j, ax in enumerate(axes[i]):
                axis_label = (("Legacy local solver regret / (t+1)" if legacy else "avg. OSR gap") if j == 0 and metric == "osr" else
                              "Raw retained local regret / updates" if j == 0 else ylabel)
                _style_axis(ax, axis_label, runtime=j == 2,
                            show_xlabel=i == len(games) - 1)
            _row_legend(axes[i, 0])
        if smoke:
            fig.suptitle("SMOKE / validation only", fontsize=29)
        fig.tight_layout(pad=.8, h_pad=.8, w_pad=.9,
                         rect=(0, 0, 1, .98 if smoke else 1))
        for extension in ("pdf", "png"):
            path = output_dir / f"SCD_main{'_smoke' if smoke else ''}.{extension}"
            fig.savefig(path, dpi=200, bbox_inches="tight")
            outputs.append(path)
        plt.close(fig)
    return outputs


def _scientific_signature(config, row):
    # Across independent main/ablation batches, thread counts and executable
    # build bytes can differ. Scientific settings and shared engine sources must
    # still match. Timing consistency remains strict within each plotted group.
    keys = ("cfr_plus", "sampler", "epsilon", "round_semantics", "iterations",
            "game_definition", "record_format", "regret_definition")
    engine = {name: digest for name, digest in config.get("source_hashes", {}).items()
              if name.startswith("hr_edl/src/hr_edl/") or name in ("hr_edl/src/bin/run_fixed_ltbr.cc", "hr_edl/src/bin/ltbr.h")}
    required = ("run_fixed_ltbr.cc", "ltbr.h", "tabular_learner.h", "adaptive_profile.h")
    if any(not any(name.endswith("/" + needed) for name in engine) for needed in required):
        raise ValueError("Shared engine source fingerprints are unavailable")
    settings = config.get("algorithm_settings", {}).get(row)
    if not settings or any(config.get(key) is None for key in keys):
        raise ValueError(f"{row}: scientific configuration is unavailable")
    return json.dumps({"config": {key: config[key] for key in keys},
                       "algorithm": settings, "engine": engine}, sort_keys=True)


def assert_main_ablation_compatible(main, ablation):
    """Check science/engine compatibility without equating independent runtimes."""
    games = set(main.game_tag) | set(ablation.game_tag)
    for game in games:
        for k in (1, 2, 4):
            row = f"{k}-QB-EFR"
            signatures = []
            for frame in (main, ablation):
                subset = frame[(frame.game_tag == game) & (frame.row_alg == row)]
                if subset.empty or "run_config" not in subset:
                    raise ValueError(f"{game}: cannot verify shared SCD_{k} configuration")
                signatures.append({_scientific_signature(config, row)
                                   for config, _ in _configurations(subset)})
            if signatures[0] != signatures[1] or len(signatures[0]) != 1:
                raise ValueError(f"{game}/SCD_{k}: scientific settings or shared engine sources differ")
