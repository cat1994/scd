"""Integrity checks that protect archived measurements and reproduction inputs."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd
from bin import run_SCD_ablation as runner
from hr_edl_data.scd_analysis import (ablation_pool, assert_main_ablation_compatible,
                                     curve_data, load_raw_data, plot_ablation)
from hr_edl_data.tournament_records import metadata_path, validate_tournament_result


class IntegrityTests(unittest.TestCase):
    def write_tournament(self, root):
        path = Path(root) / "scd_ablation.kuhn_3p.null.fixed.gen.1.ssv"
        algorithms = ablation_pool(1)
        text = "# each row learner has t+1 updates\n# cfr_plus = false\nlearner/env  " + "  ".join(algorithms) + "\n"
        for t in range(2):
            text += f"t = {t}\n"
            for row in (*algorithms, "BR"):
                text += row + "  (1, 2, 3, 4, 0.1)  (3, 4, 5, 6, 0.2)\n"
        path.write_text(text)
        config = {"game": "kuhn_3p", "mode": "fixed", "scd_budget": 1,
                  "algorithms": list(algorithms), "iterations": 2,
                  "experiment_family": "SCD_ablation", "cfr_plus": False,
                  "threads_per_run": 1, "max_parallel_jobs": 1,
                  "round_semantics": "pre_update_ev_then_update_every_round",
                  "record_format": "tournament_tuple_5"}
        metadata_path(path).write_text(json.dumps({"run_id": "fixture", "status": "complete",
              "config": config, "output_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}))
        return path

    def test_complete_matrix_and_opponent_averaging(self):
        with tempfile.TemporaryDirectory() as root:
            path = self.write_tournament(root)
            self.assertEqual(validate_tournament_result(path, ablation_pool(1), 2), (True, "complete"))
            frame = load_raw_data(root)
            curves = curve_data(frame, "kuhn_3p", ablation_pool(1), ablation_pool(1), budget=1, metrics=("osr",))
            result = curves["1-SCD"].iloc[-1]
            self.assertEqual(result.value, 2.0)
            self.assertAlmostEqual(result.runtime_seconds, 0.006)
            self.assertEqual(result.osr, 2.0)
            mixed = frame.copy()
            changed = copy.deepcopy(mixed.iloc[0].run_config)
            changed["threads_per_run"] = 2
            mixed.at[mixed.index[0], "run_config"] = changed
            with self.assertRaisesRegex(ValueError, "mixed runtime"):
                curve_data(mixed, "kuhn_3p", ablation_pool(1), ablation_pool(1), budget=1)
            damaged = frame.drop(frame[(frame.row_alg == "1-SCD") & (frame.col_alg == "1-SCD_UNCOMPRESSED")].index)
            with self.assertRaisesRegex(ValueError, "missing pairs"):
                curve_data(damaged, "kuhn_3p", ablation_pool(1), ablation_pool(1), budget=1)
            with self.assertRaisesRegex(ValueError, "duplicate matchup"):
                curve_data(pd.concat([frame, frame.iloc[[0]]]), "kuhn_3p", ablation_pool(1), ablation_pool(1), budget=1)

    def test_raw_hash_and_matrix_tampering_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            path = self.write_tournament(root)
            path.write_text(path.read_text().replace("1-SCD_UNCOMPRESSED  (", "other  (", 1))
            with self.assertRaisesRegex(ValueError, "fingerprint mismatch"):
                load_raw_data(root)
            valid, reason = validate_tournament_result(path, ablation_pool(1), 2)
            self.assertFalse(valid)
            self.assertIn("incomplete matrix", reason)

    def test_existing_result_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            executable = root / "run_fixed_ltbr"
            marker = root / "executed"
            executable.write_text(f"#!/bin/sh\ntouch '{marker}'\n")
            executable.chmod(0o755)
            args = runner.parse_args(["--games", "kuhn_3p", "--k", "1", "--iterations", "2",
                                      "--exe-dir", str(root), "--data-dir", str(root)])
            output = root / "scd_ablation.kuhn_3p.null.fixed.gen.1.ssv"
            output.write_bytes(b"existing result, even if incomplete\n")
            before = {p.name: p.read_bytes() for p in root.iterdir()}
            self.assertEqual(runner.execute_matrix(args), 1)
            self.assertFalse(marker.exists())
            self.assertEqual(before, {p.name: p.read_bytes() for p in root.iterdir()})

    def test_scd_launcher_commands_and_metadata_use_published_names(self):
        with tempfile.TemporaryDirectory() as root:
            executable = Path(root) / "run_fixed_ltbr"
            executable.write_text("#!/bin/sh\nexit 0\n")
            executable.chmod(0o755)
            snapshot = {"source_sha256": "fixture", "source_hashes": {}}
            for main, budgets in ((False, (1, 2, 4)), (True, (4,))):
                args = runner.parse_args(["--games", "kuhn_3p", "--scd-budgets",
                                          *(str(k) for k in budgets), "--exe-dir", root,
                                          "--data-dir", root], main_experiment=main)
                experiments = runner.launcher.build_experiments(args)
                self.assertEqual(len(experiments), len(budgets))
                for experiment, budget in zip(experiments, budgets):
                    group = "scd_main" if main else "scd_ablation"
                    self.assertEqual(experiment.output_path.name,
                                     f"{group}.kuhn_3p.null.fixed.gen.{budget}.ssv")
                    self.assertIn(f"--scd_budget={budget}", experiment.command)
                    config = runner.task_config(experiment, args, snapshot)
                    self.assertEqual(config["algorithm_group"], group)
                    self.assertEqual(config["scd_budget"], budget)
                    settings = config["algorithm_settings"]
                    self.assertEqual(settings[f"{budget}-SCD"]["predecessors"],
                                     "SCDPredecessors")
                    if not main:
                        self.assertEqual(config["algorithms"],
                                         [f"{budget}-SCD", f"{budget}-SCD_UNCOMPRESSED"])
                        self.assertEqual(settings[f"{budget}-SCD_UNCOMPRESSED"]["predecessors"],
                                         "UncompressedSCDPredecessors")

    def test_scd_variant_settings_preserve_sequential_and_causal_suffixes(self):
        for variant, predecessors in (("", "SCDPredecessors"),
                                      ("_UNCOMPRESSED", "UncompressedSCDPredecessors")):
            for suffix, sequential, causal in (("", False, False), ("_SEQ", True, False),
                                               ("_CS", False, True), ("_SEQ+CS", True, True)):
                with self.subTest(variant=variant, suffix=suffix):
                    settings = runner.algorithm_settings(f"2-SCD{variant}{suffix}", True)
                    self.assertEqual(settings["scd_budget"], 2)
                    self.assertEqual(settings["predecessors"], predecessors)
                    self.assertEqual(settings["sequential"], sequential)
                    self.assertEqual(settings["causal"], causal)
                    self.assertTrue(settings["cfr_plus"])

    def test_renamed_scd_ablation_data_can_render_figures(self):
        with tempfile.TemporaryDirectory() as root:
            self.write_tournament(root)
            frame = load_raw_data(root)
            self.assertEqual(set(frame.scd_budget), {1})
            outputs = plot_ablation(frame, Path(root) / "figures", games=("kuhn_3p",),
                                    budgets=(1,), smoke=True)
            self.assertEqual(len(outputs), 4)
            self.assertTrue(all(path.is_file() and path.stat().st_size > 0 for path in outputs))

    def test_snapshot_needs_no_git_head_or_identity(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            (root / "bin").mkdir()
            (root / "bin/example.py").write_text("print(1)\n")
            with patch.object(runner.launcher, "PROJECT_ROOT", root):
                snapshot = runner.repository_snapshot()
            self.assertEqual(set(snapshot), {"source_sha256", "source_hashes"})
            self.assertEqual(set(snapshot["source_hashes"]), {"bin/example.py"})
            self.assertNotIn(str(root), json.dumps(snapshot))

    def test_separate_batches_allow_threads_but_require_same_engine(self):
        engine = {"hr_edl/src/bin/run_fixed_ltbr.cc": "a", "hr_edl/src/bin/ltbr.h": "b",
                  "hr_edl/src/hr_edl/tabular_learner.h": "c", "hr_edl/src/hr_edl/adaptive_profile.h": "d"}
        records = []
        for k in (1, 2, 4):
            row = f"{k}-SCD"
            config = {"cfr_plus": False, "sampler": "null", "epsilon": 0.0,
                      "round_semantics": "pre_update_ev_then_update_every_round", "iterations": 1000,
                      "game_definition": "kuhn_poker(players=3)", "record_format": "tournament_tuple_5",
                      "regret_definition": "maximum maintained local regret", "source_hashes": engine,
                      "threads_per_run": 4, "max_parallel_jobs": 8,
                      "algorithm_settings": {row: {"scd_budget": k, "sequential": False, "causal": False}}}
            records.append({"game_tag": "kuhn_3p", "row_alg": row, "run_config": config, "cfr_plus": False})
        main = pd.DataFrame(records)
        other = copy.deepcopy(records)
        for record in other:
            record["run_config"].update(threads_per_run=2, max_parallel_jobs=6)
        assert_main_ablation_compatible(main, pd.DataFrame(other))
        other[0]["run_config"]["source_hashes"]["hr_edl/src/bin/ltbr.h"] = "changed"
        with self.assertRaisesRegex(ValueError, "engine sources differ"):
            assert_main_ablation_compatible(main, pd.DataFrame(other))


if __name__ == "__main__":
    unittest.main()
