"""Candidate 1 实验运行器的端到端冒烟与可复现性测试。"""

import tempfile
import unittest
from pathlib import Path

from experiments.candidate1.run_experiments import ExperimentConfig, run_variant


PROJECT_ROOT = Path(__file__).resolve().parents[3]


class ExperimentRunnerIntegrationTests(unittest.TestCase):
    """使用一折一 epoch 的最小配置验证训练、评估和文件产物。"""

    def test_one_epoch_smoke_run_produces_metrics_and_graph_evidence(self):
        """一次最小训练应同时产生分类指标、图证据和两个 CSV 文件。"""
        with tempfile.TemporaryDirectory() as directory:
            config = ExperimentConfig(
                project_root=PROJECT_ROOT,
                output_root=Path(directory),
                folds=(0,),
                epochs=1,
                warmup_epochs=0,
                graph_refresh=1,
                save_checkpoints=False,
            )
            result = run_variant("candidate1", config)
            self.assertEqual(len(result["epoch_rows"]), 1)
            self.assertIn("AP", result["epoch_rows"][0])
            self.assertEqual(len(result["graph_rows"]), 2)
            self.assertIn("zero_overlap_mass_rate", result["graph_rows"][0])
            self.assertTrue((Path(directory) / "candidate1" / "metrics.csv").exists())
            self.assertTrue((Path(directory) / "candidate1" / "graph_metrics.csv").exists())

    def test_same_seed_reproduces_one_epoch_metrics(self):
        """相同种子和配置的两次独立运行应得到完全相同的关键指标。"""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first_config = ExperimentConfig(
                project_root=PROJECT_ROOT,
                output_root=root / "first",
                folds=(0,),
                epochs=1,
                warmup_epochs=0,
                graph_refresh=1,
                save_checkpoints=False,
                seed=0,
            )
            second_config = ExperimentConfig(
                project_root=PROJECT_ROOT,
                output_root=root / "second",
                folds=(0,),
                epochs=1,
                warmup_epochs=0,
                graph_refresh=1,
                save_checkpoints=False,
                seed=0,
            )
            first = run_variant("candidate1", first_config)["epoch_rows"][0]
            second = run_variant("candidate1", second_config)["epoch_rows"][0]
            for key in ("loss", "AP", "micro", "HM"):
                self.assertEqual(first[key], second[key])


if __name__ == "__main__":
    unittest.main()
