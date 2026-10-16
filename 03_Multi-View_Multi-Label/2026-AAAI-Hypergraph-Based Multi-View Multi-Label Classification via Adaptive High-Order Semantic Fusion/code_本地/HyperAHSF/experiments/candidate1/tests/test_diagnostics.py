"""原始 HyperAHSF 构图诊断的真实数据集集成测试。"""

import unittest
from pathlib import Path

from experiments.candidate1.diagnose_graphs import collect_diagnostics


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_ROOT = PROJECT_ROOT / "Multi_View_dataset_k_fold_pickle" / "emotions"


class DiagnosticIntegrationTests(unittest.TestCase):
    """保护默认阈值诊断结论与分视图统计。"""

    def test_default_threshold_reproduces_all_self_loop_validation_graphs(self):
        """test_thr=99 应过滤全部非自环候选，复现原验证图退化现象。"""
        result = collect_diagnostics(DATA_ROOT, folds=(0,), thresholds=(99.0,))
        default_rows = [
            row
            for row in result["rows"]
            if row["split"] == "val"
            and row["rule"] == "feature_cosine"
            and row["threshold"] == 99.0
        ]
        self.assertEqual(len(default_rows), 2)
        self.assertTrue(all(row["retained_nonself"] == 0 for row in default_rows))
        self.assertTrue(all(row["self_loop_incidence_ratio"] == 1.0 for row in default_rows))

    def test_view_cosine_distributions_are_recorded_separately(self):
        """不同视图的余弦分布必须分开保存，且当前数据中位数确有差异。"""
        result = collect_diagnostics(DATA_ROOT, folds=(0,), thresholds=(0.9,))
        quantiles = result["cosine_quantiles"]
        self.assertIn("val_view_0", quantiles)
        self.assertIn("val_view_1", quantiles)
        self.assertNotEqual(
            quantiles["val_view_0"]["q50"], quantiles["val_view_1"]["q50"]
        )


if __name__ == "__main__":
    unittest.main()
