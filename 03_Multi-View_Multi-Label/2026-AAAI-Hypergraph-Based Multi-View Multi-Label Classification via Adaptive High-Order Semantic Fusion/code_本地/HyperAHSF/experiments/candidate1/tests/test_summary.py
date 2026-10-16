"""已完成实验产物的结果汇总集成测试。"""

import unittest
from pathlib import Path

from experiments.candidate1.summarize_results import summarize_results


PROJECT_ROOT = Path(__file__).resolve().parents[3]


class ResultSummaryIntegrationTests(unittest.TestCase):
    """保护最终研究摘要所需的变体、折数和机制判断字段。"""

    def test_completed_runs_produce_decision_fields(self):
        """完整五折产物应能汇总出 oracle 和是否继续研究的决策字段。"""
        summary = summarize_results(
            PROJECT_ROOT / "research_outputs" / "candidate1",
            PROJECT_ROOT / "full_run.log",
        )
        self.assertEqual(summary["candidate1"]["epoch_107"]["fold_count"], 5)
        self.assertIn("oracle", summary)
        self.assertIn("mechanism", summary)
        self.assertIn("recommend_continue", summary["mechanism"])


if __name__ == "__main__":
    unittest.main()
