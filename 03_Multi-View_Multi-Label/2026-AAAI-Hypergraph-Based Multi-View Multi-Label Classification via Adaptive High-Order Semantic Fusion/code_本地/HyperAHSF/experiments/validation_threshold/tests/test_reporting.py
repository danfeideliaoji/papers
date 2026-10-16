from __future__ import annotations

from pathlib import Path

import pytest


METRICS = {
    "HM": 0.2,
    "AP": 0.7,
    "OE": 0.3,
    "RL": 0.2,
    "subset": 0.1,
    "coverage": 1.5,
    "macro": 0.55,
    "micro": 0.6,
}


def test_metric_row_contains_identity_threshold_and_eight_metrics() -> None:
    from experiments.validation_threshold.reporting import metric_row

    row = metric_row(
        variant="cos_090",
        fold=0,
        epoch=3,
        threshold=0.9,
        loss=1.0,
        metrics=METRICS,
        epoch_seconds=0.01,
    )

    assert [row[key] for key in ("variant", "fold", "epoch", "threshold")] == [
        "cos_090",
        0,
        3,
        0.9,
    ]
    metric_columns = {
        "Hamming Loss",
        "Average Precision",
        "One Error",
        "Ranking Loss",
        "Subset Accuracy",
        "Coverage",
        "Macro-F1",
        "Micro-F1",
    }
    assert metric_columns <= set(row)


def _row(variant: str, fold: int, epoch: int, threshold: float, ap: float):
    from experiments.validation_threshold.reporting import metric_row

    return metric_row(
        variant=variant,
        fold=fold,
        epoch=epoch,
        threshold=threshold,
        loss=1.0,
        metrics={**METRICS, "AP": ap},
        epoch_seconds=0.01,
    )


def test_summary_uses_ap_best_epoch_and_reports_std_and_fold_wins() -> None:
    from experiments.validation_threshold.reporting import summarize_thresholds

    rows = [
        _row("author_99", 0, 0, 99.0, 0.70),
        _row("author_99", 0, 1, 99.0, 0.69),
        _row("cos_090", 0, 0, 0.9, 0.75),
        _row("cos_090", 0, 1, 0.9, 0.74),
        _row("author_99", 1, 0, 99.0, 0.80),
        _row("cos_090", 1, 0, 0.9, 0.79),
    ]

    summary = {row["variant"]: row for row in summarize_thresholds(rows)}

    assert summary["author_99"]["Average Precision_mean"] == 0.75
    assert summary["author_99"]["mean_best_epoch"] == 0.0
    assert summary["cos_090"]["AP_wins_vs_author_99"] == 1
    assert summary["cos_090"]["AP_ties_vs_author_99"] == 0
    assert summary["cos_090"]["AP_losses_vs_author_99"] == 1
    assert summary["author_99"]["Average Precision_std"] > 0


def test_report_keeps_all_variants_and_states_diagnostic_boundary(
    tmp_path: Path,
) -> None:
    from experiments.validation_threshold.reporting import write_report

    summaries = [
        {
            "variant": "author_99",
            "threshold": 99.0,
            "fold_count": 1,
            "mean_best_epoch": 3.0,
            "Average Precision_mean": 0.70,
            "Average Precision_std": 0.0,
            "Micro-F1_mean": 0.60,
            "Micro-F1_std": 0.0,
            "Macro-F1_mean": 0.55,
            "Macro-F1_std": 0.0,
            "Hamming Loss_mean": 0.20,
            "Hamming Loss_std": 0.0,
            "AP_wins_vs_author_99": 0,
            "AP_ties_vs_author_99": 1,
            "AP_losses_vs_author_99": 0,
        },
        {
            "variant": "cos_090",
            "threshold": 0.9,
            "fold_count": 1,
            "mean_best_epoch": 4.0,
            "Average Precision_mean": 0.72,
            "Average Precision_std": 0.0,
            "Micro-F1_mean": 0.62,
            "Micro-F1_std": 0.0,
            "Macro-F1_mean": 0.57,
            "Macro-F1_std": 0.0,
            "Hamming Loss_mean": 0.19,
            "Hamming Loss_std": 0.0,
            "AP_wins_vs_author_99": 1,
            "AP_ties_vs_author_99": 0,
            "AP_losses_vs_author_99": 0,
        },
    ]
    path = tmp_path / "report.md"

    write_report(
        path,
        summaries=summaries,
        graph_summaries=[],
        timing={"total_seconds": 1.5, "average_epoch_seconds": 0.1},
    )

    report = path.read_text(encoding="utf-8")
    assert "author_99" in report
    assert "cos_090" in report
    assert "## 1 折结果" in report
    assert "cos_090 的平均 AP 最高" in report
    assert "事后诊断" in report
    assert "不能直接替换论文正式结果" in report


def test_graph_summary_keeps_views_separate() -> None:
    from experiments.validation_threshold.reporting import summarize_graph_statistics

    rows = [
        {
            "variant": "cos_090",
            "threshold": 0.9,
            "fold": 0,
            "epoch": 0,
            "view": 0,
            "non_self_loop_hyperedges": 2,
            "non_self_loop_ratio": 0.2,
            "mean_hyperedge_size": 1.4,
        },
        {
            "variant": "cos_090",
            "threshold": 0.9,
            "fold": 1,
            "epoch": 0,
            "view": 0,
            "non_self_loop_hyperedges": 4,
            "non_self_loop_ratio": 0.4,
            "mean_hyperedge_size": 1.8,
        },
        {
            "variant": "cos_090",
            "threshold": 0.9,
            "fold": 0,
            "epoch": 0,
            "view": 1,
            "non_self_loop_hyperedges": 8,
            "non_self_loop_ratio": 0.8,
            "mean_hyperedge_size": 2.6,
        },
    ]

    summary = summarize_graph_statistics(rows)

    assert len(summary) == 2
    assert summary[0]["view"] == 0
    assert summary[0]["non_self_loop_hyperedges_mean"] == 3.0
    assert summary[0]["non_self_loop_ratio_mean"] == pytest.approx(0.3)
    assert summary[1]["view"] == 1


def test_learning_curve_plot_is_written_as_svg(tmp_path: Path) -> None:
    from experiments.validation_threshold.reporting import plot_learning_curves

    rows = [
        _row("author_99", 0, 0, 99.0, 0.70),
        _row("author_99", 0, 1, 99.0, 0.72),
        _row("cos_090", 0, 0, 0.9, 0.71),
        _row("cos_090", 0, 1, 0.9, 0.73),
    ]
    path = tmp_path / "curves.svg"

    plot_learning_curves(path, rows)

    assert path.read_text(encoding="utf-8").lstrip().startswith("<?xml")
