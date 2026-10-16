from __future__ import annotations

import csv
import json
from pathlib import Path


def test_one_epoch_real_sweep_writes_all_threshold_rows(tmp_path: Path) -> None:
    from experiments.validation_threshold.configuration import SweepConfig
    from experiments.validation_threshold.runner import run_sweep

    project_root = Path(__file__).resolve().parents[3]
    output_dir = tmp_path / "sweep"

    result = run_sweep(
        SweepConfig(
            project_root=project_root,
            output_dir=output_dir,
            folds=(0,),
            epochs=1,
            device="cpu",
        )
    )

    required_files = {
        "metrics.csv",
        "graph_statistics.csv",
        "aggregate_results.csv",
        "config.json",
        "learning_curves.svg",
        "验证阈值诊断报告.md",
    }
    assert required_files <= {path.name for path in output_dir.iterdir()}
    with (output_dir / "metrics.csv").open(
        "r", encoding="utf-8-sig", newline=""
    ) as handle:
        metrics_rows = list(csv.DictReader(handle))
    with (output_dir / "graph_statistics.csv").open(
        "r", encoding="utf-8-sig", newline=""
    ) as handle:
        graph_rows = list(csv.DictReader(handle))

    assert len(metrics_rows) == 5
    assert len(graph_rows) == 10
    assert {row["variant"] for row in metrics_rows} == {
        "author_99",
        "cos_099",
        "cos_095",
        "cos_090",
        "knn_all",
    }
    assert {row["threshold"] for row in metrics_rows} == {
        "99.0",
        "0.99",
        "0.95",
        "0.9",
        "-1.0",
    }
    assert all(
        int(row["non_self_loop_hyperedges"]) == 0
        for row in graph_rows
        if row["variant"] == "author_99"
    )
    with (output_dir / "config.json").open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    assert payload["source_test_thr"] == 99
    assert result["trained_models"] == 1
    assert result["metrics_rows"] == 5
