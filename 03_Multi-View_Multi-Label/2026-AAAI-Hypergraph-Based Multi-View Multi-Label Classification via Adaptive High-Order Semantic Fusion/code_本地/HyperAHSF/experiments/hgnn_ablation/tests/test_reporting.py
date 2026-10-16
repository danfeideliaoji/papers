from __future__ import annotations

import csv
from pathlib import Path

import pytest

try:
    from experiments.hgnn_ablation.reporting import (
        REQUIRED_METRIC_COLUMNS,
        metric_row,
        prepare_output_dir,
        summarize_folds,
        write_csv,
    )
except ModuleNotFoundError as exc:
    REPORTING_IMPORT_ERROR = exc
else:
    REPORTING_IMPORT_ERROR = None


def _require_reporting_api() -> None:
    if REPORTING_IMPORT_ERROR is not None:
        pytest.fail(f"common reporting module is missing: {REPORTING_IMPORT_ERROR}")


def _epoch_row(fold: int, epoch: int, ap: float, micro: float, hl: float) -> dict[str, object]:
    _require_reporting_api()
    return metric_row(
        variant="baseline",
        fold=fold,
        epoch=epoch,
        loss=1.0 - 0.1 * epoch,
        metrics={
            "HM": hl,
            "AP": ap,
            "OE": 0.3,
            "RL": 0.2,
            "subset": 0.1,
            "coverage": 1.5,
            "macro": micro - 0.05,
            "micro": micro,
        },
        epoch_seconds=0.01,
    )


def test_metric_row_contains_required_named_columns() -> None:
    _require_reporting_api()
    row = _epoch_row(fold=0, epoch=0, ap=0.7, micro=0.6, hl=0.2)

    assert REQUIRED_METRIC_COLUMNS <= set(row)
    assert row["Average Precision"] == 0.7
    assert row["Micro-F1"] == 0.6
    assert row["Hamming Loss"] == 0.2


def test_existing_nonempty_output_is_not_overwritten(tmp_path: Path) -> None:
    _require_reporting_api()
    output_dir = tmp_path / "baseline"
    output_dir.mkdir()
    (output_dir / "metrics.csv").write_text("existing", encoding="utf-8")

    with pytest.raises(FileExistsError, match="已有实验结果"):
        prepare_output_dir(output_dir)


def test_empty_output_directory_is_safe_to_use(tmp_path: Path) -> None:
    _require_reporting_api()
    output_dir = tmp_path / "baseline"
    output_dir.mkdir()

    assert prepare_output_dir(output_dir) == output_dir


def test_csv_writer_preserves_required_column_order(tmp_path: Path) -> None:
    _require_reporting_api()
    path = tmp_path / "metrics.csv"
    row = _epoch_row(fold=0, epoch=0, ap=0.7, micro=0.6, hl=0.2)

    write_csv(path, [row])

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames is not None
        assert reader.fieldnames[:3] == ["variant", "fold", "epoch"]
        assert next(reader)["Average Precision"] == "0.7"


def test_fold_summary_selects_ap_best_epoch_and_keeps_final_epoch() -> None:
    _require_reporting_api()
    rows = [
        _epoch_row(0, 0, ap=0.70, micro=0.60, hl=0.25),
        _epoch_row(0, 1, ap=0.75, micro=0.62, hl=0.23),
        _epoch_row(0, 2, ap=0.73, micro=0.64, hl=0.22),
    ]

    summary = summarize_folds(rows)

    assert len(summary) == 1
    assert summary[0]["best_epoch"] == 1
    assert summary[0]["best_Average Precision"] == 0.75
    assert summary[0]["Micro-F1_at_best_epoch"] == 0.62
    assert summary[0]["final_epoch"] == 2
    assert summary[0]["final_Micro-F1"] == 0.64
    assert summary[0]["best_Hamming Loss"] == 0.22
