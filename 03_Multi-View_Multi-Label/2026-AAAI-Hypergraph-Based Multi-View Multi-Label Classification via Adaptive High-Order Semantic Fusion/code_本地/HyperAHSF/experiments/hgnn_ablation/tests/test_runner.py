from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from experiments.hgnn_ablation.configuration import AblationFlags
from experiments.hgnn_ablation import runner as runner_module
try:
    from experiments.hgnn_ablation.runner import (
        ExperimentConfig,
        flags_for_variant,
        validate_branch_support,
    )
except ModuleNotFoundError as exc:
    RUNNER_IMPORT_ERROR = exc
else:
    RUNNER_IMPORT_ERROR = None


def _require_runner_api() -> None:
    if RUNNER_IMPORT_ERROR is not None:
        pytest.fail(f"common runner module is missing: {RUNNER_IMPORT_ERROR}")


def test_baseline_config_uses_all_five_folds_and_125_epochs(tmp_path: Path) -> None:
    _require_runner_api()
    config = ExperimentConfig(project_root=tmp_path, output_root=tmp_path / "outputs")

    assert config.variant == "baseline"
    assert config.folds == (0, 1, 2, 3, 4)
    assert config.epochs == 125
    assert config.seed == 0


@pytest.mark.parametrize(
    ("variant", "expected"),
    [
        ("baseline", AblationFlags()),
        ("residual_only", AblationFlags(use_residual=True)),
        (
            "hyperedge_weight_only",
            AblationFlags(learnable_hyperedge_weight=True),
        ),
        ("hypergraph_attention_only", AblationFlags(use_attention=True)),
    ],
)
def test_variant_maps_to_exactly_one_flag(variant: str, expected: AblationFlags) -> None:
    _require_runner_api()
    assert flags_for_variant(variant) == expected


def test_common_branch_rejects_unimplemented_mechanisms() -> None:
    _require_runner_api()
    validate_branch_support("baseline", supported_variants={"baseline"})

    with pytest.raises(ValueError, match="当前分支不支持"):
        validate_branch_support("residual_only", supported_variants={"baseline"})


def test_config_rejects_invalid_fold_and_epoch_values(tmp_path: Path) -> None:
    _require_runner_api()
    with pytest.raises(ValueError, match="fold"):
        ExperimentConfig(
            project_root=tmp_path,
            output_root=tmp_path / "outputs",
            folds=(5,),
        )
    with pytest.raises(ValueError, match="epochs"):
        ExperimentConfig(
            project_root=tmp_path,
            output_root=tmp_path / "outputs",
            epochs=0,
        )


def test_real_baseline_one_epoch_cpu_writes_complete_artifacts(tmp_path: Path) -> None:
    run_experiment = getattr(runner_module, "run_experiment", None)
    assert callable(run_experiment), "run_experiment has not been implemented"
    project_root = Path(__file__).resolve().parents[3]
    config = ExperimentConfig(
        project_root=project_root,
        output_root=tmp_path / "outputs",
        variant="baseline",
        folds=(0,),
        epochs=1,
        device="cpu",
    )

    result = run_experiment(config)

    output_dir = config.output_root / "baseline"
    required_files = {
        "config.json",
        "metrics.csv",
        "fold_summary.csv",
        "timing.json",
        "parameter_count.json",
        "learning_curves.svg",
        "单机制实验报告.md",
    }
    assert required_files <= {path.name for path in output_dir.iterdir()}
    with (output_dir / "metrics.csv").open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert rows[0]["variant"] == "baseline"
    assert rows[0]["fold"] == "0"
    assert rows[0]["epoch"] == "0"
    with (output_dir / "parameter_count.json").open("r", encoding="utf-8") as handle:
        parameter_data = json.load(handle)
    assert parameter_data["trainable_parameters"] > 0
    with (output_dir / "timing.json").open("r", encoding="utf-8") as handle:
        timing_data = json.load(handle)
    assert timing_data["fold_count"] == 1
    assert timing_data["total_seconds"] > 0
    assert result["metrics_rows"] == 1
