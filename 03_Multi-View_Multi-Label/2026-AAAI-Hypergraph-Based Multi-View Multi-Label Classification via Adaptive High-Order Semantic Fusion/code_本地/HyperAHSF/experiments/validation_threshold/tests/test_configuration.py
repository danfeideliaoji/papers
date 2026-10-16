from __future__ import annotations

from pathlib import Path

import pytest


def test_default_thresholds_preserve_author_value_and_order() -> None:
    from experiments.validation_threshold.configuration import DEFAULT_THRESHOLDS

    assert [(item.name, item.value) for item in DEFAULT_THRESHOLDS] == [
        ("author_99", 99.0),
        ("cos_099", 0.99),
        ("cos_095", 0.95),
        ("cos_090", 0.90),
        ("knn_all", -1.0),
    ]


def test_sweep_defaults_to_five_folds_and_125_epochs(tmp_path: Path) -> None:
    from experiments.validation_threshold.configuration import SweepConfig

    config = SweepConfig(project_root=tmp_path, output_dir=tmp_path / "out")

    assert config.folds == (0, 1, 2, 3, 4)
    assert config.epochs == 125
    assert config.seed == 0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"folds": ()},
        {"folds": (5,)},
        {"epochs": 0},
    ],
)
def test_sweep_rejects_invalid_fold_and_epoch_values(
    tmp_path: Path, kwargs: dict[str, object]
) -> None:
    from experiments.validation_threshold.configuration import SweepConfig

    with pytest.raises(ValueError):
        SweepConfig(
            project_root=tmp_path,
            output_dir=tmp_path / "out",
            **kwargs,
        )


@pytest.mark.parametrize(
    "variants",
    [
        (("same", 0.9), ("same", 0.8)),
        (("first", 0.9), ("second", 0.9)),
    ],
)
def test_sweep_requires_unique_threshold_names_and_values(
    tmp_path: Path, variants: tuple[tuple[str, float], ...]
) -> None:
    from experiments.validation_threshold.configuration import (
        SweepConfig,
        ThresholdVariant,
    )

    with pytest.raises(ValueError, match="唯一"):
        SweepConfig(
            project_root=tmp_path,
            output_dir=tmp_path / "out",
            thresholds=tuple(ThresholdVariant(*item) for item in variants),
        )
