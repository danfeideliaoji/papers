"""验证图余弦阈值诊断实验的固定变量与运行配置。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ThresholdVariant:
    """一个有稳定名称的验证图余弦阈值。"""

    name: str
    value: float


DEFAULT_THRESHOLDS = (
    ThresholdVariant("author_99", 99.0),
    ThresholdVariant("cos_099", 0.99),
    ThresholdVariant("cos_095", 0.95),
    ThresholdVariant("cos_090", 0.90),
    ThresholdVariant("knn_all", -1.0),
)


@dataclass(frozen=True)
class SweepConfig:
    """一次阈值诊断运行的不可变配置。"""

    project_root: Path
    output_dir: Path
    folds: tuple[int, ...] = (0, 1, 2, 3, 4)
    epochs: int = 125
    seed: int = 0
    device: str = "cuda:0"
    thresholds: tuple[ThresholdVariant, ...] = DEFAULT_THRESHOLDS

    def __post_init__(self) -> None:
        if not self.folds or any(fold not in range(5) for fold in self.folds):
            raise ValueError("fold 必须是 0..4 中至少一个编号。")
        if self.epochs <= 0:
            raise ValueError("epochs 必须为正整数。")
        names = [variant.name for variant in self.thresholds]
        values = [variant.value for variant in self.thresholds]
        if len(names) != len(set(names)) or len(values) != len(set(values)):
            raise ValueError("阈值名称和值必须唯一。")
