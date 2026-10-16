"""公共实验结果的无损保存与逐折摘要。"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


METRIC_NAME_MAP = {
    "HM": "Hamming Loss",
    "AP": "Average Precision",
    "OE": "One Error",
    "RL": "Ranking Loss",
    "subset": "Subset Accuracy",
    "coverage": "Coverage",
    "macro": "Macro-F1",
    "micro": "Micro-F1",
}

METRIC_COLUMNS = tuple(METRIC_NAME_MAP.values())
REQUIRED_METRIC_COLUMNS = {
    "variant",
    "fold",
    "epoch",
    *METRIC_COLUMNS,
}


def metric_row(
    *,
    variant: str,
    fold: int,
    epoch: int,
    loss: float,
    metrics: Mapping[str, float],
    epoch_seconds: float,
) -> dict[str, object]:
    """把项目原有八项缩写指标转换为报告要求的明确列名。"""

    missing = set(METRIC_NAME_MAP) - set(metrics)
    if missing:
        raise ValueError(f"缺少原有指标: {sorted(missing)}")
    row: dict[str, object] = {
        "variant": variant,
        "fold": int(fold),
        "epoch": int(epoch),
        "loss": float(loss),
        "epoch_seconds": float(epoch_seconds),
    }
    row.update({long_name: float(metrics[short]) for short, long_name in METRIC_NAME_MAP.items()})
    return row


def prepare_output_dir(path: Path) -> Path:
    """创建空输出目录，并拒绝覆盖其中的任何已有材料。"""

    path = Path(path)
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"已有实验结果，拒绝覆盖: {path}")
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    """按首行键顺序写 UTF-8 BOM CSV，便于 Excel 直接打开。"""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def summarize_folds(rows: Iterable[Mapping[str, object]]) -> list[dict[str, object]]:
    """按折选择 AP 最佳 epoch，同时保存逐指标最优值与最后 epoch。"""

    grouped: dict[int, list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[int(row["fold"])].append(row)

    summaries: list[dict[str, object]] = []
    lower_is_better = {"Hamming Loss", "One Error", "Ranking Loss", "Coverage"}
    for fold in sorted(grouped):
        fold_rows = sorted(grouped[fold], key=lambda item: int(item["epoch"]))
        best_ap_row = max(fold_rows, key=lambda item: float(item["Average Precision"]))
        final_row = fold_rows[-1]
        summary: dict[str, object] = {
            "variant": best_ap_row["variant"],
            "fold": fold,
            "best_epoch": int(best_ap_row["epoch"]),
            "final_epoch": int(final_row["epoch"]),
        }
        for metric in METRIC_COLUMNS:
            chooser = min if metric in lower_is_better else max
            best_metric_row = chooser(fold_rows, key=lambda item: float(item[metric]))
            summary[f"best_{metric}"] = float(best_metric_row[metric])
            summary[f"best_{metric}_epoch"] = int(best_metric_row["epoch"])
            summary[f"{metric}_at_best_epoch"] = float(best_ap_row[metric])
            summary[f"final_{metric}"] = float(final_row[metric])
        summaries.append(summary)
    return summaries


def write_json(path: Path, payload: Mapping[str, object]) -> None:
    """以 UTF-8 和稳定缩进写 JSON。"""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def plot_learning_curves(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    """绘制每折 AP、Micro/Macro-F1 与 Hamming Loss 学习曲线。"""

    path = Path(path)
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    metrics = ("Average Precision", "Micro-F1", "Macro-F1", "Hamming Loss")
    folds = sorted({int(row["fold"]) for row in rows})
    for axis, metric in zip(axes.flat, metrics):
        for fold in folds:
            fold_rows = sorted(
                (row for row in rows if int(row["fold"]) == fold),
                key=lambda row: int(row["epoch"]),
            )
            axis.plot(
                [int(row["epoch"]) for row in fold_rows],
                [float(row[metric]) for row in fold_rows],
                label=f"fold{fold}",
                linewidth=1.2,
            )
        axis.set_title(metric)
        axis.set_xlabel("epoch")
        axis.grid(alpha=0.25)
        axis.legend()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, format="svg")
    plt.close(figure)


def write_variant_report(
    path: Path,
    *,
    variant: str,
    summaries: Sequence[Mapping[str, object]],
    timing: Mapping[str, object],
    trainable_parameters: int,
) -> None:
    """写出不预设优劣的单变体运行事实报告。"""

    lines = [
        f"# {variant} 单机制实验报告",
        "",
        "## 运行设置",
        "",
        f"- 变体：`{variant}`",
        f"- 折数：{len(summaries)}",
        f"- 可训练参数量：{trainable_parameters}",
        f"- 总时间（秒）：{float(timing['total_seconds']):.6f}",
        f"- 平均每 epoch 时间（秒）：{float(timing['average_epoch_seconds']):.6f}",
        "",
        "## 逐折结果（各折以 AP 最佳 epoch 为主要快照）",
        "",
        "| fold | 最佳 epoch | AP | Micro-F1 | Macro-F1 | HL | 最后 epoch | 最后 AP |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        lines.append(
            "| {fold} | {best_epoch} | {ap:.4f} | {micro:.4f} | {macro:.4f} | "
            "{hl:.4f} | {final_epoch} | {final_ap:.4f} |".format(
                fold=int(row["fold"]),
                best_epoch=int(row["best_epoch"]),
                ap=float(row["Average Precision_at_best_epoch"]),
                micro=float(row["Micro-F1_at_best_epoch"]),
                macro=float(row["Macro-F1_at_best_epoch"]),
                hl=float(row["Hamming Loss_at_best_epoch"]),
                final_epoch=int(row["final_epoch"]),
                final_ap=float(row["final_Average Precision"]),
            )
        )
    lines.extend(
        [
            "",
            "## 说明",
            "",
            "本报告只记录该独立变体的完整运行事实；是否有效必须在最终汇总中与同一公共 baseline 的五折结果比较后判断。",
            "",
        ]
    )
    Path(path).write_text("\n".join(lines), encoding="utf-8")
