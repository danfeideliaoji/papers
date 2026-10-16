"""验证阈值实验的指标记录、聚合、绘图和中文报告。"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Mapping, Sequence

import matplotlib.pyplot as plt

from experiments.hgnn_ablation.reporting import METRIC_COLUMNS, METRIC_NAME_MAP


def metric_row(
    *,
    variant: str,
    fold: int,
    epoch: int,
    threshold: float,
    loss: float,
    metrics: Mapping[str, float],
    epoch_seconds: float,
) -> dict[str, object]:
    """把一次阈值验证转换为具有稳定列名的指标行。"""

    missing = set(METRIC_NAME_MAP) - set(metrics)
    if missing:
        raise ValueError(f"缺少原有指标: {sorted(missing)}")
    row: dict[str, object] = {
        "variant": variant,
        "fold": int(fold),
        "epoch": int(epoch),
        "threshold": float(threshold),
        "loss": float(loss),
        "epoch_seconds": float(epoch_seconds),
    }
    row.update(
        {
            long_name: float(metrics[short_name])
            for short_name, long_name in METRIC_NAME_MAP.items()
        }
    )
    return row


def summarize_thresholds(
    rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    """按每折 AP 最佳 epoch 汇总均值、样本标准差和相对 99 的胜负。"""

    grouped: dict[tuple[str, int], list[Mapping[str, object]]] = defaultdict(list)
    variant_order: list[str] = []
    for row in rows:
        variant = str(row["variant"])
        if variant not in variant_order:
            variant_order.append(variant)
        grouped[(variant, int(row["fold"]))].append(row)

    best_rows: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for (variant, _fold), fold_rows in grouped.items():
        best_rows[variant].append(
            max(fold_rows, key=lambda item: float(item["Average Precision"]))
        )
    for variant in best_rows:
        best_rows[variant].sort(key=lambda item: int(item["fold"]))

    baseline_by_fold = {
        int(row["fold"]): float(row["Average Precision"])
        for row in best_rows.get("author_99", [])
    }
    summaries: list[dict[str, object]] = []
    for variant in variant_order:
        selected = best_rows[variant]
        summary: dict[str, object] = {
            "variant": variant,
            "threshold": float(selected[0]["threshold"]),
            "fold_count": len(selected),
            "mean_best_epoch": mean(int(row["epoch"]) for row in selected),
        }
        for metric in METRIC_COLUMNS:
            values = [float(row[metric]) for row in selected]
            summary[f"{metric}_mean"] = mean(values)
            summary[f"{metric}_std"] = stdev(values) if len(values) > 1 else 0.0

        wins = ties = losses = 0
        for row in selected:
            fold = int(row["fold"])
            if fold not in baseline_by_fold:
                continue
            difference = float(row["Average Precision"]) - baseline_by_fold[fold]
            if abs(difference) <= 1e-12:
                ties += 1
            elif difference > 0:
                wins += 1
            else:
                losses += 1
        summary.update(
            {
                "AP_wins_vs_author_99": wins,
                "AP_ties_vs_author_99": ties,
                "AP_losses_vs_author_99": losses,
            }
        )
        summaries.append(summary)
    return summaries


def summarize_graph_statistics(
    rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    """按阈值和视图分别汇总图密度，避免掩盖跨视图差异。"""

    grouped: dict[tuple[str, int], list[Mapping[str, object]]] = defaultdict(list)
    key_order: list[tuple[str, int]] = []
    for row in rows:
        key = (str(row["variant"]), int(row["view"]))
        if key not in grouped:
            key_order.append(key)
        grouped[key].append(row)

    summaries: list[dict[str, object]] = []
    for variant, view in key_order:
        selected = grouped[(variant, view)]
        summaries.append(
            {
                "variant": variant,
                "threshold": float(selected[0]["threshold"]),
                "view": view,
                "observations": len(selected),
                "non_self_loop_hyperedges_mean": mean(
                    float(row["non_self_loop_hyperedges"]) for row in selected
                ),
                "non_self_loop_ratio_mean": mean(
                    float(row["non_self_loop_ratio"]) for row in selected
                ),
                "mean_hyperedge_size_mean": mean(
                    float(row["mean_hyperedge_size"]) for row in selected
                ),
            }
        )
    return summaries


def plot_learning_curves(
    path: Path,
    rows: Sequence[Mapping[str, object]],
) -> None:
    """绘制五个阈值跨折均值的 AP、F1 和 Hamming Loss 曲线。"""

    metrics = ("Average Precision", "Micro-F1", "Macro-F1", "Hamming Loss")
    variants = list(dict.fromkeys(str(row["variant"]) for row in rows))
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for axis, metric in zip(axes.flat, metrics):
        for variant in variants:
            variant_rows = [row for row in rows if row["variant"] == variant]
            epochs = sorted({int(row["epoch"]) for row in variant_rows})
            values = [
                mean(
                    float(row[metric])
                    for row in variant_rows
                    if int(row["epoch"]) == epoch
                )
                for epoch in epochs
            ]
            axis.plot(epochs, values, label=variant, linewidth=1.2)
        axis.set_title(metric)
        axis.set_xlabel("epoch")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, format="svg")
    plt.close(figure)


def write_report(
    path: Path,
    *,
    summaries: Sequence[Mapping[str, object]],
    graph_summaries: Sequence[Mapping[str, object]],
    timing: Mapping[str, object],
) -> None:
    """写出保留全部阈值、图密度和结论边界的中文诊断报告。"""

    fold_counts = {int(row["fold_count"]) for row in summaries}
    if len(fold_counts) != 1:
        raise ValueError("所有阈值汇总必须包含相同折数。")
    fold_count = next(iter(fold_counts))
    best_ap = max(summaries, key=lambda row: float(row["Average Precision_mean"]))
    best_variant = str(best_ap["variant"])
    best_ap_value = float(best_ap["Average Precision_mean"])
    if best_variant == "author_99":
        conclusion = (
            "本次比较没有发现合法余弦阈值在平均 AP 上超过作者 99，"
            "因此不支持仅凭本实验修改原值。"
        )
    else:
        conclusion = (
            "该优势仍是同一验证折上的事后比较，不能直接把该阈值作为正式修复值。"
        )
    lines = [
        "# 验证阈值诊断报告",
        "",
        "## 实验边界",
        "",
        "本实验复用同一训练权重，仅改变验证图的余弦过滤阈值。结果属于事后诊断，不能直接替换论文正式结果；正式选择阈值仍需要嵌套验证或独立测试集。",
        "",
        "## 主要判断",
        "",
        f"{best_variant} 的平均 AP 最高（{best_ap_value:.4f}）。{conclusion}",
        "",
        f"## {fold_count} 折结果（每折选择 AP 最佳 epoch）",
        "",
        "| 变体 | 阈值 | 最佳 epoch 均值 | AP | Micro-F1 | Macro-F1 | HL | AP 胜/平/负（相对 99） |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        lines.append(
            "| {variant} | {threshold:g} | {epoch:.1f} | {ap:.4f}±{ap_std:.4f} | "
            "{micro:.4f}±{micro_std:.4f} | {macro:.4f}±{macro_std:.4f} | "
            "{hl:.4f}±{hl_std:.4f} | {wins}/{ties}/{losses} |".format(
                variant=row["variant"],
                threshold=float(row["threshold"]),
                epoch=float(row["mean_best_epoch"]),
                ap=float(row["Average Precision_mean"]),
                ap_std=float(row["Average Precision_std"]),
                micro=float(row["Micro-F1_mean"]),
                micro_std=float(row["Micro-F1_std"]),
                macro=float(row["Macro-F1_mean"]),
                macro_std=float(row["Macro-F1_std"]),
                hl=float(row["Hamming Loss_mean"]),
                hl_std=float(row["Hamming Loss_std"]),
                wins=int(row["AP_wins_vs_author_99"]),
                ties=int(row["AP_ties_vs_author_99"]),
                losses=int(row["AP_losses_vs_author_99"]),
            )
        )

    lines.extend(["", "## 验证图结构", ""])
    if graph_summaries:
        lines.extend(
            [
                "| 变体 | 视图 | 非自环超边均值 | 非自环超边比例 | 平均超边大小 |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for row in graph_summaries:
            lines.append(
                "| {variant} | {view} | {edges:.2f} | {ratio:.4f} | {size:.4f} |".format(
                    variant=row["variant"],
                    view=int(row["view"]),
                    edges=float(row["non_self_loop_hyperedges_mean"]),
                    ratio=float(row["non_self_loop_ratio_mean"]),
                    size=float(row["mean_hyperedge_size_mean"]),
                )
            )
        lines.extend(
            [
                "",
                "固定阈值若在两个视图产生明显不同的非自环边比例，只能说明各视图余弦分布不可直接按同一数值比较，本实验不继续按视图调参。",
            ]
        )
    else:
        lines.append("当前调用未提供图结构汇总。")

    lines.extend(
        [
            "",
            "## 运行时间",
            "",
            f"- 总时间：{float(timing['total_seconds']):.3f} 秒",
            f"- 平均每个训练 epoch：{float(timing['average_epoch_seconds']):.3f} 秒",
            "",
            "## 解释限制",
            "",
            "即使某个合法余弦阈值优于 99，也只能说明保留验证近邻在本次诊断中有利，不能据此证明作者的 99 是笔误。",
            "",
        ]
    )
    Path(path).write_text("\n".join(lines), encoding="utf-8")
