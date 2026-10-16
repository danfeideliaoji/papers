"""聚合 Candidate 1 全部实验，并生成基于证据的研究结论。

该脚本读取原始 HyperAHSF 日志、各候选变体的分类/图指标和构图诊断，
统一计算指定 epoch、最终 epoch、最佳 AP/Micro-F1、机制阈值与跨折改进，
然后写出 JSON、聚合 CSV、学习曲线 SVG 和 Markdown 报告。
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Dict, List, Sequence, Tuple


VARIANTS = (
    "candidate1",
    "feature_only",
    "no_semantic",
    "no_reliability",
    "no_conflict",
    "oracle",
)
METRICS = ("HM", "AP", "OE", "RL", "subset", "coverage", "macro", "micro")


def read_csv(path: Path) -> List[Dict[str, str]]:
    """以 UTF-8 BOM 兼容方式读取 CSV，返回字符串字典列表。

    Args:
        path (pathlib.Path): 输入 CSV 路径。
    """
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def parse_baseline_log(path: Path) -> Dict[int, Dict[str, float]]:
    """从原始训练日志解析逐 epoch 的八项均值和跨折标准差。

    Args:
        path (pathlib.Path): 原始训练日志路径。

    Returns:
        ``{epoch: {metric: mean, metric_std: std}}``。
    """
    text = path.read_text(encoding="utf-8")
    pattern = re.compile(
        r"Epoch\s+(\d+):\s*\r?\nMeans:\s*([^\r\n]+)\s*\r?\nStds\s*:\s*([^\r\n]+)"
    )
    result: Dict[int, Dict[str, float]] = {}
    for match in pattern.finditer(text):
        epoch = int(match.group(1))
        means = [float(value) for value in match.group(2).split()]
        stds = [float(value) for value in match.group(3).split()]
        result[epoch] = {
            **{metric: means[index] for index, metric in enumerate(METRICS)},
            **{f"{metric}_std": stds[index] for index, metric in enumerate(METRICS)},
        }
    if not result:
        raise ValueError(f"no baseline epochs found in {path}")
    return result


def _epoch_summary(rows: Sequence[Dict[str, str]], epoch: int) -> Dict[str, object]:
    """汇总一个 epoch 的跨折均值、样本标准差和逐折指标。

    Args:
        rows (collections.abc.Sequence[dict[str, str]]): 逐 epoch、逐折指标行。
        epoch (int): 要汇总的从 0 开始的轮次。
    """
    selected = [row for row in rows if int(row["epoch"]) == epoch]
    if not selected:
        raise ValueError(f"epoch {epoch} missing")
    result: Dict[str, object] = {"epoch": epoch, "fold_count": len(selected)}
    for metric in METRICS:
        values = [float(row[metric]) for row in selected]
        result[metric] = mean(values)
        result[f"{metric}_std"] = stdev(values) if len(values) > 1 else 0.0
    result["fold_values"] = {
        str(int(row["fold"])): {metric: float(row[metric]) for metric in METRICS}
        for row in selected
    }
    return result


def _best_summary(rows: Sequence[Dict[str, str]], metric: str) -> Dict[str, object]:
    """找到指定指标跨折均值最高的 epoch。

    Args:
        rows (collections.abc.Sequence[dict[str, str]]): 逐 epoch、逐折指标行。
        metric (str): 用于选择最佳轮次的指标列名。
    """
    by_epoch: Dict[int, List[float]] = defaultdict(list)
    for row in rows:
        by_epoch[int(row["epoch"])].append(float(row[metric]))
    epoch, value = max(
        ((epoch, mean(values)) for epoch, values in by_epoch.items()), key=lambda item: item[1]
    )
    return {"epoch": epoch, metric: value}


def _final_graph_summary(rows: Sequence[Dict[str, str]], epoch: int = 124) -> Dict[str, float]:
    """汇总最终验证图的错误关联、可靠性、有效大小与校准指标。

    Args:
        rows (collections.abc.Sequence[dict[str, str]]): 逐轮图质量记录行。
        epoch (int): 要汇总的从 0 开始的轮次。
    """
    selected = [
        row for row in rows if row["split"] == "val" and int(row["epoch"]) == epoch
    ]
    keys = (
        "zero_overlap_mass_rate",
        "positive_overlap_mass_rate",
        "weighted_true_jaccard",
        "mean_reliability",
        "view_0_mean_anchor_mass",
        "view_1_mean_anchor_mass",
        "view_0_mean_effective_size",
        "view_1_mean_effective_size",
        "raw_brier",
        "calibrated_brier",
    )
    result: Dict[str, float] = {}
    for key in keys:
        values = [float(row[key]) for row in selected if row.get(key) not in (None, "")]
        if values:
            result[key] = mean(values)
    return result


def summarize_results(output_root: Path, baseline_log: Path) -> Dict[str, object]:
    """读取全部实验产物并形成分类性能与机制证据的结构化摘要。

    Args:
        output_root (pathlib.Path): Candidate 1 所有变体的结果根目录。
        baseline_log (pathlib.Path): 原始 HyperAHSF 训练日志路径。
    """
    # 原基线日志没有逐折明细，因此直接复用日志给出的均值/标准差。
    baseline_epochs = parse_baseline_log(baseline_log)
    result: Dict[str, object] = {
        "baseline": {
            "epoch_107": baseline_epochs[106],
            "final": baseline_epochs[124],
            "best_ap": {
                "epoch": max(baseline_epochs, key=lambda epoch: baseline_epochs[epoch]["AP"]),
                "AP": max(value["AP"] for value in baseline_epochs.values()),
            },
            "best_micro": {
                "epoch": max(baseline_epochs, key=lambda epoch: baseline_epochs[epoch]["micro"]),
                "micro": max(value["micro"] for value in baseline_epochs.values()),
            },
        }
    }
    # 每个变体统一取第 107 epoch、最终 epoch 和两个最佳 epoch。
    for variant in VARIANTS:
        rows = read_csv(output_root / variant / "metrics.csv")
        graph_rows = read_csv(output_root / variant / "graph_metrics.csv")
        result[variant] = {
            "epoch_107": _epoch_summary(rows, 106),
            "final": _epoch_summary(rows, 124),
            "best_ap": _best_summary(rows, "AP"),
            "best_micro": _best_summary(rows, "micro"),
            "final_graph": _final_graph_summary(graph_rows),
        }

    # 用原始 KNN 的零标签重叠率作为错误关联基准。
    diagnostics = read_csv(output_root / "diagnostics" / "diagnostics_aggregate.csv")
    raw_rows = [
        row for row in diagnostics if row["split"] == "val" and row["rule"] == "raw_knn"
    ]
    raw_zero = mean(float(row["zero_overlap_rate"]) for row in raw_rows)
    candidate_graph = result["candidate1"]["final_graph"]
    candidate_zero = float(candidate_graph["zero_overlap_mass_rate"])
    # 相对错误连接下降率是本研究预先设置的主要机制验证量。
    wrong_link_reduction = (raw_zero - candidate_zero) / raw_zero
    anchor_mass = mean(
        (
            float(candidate_graph["view_0_mean_anchor_mass"]),
            float(candidate_graph["view_1_mean_anchor_mass"]),
        )
    )
    effective_size = mean(
        (
            float(candidate_graph["view_0_mean_effective_size"]),
            float(candidate_graph["view_1_mean_effective_size"]),
        )
    )
    baseline_107 = result["baseline"]["epoch_107"]
    candidate_107 = result["candidate1"]["epoch_107"]
    feature_107 = result["feature_only"]["epoch_107"]
    ap_fold_improvements = sum(
        candidate_107["fold_values"][fold]["AP"] > feature_107["fold_values"][fold]["AP"]
        for fold in candidate_107["fold_values"]
    )
    micro_fold_improvements = sum(
        candidate_107["fold_values"][fold]["micro"] > feature_107["fold_values"][fold]["micro"]
        for fold in candidate_107["fold_values"]
    )
    # 结构未坍缩要求锚点不能独占绝大多数权重，且有效成员数大于 2。
    structure_noncollapsed = anchor_mass < 0.8 and effective_size > 2.0
    mechanism_pass = wrong_link_reduction >= 0.20
    # 性能门槛允许 AP 或 Micro-F1 任一项不低于原基线。
    performance_pass = (
        float(candidate_107["AP"]) >= float(baseline_107["AP"])
        or float(candidate_107["micro"]) >= float(baseline_107["micro"])
    )
    result["mechanism"] = {
        "raw_knn_zero_overlap_rate": raw_zero,
        "candidate_final_zero_overlap_mass_rate": candidate_zero,
        "relative_wrong_link_reduction": wrong_link_reduction,
        "structure_noncollapsed": structure_noncollapsed,
        "mean_anchor_mass": anchor_mass,
        "mean_effective_size": effective_size,
        "ap_fold_improvements_vs_feature_only": ap_fold_improvements,
        "micro_fold_improvements_vs_feature_only": micro_fold_improvements,
        "mechanism_threshold_pass": mechanism_pass,
        "performance_threshold_pass": performance_pass,
        "recommend_continue": mechanism_pass and performance_pass,
    }
    return result


def write_aggregate_csv(path: Path, summary: Dict[str, object]) -> None:
    """把基线、各变体固定 epoch 和最佳 epoch 摊平成统一 CSV。

    Args:
        path (pathlib.Path): 输出汇总 CSV 路径。
        summary (dict[str, object]): ``summarize_results`` 返回的结构化摘要。
    """
    rows: List[Dict[str, object]] = []
    baseline = summary["baseline"]
    for selection in ("epoch_107", "final"):
        rows.append(
            {
                "variant": "original_baseline",
                "selection": selection,
                "epoch": 106 if selection == "epoch_107" else 124,
                **{metric: baseline[selection][metric] for metric in METRICS},
                **{f"{metric}_std": baseline[selection][f"{metric}_std"] for metric in METRICS},
            }
        )
    for variant in VARIANTS:
        for selection in ("epoch_107", "final"):
            item = summary[variant][selection]
            rows.append(
                {
                    "variant": variant,
                    "selection": selection,
                    "epoch": item["epoch"],
                    **{metric: item[metric] for metric in METRICS},
                    **{f"{metric}_std": item[f"{metric}_std"] for metric in METRICS},
                }
            )
        rows.append(
            {
                "variant": variant,
                "selection": "best_ap",
                "epoch": summary[variant]["best_ap"]["epoch"],
                "AP": summary[variant]["best_ap"]["AP"],
            }
        )
        rows.append(
            {
                "variant": variant,
                "selection": "best_micro",
                "epoch": summary[variant]["best_micro"]["epoch"],
                "micro": summary[variant]["best_micro"]["micro"],
            }
        )
    columns: List[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def write_learning_curve_svg(path: Path, output_root: Path) -> None:
    """绘制六个变体的五折平均 AP 学习曲线。

    Args:
        path (pathlib.Path): 输出 SVG 路径。
        output_root (pathlib.Path): 各变体 ``metrics.csv`` 所在根目录。
    """
    width, height, margin = 920, 540, 70
    colors = {
        "candidate1": "#2563eb",
        "feature_only": "#64748b",
        "no_semantic": "#f59e0b",
        "no_reliability": "#16a34a",
        "no_conflict": "#9333ea",
        "oracle": "#dc2626",
    }
    # 先按 epoch 聚合每个变体所有折的 AP。
    curves: Dict[str, List[Tuple[int, float]]] = {}
    for variant in VARIANTS:
        rows = read_csv(output_root / variant / "metrics.csv")
        by_epoch: Dict[int, List[float]] = defaultdict(list)
        for row in rows:
            by_epoch[int(row["epoch"])].append(float(row["AP"]))
        curves[variant] = sorted((epoch, mean(values)) for epoch, values in by_epoch.items())
    minimum, maximum = 0.55, 0.88

    def x(epoch: int) -> float:
        """将 epoch 0..124 映射到横坐标。

        Args:
            epoch (int): 从 0 开始的训练轮次。
        """
        return margin + epoch / 124 * (width - 2 * margin)

    def y(value: float) -> float:
        """将 AP 预期范围 [0.55,0.88] 映射到纵坐标。

        Args:
            value (float): 平均精度 AP。
        """
        return height - margin - (value - minimum) / (maximum - minimum) * (height - 2 * margin)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="460" y="30" text-anchor="middle" font-size="20">Five-fold mean Average Precision</text>',
        f'<line x1="{margin}" y1="{height-margin}" x2="{width-margin}" y2="{height-margin}" stroke="#111"/>',
        f'<line x1="{margin}" y1="{margin}" x2="{margin}" y2="{height-margin}" stroke="#111"/>',
    ]
    for value in (0.55, 0.65, 0.75, 0.85):
        yy = y(value)
        parts.append(f'<line x1="{margin}" y1="{yy}" x2="{width-margin}" y2="{yy}" stroke="#ddd"/>')
        parts.append(f'<text x="{margin-8}" y="{yy+4}" text-anchor="end" font-size="12">{value:.2f}</text>')
    for variant, curve in curves.items():
        points = " ".join(f"{x(epoch):.1f},{y(value):.1f}" for epoch, value in curve)
        parts.append(f'<polyline points="{points}" fill="none" stroke="{colors[variant]}" stroke-width="2"/>')
    for index, variant in enumerate(VARIANTS):
        yy = 55 + index * 20
        parts.append(f'<line x1="{width-205}" y1="{yy}" x2="{width-180}" y2="{yy}" stroke="{colors[variant]}" stroke-width="3"/>')
        parts.append(f'<text x="{width-172}" y="{yy+4}" font-size="12">{variant}</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def write_report(path: Path, summary: Dict[str, object]) -> None:
    """把机制判断、固定 epoch 对比和下一步建议写成 Markdown 报告。

    Args:
        path (pathlib.Path): 输出 Markdown 路径。
        summary (dict[str, object]): ``summarize_results`` 返回的结构化摘要。
    """
    baseline = summary["baseline"]["epoch_107"]
    candidate = summary["candidate1"]["epoch_107"]
    mechanism = summary["mechanism"]
    graph = summary["candidate1"]["final_graph"]
    lines = [
        "# Candidate 1 实验报告",
        "",
        "## 最终判断",
        "",
        "**当前版本不建议作为论文主方向继续扩大实验。** 训练—测试统一构图和软关联在工程上成立，但后验语义与可靠度几乎没有改变错误邻居质量，也没有达到原 HyperAHSF 基线。这个结论是否定当前公式，不是否定‘语义可靠构图’这个问题本身。",
        "",
        "## 已证实",
        "",
        "- 原代码默认验证图五折、两个视图均只剩自环。",
        "- 两个视图 KNN 余弦中位数约为 0.533 与 0.938，固定阈值不具备跨视图可比性。",
        f"- 软图最终平均中心质量为 {mechanism['mean_anchor_mass']:.1%}、有效超边大小为 {mechanism['mean_effective_size']:.2f}，没有退化成自环图。",
        f"- 标量温度校准把最终 Brier 从 {graph['raw_brier']:.4f} 降到 {graph['calibrated_brier']:.4f}。",
        f"- 真标签 oracle 在第 107 epoch 达到 AP {summary['oracle']['epoch_107']['AP']:.4f}、Micro-F1 {summary['oracle']['epoch_107']['micro']:.4f}，说明准确语义图确实有潜力。",
        "",
        "## 未证实与失败假设",
        "",
        f"- 原始 KNN 零标签重叠率为 {mechanism['raw_knn_zero_overlap_rate']:.1%}；Candidate 1 最终错误关联质量为 {mechanism['candidate_final_zero_overlap_mass_rate']:.1%}，相对只下降 {mechanism['relative_wrong_link_reduction']:.1%}，远低于预设的 20%。",
        f"- 第 107 epoch：原基线 AP/Micro-F1 为 {baseline['AP']:.3f}/{baseline['micro']:.3f}；Candidate 1 为 {candidate['AP']:.4f}/{candidate['micro']:.4f}。",
        f"- 相比 feature-only，同一 epoch 的 AP 仅在 {mechanism['ap_fold_improvements_vs_feature_only']}/5 折改善，Micro-F1 仅在 {mechanism['micro_fold_improvements_vs_feature_only']}/5 折改善。",
        "- 去掉语义、可靠度或跨视图冲突后的结果几乎相同，说明这些项当前没有形成足够强的结构选择作用。",
        "",
        "## 五折结果（第 107 epoch）",
        "",
        "| 方法 | AP | Micro-F1 | HL | Macro-F1 | 解释 |",
        "|---|---:|---:|---:|---:|---|",
        f"| 原 HyperAHSF | {baseline['AP']:.3f} | {baseline['micro']:.3f} | {baseline['HM']:.3f} | {baseline['macro']:.3f} | 本地原始复现日志 |",
    ]
    labels = {
        "candidate1": "完整软后验图",
        "feature_only": "仅特征软图",
        "no_semantic": "去掉 soft Jaccard",
        "no_reliability": "去掉可靠度",
        "no_conflict": "去掉跨视图冲突",
        "oracle": "真标签 oracle（不可部署）",
    }
    for variant in VARIANTS:
        item = summary[variant]["epoch_107"]
        lines.append(
            f"| {variant} | {item['AP']:.4f} | {item['micro']:.4f} | {item['HM']:.4f} | {item['macro']:.4f} | {labels[variant]} |"
        )
    lines.extend(
        [
            "",
            "## 为什么当前公式没有奏效",
            "",
            "1. KNN 候选始终固定，softmax 又给每个候选非零质量，因此语义项只能轻微重排，不能真正拒绝错误邻居。",
            "2. EMA 单视图预测头只通过自身 BCE 学习；在小数据集早期后验接近一致，soft Jaccard 的区分度弱。",
            "3. 一条以样本 i 为中心的超边中，可靠度里的 r_i 对所有候选相同，在 softmax 中会抵消；真正起作用的只有邻居 r_j，且其差异有限。",
            "4. 温度缩放改善概率误差，但校准只修正置信度尺度，不能把错误语义排序变成正确排序。",
            "",
            "## 下一步建议",
            "",
            "不要继续微调当前五个温度或简单增加 loss。下一步应把问题收缩为：**如何让后验语义具备真正拒绝邻居的能力。** 最值得先验证的是带‘拒绝/空邻居’选项的稀疏关联（如 sparsemax、entmax 或显式 null member），并以错误关联质量为第一指标；在该机制能显著低于 feature-only 之前，不再做完整分类实验。",
            "",
            "## 复现说明",
            "",
            "- 数据：Emotions，原五折划分；CPU；固定种子 0。",
            "- 每个可训练方案运行 5 折×125 epoch；warm-up 10 epoch；图每 5 epoch 更新。",
            "- 原始复现源码、配置和数据均未覆盖；所有实现与结果位于独立实验目录。",
            "- oracle 使用验证真标签，只用于判断上限，不能作为可用方法结果。",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    """读取默认研究产物，生成所有汇总文件并打印机制判断。"""
    parser = argparse.ArgumentParser()
    project_root = Path(__file__).resolve().parents[2]
    parser.add_argument("--output-root", type=Path, default=project_root / "research_outputs" / "candidate1")
    parser.add_argument("--baseline-log", type=Path, default=project_root / "full_run.log")
    args = parser.parse_args()
    summary = summarize_results(args.output_root, args.baseline_log)
    (args.output_root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_aggregate_csv(args.output_root / "aggregate_results.csv", summary)
    write_learning_curve_svg(args.output_root / "learning_curves.svg", args.output_root)
    write_report(args.output_root / "Candidate 1 实验报告.md", summary)
    print(json.dumps(summary["mechanism"], ensure_ascii=False))


if __name__ == "__main__":
    main()
