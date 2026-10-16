"""只读诊断原始 HyperAHSF 构图规则。

本模块不训练或修改模型，而是在每折训练/验证数据上复现 KNN 候选、标签
Jaccard 过滤和特征余弦过滤，统计图密度、标签重叠、节点度与连通分量，
最终输出逐折 CSV、聚合 CSV、JSON、SVG 和 Markdown 诊断报告。
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn.functional as F


DEFAULT_THRESHOLDS = (99.0, 0.99, 0.95, 0.9, 0.8, 0.7, 0.5, 0.0, -1.0)


def load_split(data_root: Path, fold: int, split: str) -> Tuple[List[torch.Tensor], torch.Tensor]:
    """读取指定折和划分的全部视图特征及标签。

    视图文件按 ``features_view0.pickle`` 起连续编号，返回 CPU float32 特征
    列表和 long 类型标签矩阵。

    Args:
        data_root (pathlib.Path): 五折 pickle 数据根目录。
        fold (int): 折编号。
        split (str): ``'train'`` 或 ``'val'``。
    """
    directory = data_root / f"fold{fold}" / split
    features: List[torch.Tensor] = []
    view = 0
    while (directory / f"features_view{view}.pickle").exists():
        with (directory / f"features_view{view}.pickle").open("rb") as handle:
            # as_tensor 把 NumPy/pickle 数组转成张量，并统一为 float32；能复用内存时不会复制。
            features.append(torch.as_tensor(pickle.load(handle), dtype=torch.float32))
        view += 1
    if not features:
        raise FileNotFoundError(f"no view features found in {directory}")
    with (directory / "labels.pickle").open("rb") as handle:
        # torch.long 是 64 位整数类型，适合标签位运算和索引。
        labels = torch.as_tensor(pickle.load(handle), dtype=torch.long)
    return features, labels


def knn_evidence(
    features: torch.Tensor, labels: torch.Tensor, k: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """提取每个样本的 KNN、候选余弦相似度和真实标签 Jaccard。

    三个返回张量形状均为 ``[N,effective_k]``，用于比较“特征近邻”与
    “标签真正相近”是否一致。

    Args:
        features (torch.Tensor): 单视图特征 ``[N,D]``。
        labels (torch.Tensor): long 型二值标签 ``[N,C]``。
        k (int): 最大 KNN 候选数。
    """
    effective_k = min(k, max(0, features.shape[0] - 1))
    if effective_k == 0:
        empty = torch.empty((features.shape[0], 0))
        return empty.long(), empty, empty
    # 欧氏距离只负责确定候选邻居；相同候选随后同时评估余弦和标签重叠。
    # cdist 计算全部样本对欧氏距离：[N,D] 与 [N,D] 得到 [N,N]。
    distances = torch.cdist(features, features, p=2)
    distances.fill_diagonal_(float("inf"))
    # topk 在每行取最小的 K 项；.indices 只返回邻居下标 [N,K]。
    neighbors = torch.topk(distances, k=effective_k, dim=1, largest=False).indices
    # normalize 按行做 L2 归一化，归一化向量点积就是余弦相似度。
    normalized = F.normalize(features, p=2, dim=1)
    cosine = (normalized @ normalized.T).gather(1, neighbors)
    anchor_labels = labels[:, None, :].expand(-1, effective_k, -1)
    neighbor_labels = labels[neighbors]
    intersection = (anchor_labels & neighbor_labels).sum(dim=-1).float()
    union = (anchor_labels | neighbor_labels).sum(dim=-1).float()
    jaccard = intersection / (union + 1e-6)
    return neighbors, cosine, jaccard


def _component_count(mask: torch.Tensor, neighbors: torch.Tensor) -> int:
    """把保留的锚点—邻居关系视为无向边，用并查集统计连通分量数。

    Args:
        mask (torch.Tensor): bool 型候选保留掩码 ``[N,K]``。
        neighbors (torch.Tensor): long 型近邻下标 ``[N,K]``。
    """
    num_samples = mask.shape[0]
    parent = list(range(num_samples))

    def find(node: int) -> int:
        """查找并压缩节点所属集合的根。

        Args:
            node (int): 待查询节点编号。
        """
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(left: int, right: int) -> None:
        """合并一条被掩码保留的锚点—邻居关系。

        Args:
            left (int): 关系左端节点编号。
            right (int): 关系右端节点编号。
        """
        first, second = find(left), find(right)
        if first != second:
            parent[second] = first

    # where(mask) 分别返回 True 元素的行、列下标，即所有保留的锚点—邻居位置。
    anchors, positions = torch.where(mask)
    for anchor, position in zip(anchors.tolist(), positions.tolist()):
        union(anchor, int(neighbors[anchor, position]))
    return len({find(node) for node in range(num_samples)})


def summarize_mask(
    mask: torch.Tensor,
    neighbors: torch.Tensor,
    jaccard: torch.Tensor,
) -> Dict[str, object]:
    """汇总某一种候选保留掩码对应的超图统计量。

    掩码形状为 ``[N,k]``。每个锚点自带一条自环成员关系，掩码只描述
    非自环候选，因此平均超边大小为 ``1 + retained/N``。

    Args:
        mask (torch.Tensor): bool 型候选保留掩码 ``[N,K]``。
        neighbors (torch.Tensor): long 型候选邻居下标 ``[N,K]``。
        jaccard (torch.Tensor): 锚点与候选的真实标签 Jaccard ``[N,K]``。
    """
    num_samples, k = mask.shape
    retained = int(mask.sum())
    candidate_count = num_samples * k
    selected_jaccard = jaccard[mask]
    # 每个节点先计入自身锚点超边，再累加作为其他锚点邻居出现的次数。
    degrees = torch.ones(num_samples, dtype=torch.float32)
    # where 把二维布尔掩码转换为两组坐标向量。
    anchors, positions = torch.where(mask)
    if anchors.numel():
        # scatter_add_ 按邻居编号把 1 原地累加到 degrees，统计节点作为邻居出现的次数。
        degrees.scatter_add_(0, neighbors[anchors, positions], torch.ones_like(anchors, dtype=torch.float32))
    degree_mean = float(degrees.mean())
    degree_std = float(degrees.std(unbiased=False))

    def selected_stat(operation) -> object:
        """仅在至少保留一条候选时计算统计量，否则返回 None。

        Args:
            operation (collections.abc.Callable[[torch.Tensor], object]): 应用于已选 Jaccard 的统计函数。
        """
        return None if retained == 0 else float(operation(selected_jaccard))

    return {
        "num_samples": num_samples,
        "candidate_count": candidate_count,
        "retained_nonself": retained,
        "retention_rate": retained / candidate_count if candidate_count else 0.0,
        "nontrivial_hyperedges": int(mask.any(dim=1).sum()),
        "nontrivial_hyperedge_rate": float(mask.any(dim=1).float().mean()),
        "mean_hyperedge_size": 1.0 + retained / num_samples,
        "self_loop_incidence_ratio": num_samples / (num_samples + retained),
        "mean_true_jaccard": selected_stat(torch.mean),
        "zero_overlap_rate": selected_stat(lambda values: (values == 0).float().mean()),
        "positive_overlap_rate": selected_stat(lambda values: (values > 0).float().mean()),
        "exact_label_rate": selected_stat(lambda values: (values >= 0.999999).float().mean()),
        "node_degree_mean": degree_mean,
        "node_degree_std": degree_std,
        "node_degree_cv": degree_std / degree_mean if degree_mean else 0.0,
        "node_degree_min": float(degrees.min()),
        "node_degree_max": float(degrees.max()),
        "connected_components": _component_count(mask, neighbors),
    }


def collect_diagnostics(
    data_root: Path,
    folds: Sequence[int] = tuple(range(5)),
    thresholds: Sequence[float] = DEFAULT_THRESHOLDS,
    k: int = 6,
) -> Dict[str, object]:
    """遍历所有折、划分、视图和阈值，收集原始构图证据。

    每个视图记录三类规则：不筛选的 ``raw_knn``、使用真实标签的
    ``label_oracle``，以及多个阈值下的 ``feature_cosine``。

    Args:
        data_root (pathlib.Path): 五折 pickle 数据根目录。
        folds (collections.abc.Sequence[int]): 要诊断的折编号序列。
        thresholds (collections.abc.Sequence[float]): 要评估的余弦阈值序列。
        k (int): 每个样本的最大 KNN 候选数。
    """
    rows: List[Dict[str, object]] = []
    cosine_values: Dict[str, List[torch.Tensor]] = defaultdict(list)
    for fold in folds:
        for split in ("train", "val"):
            features, labels = load_split(data_root, fold, split)
            for view, tensor in enumerate(features):
                neighbors, cosine, jaccard = knn_evidence(tensor, labels, k)
                cosine_values[f"{split}_view_{view}"].append(cosine.reshape(-1))
                base = {"fold": fold, "split": split, "view": view}
                # 原始 KNN 是候选上界：所有 k 个邻居都保留。
                rows.append(
                    {
                        **base,
                        "rule": "raw_knn",
                        "threshold": None,
                        **summarize_mask(torch.ones_like(jaccard, dtype=torch.bool), neighbors, jaccard),
                    }
                )
                # 标签 oracle 仅用于诊断，不允许用于真实验证构图。
                rows.append(
                    {
                        **base,
                        "rule": "label_oracle",
                        "threshold": 0.9,
                        **summarize_mask(jaccard >= 0.9, neighbors, jaccard),
                    }
                )
                # 复现测试阶段余弦阈值过滤，特别包含配置中的异常阈值 99。
                for threshold in thresholds:
                    rows.append(
                        {
                            **base,
                            "rule": "feature_cosine",
                            "threshold": float(threshold),
                            **summarize_mask(cosine >= threshold, neighbors, jaccard),
                        }
                    )

    # 分训练/验证及视图统计候选余弦分布，判断统一阈值是否合适。
    quantiles: Dict[str, Dict[str, float]] = {}
    for key, chunks in cosine_values.items():
        # cat 沿现有的一维首尾拼接各折余弦值，不像 stack 那样新增“折”维。
        values = torch.cat(chunks)
        quantiles[key] = {
            # quantile(values,q) 返回 q 分位数，例如 q=0.5 即中位数。
            f"q{int(q * 100):02d}": float(torch.quantile(values, q))
            for q in (0.0, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0)
        }
    return {"rows": rows, "cosine_quantiles": quantiles, "k": k, "folds": list(folds)}


def aggregate_rows(rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    """按划分、视图、规则和阈值分组，对不同折的数值列取平均。

    Args:
        rows (collections.abc.Sequence[dict[str, object]]): 逐折诊断结果行。
    """
    groups: Dict[Tuple[object, ...], List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        key = (row["split"], row["view"], row["rule"], row["threshold"])
        groups[key].append(row)
    aggregated: List[Dict[str, object]] = []
    ignored = {"fold", "split", "view", "rule", "threshold"}
    for key, group in groups.items():
        item: Dict[str, object] = {
            "split": key[0],
            "view": key[1],
            "rule": key[2],
            "threshold": key[3],
        }
        for column in group[0]:
            if column in ignored:
                continue
            values = [row[column] for row in group if row[column] is not None]
            item[column] = None if not values else mean(values)
        aggregated.append(item)
    return sorted(
        aggregated,
        key=lambda row: (
            str(row["split"]), int(row["view"]), str(row["rule"]),
            -999.0 if row["threshold"] is None else -float(row["threshold"]),
        ),
    )


def write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    """以 UTF-8 BOM 编码写出字典行，便于 Excel 正确识别中文。

    Args:
        path (pathlib.Path): 输出 CSV 路径。
        rows (collections.abc.Sequence[dict[str, object]]): 待写出的字典行。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def write_svg(path: Path, aggregate: Sequence[Dict[str, object]]) -> None:
    """生成验证集阈值—保留率/标签重叠率的无依赖 SVG 折线图。

    Args:
        path (pathlib.Path): 输出 SVG 路径。
        aggregate (collections.abc.Sequence[dict[str, object]]): 聚合后的诊断行。
    """
    rows = [
        row
        for row in aggregate
        if row["split"] == "val" and row["rule"] == "feature_cosine"
        and -1.0 <= float(row["threshold"]) <= 0.99
    ]
    width, height, margin = 900, 520, 70
    colors = {0: "#2563eb", 1: "#dc2626"}

    def x_pos(threshold: float) -> float:
        """把余弦阈值 [-1,0.99] 线性映射到画布横坐标。

        Args:
            threshold (float): 余弦阈值。
        """
        return margin + (threshold + 1.0) / 1.99 * (width - 2 * margin)

    def y_pos(value: float) -> float:
        """把比例 [0,1] 映射到 SVG 自下而上的纵坐标。

        Args:
            value (float): 比例值。
        """
        return height - margin - value * (height - 2 * margin)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<line x1="{margin}" y1="{height-margin}" x2="{width-margin}" y2="{height-margin}" stroke="#111"/>',
        f'<line x1="{margin}" y1="{margin}" x2="{margin}" y2="{height-margin}" stroke="#111"/>',
        '<text x="450" y="28" text-anchor="middle" font-size="20">Validation cosine threshold: retention and label overlap</text>',
        f'<text x="450" y="505" text-anchor="middle" font-size="15">Cosine threshold</text>',
        f'<text x="18" y="260" text-anchor="middle" transform="rotate(-90 18 260)" font-size="15">Rate</text>',
    ]
    for tick in (-1.0, -0.5, 0.0, 0.5, 0.99):
        x = x_pos(tick)
        parts.append(f'<line x1="{x}" y1="{height-margin}" x2="{x}" y2="{height-margin+5}" stroke="#111"/>')
        parts.append(f'<text x="{x}" y="{height-margin+22}" text-anchor="middle" font-size="12">{tick:g}</text>')
    for tick in (0.0, 0.25, 0.5, 0.75, 1.0):
        y = y_pos(tick)
        parts.append(f'<line x1="{margin-5}" y1="{y}" x2="{width-margin}" y2="{y}" stroke="#ddd"/>')
        parts.append(f'<text x="{margin-10}" y="{y+4}" text-anchor="end" font-size="12">{tick:g}</text>')
    for view in (0, 1):
        selected = sorted((row for row in rows if row["view"] == view), key=lambda row: float(row["threshold"]))
        retention = " ".join(
            f'{x_pos(float(row["threshold"])):.1f},{y_pos(float(row["retention_rate"])):.1f}'
            for row in selected
        )
        overlap = " ".join(
            f'{x_pos(float(row["threshold"])):.1f},{y_pos(float(row["positive_overlap_rate"] or 0.0)):.1f}'
            for row in selected
        )
        color = colors[view]
        parts.append(f'<polyline points="{retention}" fill="none" stroke="{color}" stroke-width="3"/>')
        parts.append(f'<polyline points="{overlap}" fill="none" stroke="{color}" stroke-width="2" stroke-dasharray="7 5"/>')
        parts.append(f'<text x="{width-margin-145}" y="{55+view*22}" fill="{color}" font-size="13">View {view}: solid retention, dashed overlap</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def write_report(
    path: Path,
    aggregate: Sequence[Dict[str, object]],
    quantiles: Dict[str, Dict[str, float]],
) -> None:
    """根据聚合诊断量生成面向研究判断的 Markdown 报告。

    Args:
        path (pathlib.Path): 输出 Markdown 路径。
        aggregate (collections.abc.Sequence[dict[str, object]]): 聚合后的诊断行。
        quantiles (dict[str, dict[str, float]]): 分划分、视图保存的余弦分位数。
    """
    val_default = [
        row for row in aggregate
        if row["split"] == "val" and row["rule"] == "feature_cosine" and row["threshold"] == 99.0
    ]
    val_raw = [
        row for row in aggregate if row["split"] == "val" and row["rule"] == "raw_knn"
    ]
    val_oracle = [
        row for row in aggregate if row["split"] == "val" and row["rule"] == "label_oracle"
    ]
    raw_overlap = mean(float(row["positive_overlap_rate"]) for row in val_raw)
    raw_zero = mean(float(row["zero_overlap_rate"]) for row in val_raw)
    oracle_retain = mean(float(row["retention_rate"]) for row in val_oracle)
    lines = [
        "# HyperAHSF 原始构图诊断",
        "",
        "## 结论",
        "",
        f"- 默认 `test_thr=99` 在两个视图、五折验证集中保留的非自环关联均为 0，自环关联占比为 {mean(float(row['self_loop_incidence_ratio']) for row in val_default):.1%}。",
        f"- 原始特征 KNN 的验证邻居中，{raw_overlap:.1%} 至少共享一个标签，{raw_zero:.1%} 完全没有标签重叠。",
        f"- 标签 Jaccard≥0.9 的 oracle 仅保留约 {oracle_retain:.1%} 的候选，说明原训练图实际上要求近似完全相同的标签集合。",
        f"- 两个视图的验证 KNN 余弦中位数分别为 {quantiles['val_view_0']['q50']:.3f} 和 {quantiles['val_view_1']['q50']:.3f}；同一个固定阈值会产生明显不同的图密度。",
        "",
        "## 可证伪判断",
        "",
        "诊断支持“训练—测试构图规则和图密度存在结构分布偏移”，但它本身不能证明软后验图一定提高分类。Candidate 1 必须同时降低错误关联质量并保持有效非自环传播，才算机制成立。",
        "",
        "## 产物",
        "",
        "- `diagnostics_rows.csv`：逐折、逐视图原始结果；",
        "- `diagnostics_aggregate.csv`：五折平均；",
        "- `diagnostics.json`：完整机器可读结果；",
        "- `threshold_diagnostic.svg`：阈值、保留率与标签重叠率关系。",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    """解析路径参数，运行全量诊断并写出全部研究产物。"""
    parser = argparse.ArgumentParser()
    project_root = Path(__file__).resolve().parents[2]
    parser.add_argument(
        "--data-root",
        type=Path,
        default=project_root / "Multi_View_dataset_k_fold_pickle" / "emotions",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_root / "research_outputs" / "candidate1" / "diagnostics",
    )
    args = parser.parse_args()
    result = collect_diagnostics(args.data_root)
    aggregate = aggregate_rows(result["rows"])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "diagnostics_rows.csv", result["rows"])
    write_csv(args.output_dir / "diagnostics_aggregate.csv", aggregate)
    (args.output_dir / "diagnostics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_svg(args.output_dir / "threshold_diagnostic.svg", aggregate)
    write_report(args.output_dir / "构图诊断报告.md", aggregate, result["cosine_quantiles"])
    print(json.dumps({"rows": len(result["rows"]), "output_dir": str(args.output_dir)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
