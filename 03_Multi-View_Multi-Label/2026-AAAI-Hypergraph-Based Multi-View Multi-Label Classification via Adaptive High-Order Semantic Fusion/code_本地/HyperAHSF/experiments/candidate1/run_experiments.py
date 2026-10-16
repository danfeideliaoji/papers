"""在 Emotions 数据集上运行 Candidate 1 训练、消融与 oracle 对照实验。

脚本按变体和折循环训练学生模型，通过 EMA 教师周期性刷新软超图，并逐
epoch 记录分类指标与图质量证据。``oracle`` 使用真实标签构图，只能作为
不可部署的诊断上界，不能与无标签验证构图混为实际方法结果。
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from metrics import all_metrics
from utils import PrecomputedContrastiveLoss
from experiments.candidate1.candidate_model import (
    Candidate1Model,
    fit_temperatures,
    make_ema_teacher,
    update_ema,
)
from experiments.candidate1.diagnose_graphs import knn_evidence, load_split
from experiments.candidate1.soft_graph import (
    EPS,
    GraphBundle,
    SoftHypergraphBuilder,
    WeightedHypergraph,
)


VARIANTS = (
    "candidate1",
    "feature_only",
    "no_semantic",
    "no_reliability",
    "no_conflict",
    "oracle",
)


@dataclass(frozen=True)
class ExperimentConfig:
    """一次实验运行的不可变配置。

    ``warmup_epochs`` 内只按特征构图；之后按 ``graph_refresh`` 周期使用
    EMA 教师刷新软图。``folds`` 允许冒烟测试时只运行少数折。
    """
    project_root: Path
    output_root: Path
    folds: Tuple[int, ...] = tuple(range(5))
    epochs: int = 125
    warmup_epochs: int = 10
    graph_refresh: int = 5
    ema_decay: float = 0.99
    view_loss_weight: float = 0.2
    k: int = 6
    seed: int = 0
    device: str = "cpu"
    save_checkpoints: bool = True

    @property
    def data_root(self) -> Path:
        """返回 Emotions 五折 pickle 数据的固定根目录。"""
        return self.project_root / "Multi_View_dataset_k_fold_pickle" / "emotions"


def fix_seed(seed: int) -> None:
    """固定 Python、NumPy 和 PyTorch 随机状态，并启用确定性 cuDNN。

    Args:
        seed (int): 实验随机种子。
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    """收集所有行出现过的列名，并以 UTF-8 BOM 写出 CSV。

    Args:
        path (pathlib.Path): 输出 CSV 路径。
        rows (collections.abc.Sequence[dict[str, object]]): 待写出的字典行序列。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    columns: List[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _builder_for_variant(variant: str, k: int, warmup: bool) -> SoftHypergraphBuilder:
    """根据实验变体/预热阶段创建对应的软图构造器。

    ``feature_only`` 与 warmup 关闭语义、可靠性和冲突；其他 ``no_*``
    变体每次只关闭一个机制，用于测量该机制的独立贡献。

    Args:
        variant (str): 实验变体名称。
        k (int): 每个锚点的最大候选邻居数。
        warmup (bool): 是否处于仅按特征构图的预热期。
    """
    if warmup or variant == "feature_only":
        return SoftHypergraphBuilder(
            k=k, semantic_weight=0.0, use_reliability=False, use_conflict=False
        )
    if variant == "candidate1":
        return SoftHypergraphBuilder(k=k)
    if variant == "no_semantic":
        return SoftHypergraphBuilder(k=k, semantic_weight=0.0)
    if variant == "no_reliability":
        return SoftHypergraphBuilder(k=k, use_reliability=False)
    if variant == "no_conflict":
        return SoftHypergraphBuilder(k=k, use_conflict=False)
    raise ValueError(f"unsupported soft-graph variant: {variant}")


def build_label_oracle_bundle(
    features: Sequence[torch.Tensor], labels: torch.Tensor, k: int
) -> GraphBundle:
    """使用真实标签建立不可部署的归一化 oracle 图，仅供上界对照。

    每个锚点保留标签 Jaccard 不低于 0.9 的 KNN；一条超边内成员权重均分。

    Args:
        features (collections.abc.Sequence[torch.Tensor]): V 个视图的特征，每项 ``[N,D_v]``。
        labels (torch.Tensor): 二值真实标签 ``[N,C]``。
        k (int): 每个锚点的最大 KNN 候选数。
    """
    num_samples = labels.shape[0]
    view_graphs: List[WeightedHypergraph] = []
    # 各视图独立用真实标签过滤近邻。
    for tensor in features:
        neighbors, _, jaccard = knn_evidence(tensor, labels.long(), k)
        index_parts: List[torch.Tensor] = []
        weight_parts: List[torch.Tensor] = []
        for anchor in range(num_samples):
            selected = neighbors[anchor][jaccard[anchor] >= 0.9]
            # cat 沿现有的一维成员轴拼接：锚点 [1] 加选中邻居 [K'] 得到 [K'+1]。
            members = torch.cat(
                (torch.tensor([anchor], device=tensor.device), selected.to(tensor.device))
            )
            # full_like 创建与 members 同形状、类型、设备的张量，并全部填入锚点边编号。
            edge_ids = torch.full_like(members, anchor)
            # stack 新增第 0 维，把成员号和边号两个 [M] 向量堆为 [2,M]。
            index_parts.append(torch.stack((members, edge_ids), dim=0))
            # full 创建定值向量；这里令一条边内每个成员权重均为 1/成员数。
            weight_parts.append(torch.full((members.numel(),), 1.0 / members.numel(), device=tensor.device))
        # cat(dim=1) 沿成员列合并各锚点的 [2,M_h]；权重则沿唯一的一维首尾相接。
        edge_index = torch.cat(index_parts, dim=1)
        weights = torch.cat(weight_parts)
        view_graphs.append(
            WeightedHypergraph(
                edge_index=edge_index,
                incidence_weight=weights,
                num_nodes=num_samples,
                num_edges=num_samples,
                stats={
                    # 每个 part[0] 是一条边的锚点标量权重；stack 后成为 [N] 才能统一求均值。
                    "mean_anchor_mass": float(
                        torch.stack([part[0] for part in weight_parts]).mean().cpu()
                    ),
                    "mean_nonself_mass": float(
                        1.0 - torch.stack([part[0] for part in weight_parts]).mean().cpu()
                    ),
                },
            )
        )
    # torch.ones 创建全 1 张量；oracle 不建模视图可靠性，因此所有视图统一赋权为 1。
    reliability = torch.ones((num_samples, len(features)), device=labels.device)
    common = SoftHypergraphBuilder._build_common_graph(tuple(view_graphs), reliability)
    return GraphBundle(
        view_graphs=tuple(view_graphs),
        common_graph=common,
        reliability=reliability,
        consensus_probabilities=labels.float(),
        stats={"oracle": 1.0},
    )


@torch.no_grad()  # 教师只负责生成构图信号，此函数内不记录梯度。
def build_teacher_graph(
    variant: str,
    teacher: Candidate1Model,
    features: Sequence[torch.Tensor],
    labels: torch.Tensor,
    temperatures: torch.Tensor,
    k: int,
    warmup: bool,
) -> Tuple[GraphBundle, Tuple[torch.Tensor, ...], Tuple[torch.Tensor, ...]]:
    """用 EMA 教师输出构图，或为 oracle 变体建立真实标签图。

    Returns:
        图包、教师嵌入元组和教师 logits 元组。oracle 不需要后两项，返回
        与标签同形状的零张量占位。

    Args:
        variant (str): 构图实验变体名称。
        teacher (Candidate1Model): 当前 EMA 教师模型。
        features (collections.abc.Sequence[torch.Tensor]): 多视图特征序列。
        labels (torch.Tensor): 当前划分的标签 ``[N,C]``；仅 oracle 构图直接使用。
        temperatures (torch.Tensor): 各视图温度 ``[V]``。
        k (int): 每个锚点的最大候选数。
        warmup (bool): 是否关闭语义和可靠性机制。
    """
    if variant == "oracle":
        graph = build_label_oracle_bundle(features, labels, k)
        # zeros_like 创建与 labels 形状、类型、设备一致的全零占位张量。
        empty = tuple(torch.zeros_like(labels) for _ in features)
        return graph, empty, empty
    teacher.eval()
    embeddings, logits = teacher.encode_for_graph(features)
    # sigmoid 将各视图 logits 映射为 (0,1) 多标签概率。
    probabilities = tuple(torch.sigmoid(value) for value in logits)
    builder = _builder_for_variant(variant, k, warmup)
    graph = builder.build(features, embeddings, probabilities, temperatures)
    return graph, embeddings, logits


def _incidence_label_quality(graph: WeightedHypergraph, labels: torch.Tensor) -> Dict[str, float]:
    """利用真实标签离线评估局部图非自环成员关系的质量。

    只诊断前 N 条锚点超边；按成员权重统计零重叠/正重叠质量和加权真实
    Jaccard，不参与模型训练或构图决策。

    Args:
        graph (WeightedHypergraph): 待诊断的单视图加权超图。
        labels (torch.Tensor): 二值真实标签 ``[N,C]``。
    """
    node_idx, edge_idx = graph.edge_index
    valid = edge_idx < labels.shape[0]
    node_idx = node_idx[valid]
    edge_idx = edge_idx[valid]
    weights = graph.incidence_weight[valid]
    nonself = node_idx != edge_idx
    if not nonself.any():
        return {
            "nonself_mass": 0.0,
            "zero_overlap_mass_rate": 0.0,
            "positive_overlap_mass_rate": 0.0,
            "weighted_true_jaccard": 0.0,
        }
    node_idx = node_idx[nonself]
    edge_idx = edge_idx[nonself]
    weights = weights[nonself]
    anchor_labels = labels[edge_idx].long()
    member_labels = labels[node_idx].long()
    intersection = (anchor_labels & member_labels).sum(dim=1).float()
    union = (anchor_labels | member_labels).sum(dim=1).float()
    jaccard = intersection / (union + 1e-6)
    mass = weights.sum().clamp_min(EPS)
    return {
        "nonself_mass": float(mass.cpu()),
        "zero_overlap_mass_rate": float((weights * (jaccard == 0)).sum().cpu() / mass.cpu()),
        "positive_overlap_mass_rate": float((weights * (jaccard > 0)).sum().cpu() / mass.cpu()),
        "weighted_true_jaccard": float((weights * jaccard).sum().cpu() / mass.cpu()),
    }


def graph_evidence_row(
    graph_bundle: GraphBundle,
    labels: torch.Tensor,
    variant: str,
    fold: int,
    epoch: int,
    split: str,
    temperatures: torch.Tensor,
    teacher_logits: Sequence[torch.Tensor],
) -> Dict[str, object]:
    """把一次图刷新/验证对应的图质量与校准指标整理为 CSV 行。

    Args:
        graph_bundle (GraphBundle): 本次构图生成的局部图、公共图和可靠性。
        labels (torch.Tensor): 当前划分的二值标签 ``[N,C]``。
        variant (str): 实验变体名称。
        fold (int): 折编号。
        epoch (int): 从 0 开始的训练轮次。
        split (str): ``'train'`` 或 ``'val'``。
        temperatures (torch.Tensor): 各视图温度 ``[V]``。
        teacher_logits (collections.abc.Sequence[torch.Tensor]): 各视图教师 logits。
    """
    qualities = [_incidence_label_quality(graph, labels) for graph in graph_bundle.view_graphs]
    row: Dict[str, object] = {
        "variant": variant,
        "fold": fold,
        "epoch": epoch,
        "split": split,
        "temperature_mean": float(temperatures.mean().cpu()),
        "mean_reliability": float(graph_bundle.reliability.mean().cpu()),
    }
    for key in qualities[0]:
        row[key] = float(mean(item[key] for item in qualities))
    for key, value in graph_bundle.stats.items():
        row[key] = value
    # Brier 分数比较温度校准前后的教师概率质量。
    if teacher_logits and teacher_logits[0].numel():
        calibrated_brier: List[float] = []
        raw_brier: List[float] = []
        for view, logits in enumerate(teacher_logits):
            raw_brier.append(float(((torch.sigmoid(logits) - labels) ** 2).mean().cpu()))
            calibrated_brier.append(
                float(((torch.sigmoid(logits / temperatures[view]) - labels) ** 2).mean().cpu())
            )
        row["raw_brier"] = mean(raw_brier)
        row["calibrated_brier"] = mean(calibrated_brier)
    return row


def _classification_metrics(logits: torch.Tensor, labels: torch.Tensor, threshold: float) -> Dict[str, float]:
    """阈值化 logits 后调用主项目的八项多标签指标。

    Args:
        logits (torch.Tensor): 主分类器分数 ``[N,C]``。
        labels (torch.Tensor): 二值真实标签 ``[N,C]``。
        threshold (float): 将 sigmoid 概率转为 0/1 的阈值。
    """
    predictions = (torch.sigmoid(logits) > threshold).cpu().numpy().astype(np.int64)
    pairs = all_metrics(logits.detach().cpu().numpy(), predictions, labels.cpu().numpy())
    return {name: float(value) for name, value in pairs}


def _config_to_json(config: ExperimentConfig) -> Dict[str, object]:
    """把含 Path/Tuple 的 dataclass 转换为 JSON 可序列化字典。

    Args:
        config (ExperimentConfig): 一次实验的完整不可变配置。
    """
    result = asdict(config)
    result["project_root"] = str(config.project_root)
    result["output_root"] = str(config.output_root)
    result["folds"] = list(config.folds)
    return result


def run_variant(variant: str, config: ExperimentConfig) -> Dict[str, object]:
    """完整运行一个实验变体的全部指定折。

    每折独立初始化学生、EMA 教师、优化器、对比损失和温度。训练图按周期
    刷新；验证图每个 epoch 使用当前教师重新构造。返回逐 epoch 指标行和
    图证据行，同时在变体输出目录写出 CSV、配置和可选检查点。

    Args:
        variant (str): ``VARIANTS`` 中的实验变体名称。
        config (ExperimentConfig): 折、轮数、设备和输出目录等运行配置。
    """
    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of {VARIANTS}")
    with (config.project_root / "config.yaml").open("r", encoding="utf-8") as handle:
        params = yaml.safe_load(handle)["emotions"]
    device = torch.device(config.device)
    output_dir = config.output_root / variant
    output_dir.mkdir(parents=True, exist_ok=True)
    epoch_rows: List[Dict[str, object]] = []
    graph_rows: List[Dict[str, object]] = []

    for fold in config.folds:
        # 每折从相同随机种子开始，使折间初始化策略保持一致。
        fix_seed(config.seed)
        train_features, train_labels_long = load_split(config.data_root, fold, "train")
        val_features, val_labels_long = load_split(config.data_root, fold, "val")
        train_features = [tensor.to(device) for tensor in train_features]
        val_features = [tensor.to(device) for tensor in val_features]
        train_labels = train_labels_long.float().to(device)
        val_labels = val_labels_long.float().to(device)

        # 学生参与梯度更新；教师从学生深拷贝后仅通过 EMA 更新。
        model = Candidate1Model(params).to(device)
        teacher = make_ema_teacher(model).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=float(params["lr"]), weight_decay=float(params["weight_decay"])
        )
        # 对比正组只由训练标签预计算，不读取验证标签。
        contrastive = PrecomputedContrastiveLoss(train_labels_long, float(params["w_intra"]))
        temperatures = torch.ones(int(params["num_views"]), device=device)
        train_graph: Optional[GraphBundle] = None
        best_ap = -float("inf")

        for epoch in range(config.epochs):
            # 预热期禁用教师语义/可靠性，防止早期不稳定预测污染图结构。
            warmup = epoch < config.warmup_epochs
            # 图首次为空或达到刷新周期时才重建训练图，以节省构图成本。
            refresh = train_graph is None or epoch % config.graph_refresh == 0
            if refresh:
                with torch.no_grad():
                    _, teacher_logits = teacher.encode_for_graph(train_features)
                    # oracle 直接使用标签；其他变体在预热后拟合每视图温度。
                    if not warmup and variant != "oracle":
                        temperatures = fit_temperatures(teacher_logits, train_labels)
                    train_graph, _, graph_logits = build_teacher_graph(
                        variant,
                        teacher,
                        train_features,
                        train_labels,
                        temperatures,
                        config.k,
                        warmup,
                    )
                    graph_rows.append(
                        graph_evidence_row(
                            train_graph,
                            train_labels,
                            variant,
                            fold,
                            epoch,
                            "train",
                            temperatures,
                            graph_logits,
                        )
                    )

            # 学生训练目标 = 主分类 + 对比 + 加权视图辅助分类。
            model.train()
            optimizer.zero_grad(set_to_none=True)
            output = model(train_features, train_graph)
            classification_loss = F.binary_cross_entropy_with_logits(output.main_logits, train_labels)
            contrastive_loss = contrastive(output.contrastive_representation)
            # stack 把各视图标量 BCE 新增一维堆成 [V]，mean 后得到平均辅助损失。
            view_loss = torch.stack(
                [F.binary_cross_entropy_with_logits(logits, train_labels) for logits in output.view_logits]
            ).mean()
            loss = classification_loss + contrastive_loss + config.view_loss_weight * view_loss
            loss.backward()
            optimizer.step()
            # 优化器更新学生后，用新学生参数更新教师。
            update_ema(teacher, model, config.ema_decay)

            # 每个 epoch 在当前 EMA 教师生成的验证图上评估。
            model.eval()
            # 验证只做前向计算，不需要构建反向传播图。
            with torch.no_grad():
                val_graph, _, val_teacher_logits = build_teacher_graph(
                    variant,
                    teacher,
                    val_features,
                    val_labels,
                    temperatures,
                    config.k,
                    warmup,
                )
                val_output = model(val_features, val_graph)
                metrics = _classification_metrics(
                    val_output.main_logits, val_labels, float(params["confidence_bar"])
                )
                graph_rows.append(
                    graph_evidence_row(
                        val_graph,
                        val_labels,
                        variant,
                        fold,
                        epoch,
                        "val",
                        temperatures,
                        val_teacher_logits,
                    )
                )
            row: Dict[str, object] = {
                "variant": variant,
                "fold": fold,
                "epoch": epoch,
                "loss": float(loss.detach().cpu()),
                "classification_loss": float(classification_loss.detach().cpu()),
                "contrastive_loss": float(contrastive_loss.detach().cpu()),
                "view_loss": float(view_loss.detach().cpu()),
                **metrics,
            }
            epoch_rows.append(row)

            # 保存最佳 AP，以及原实验关注的第 107/最后 epoch 快照。
            if config.save_checkpoints:
                checkpoint = {
                    "variant": variant,
                    "fold": fold,
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "teacher": teacher.state_dict(),
                    "temperatures": temperatures.cpu(),
                    "params": params,
                    "experiment_config": _config_to_json(config),
                }
                if metrics["AP"] > best_ap:
                    best_ap = metrics["AP"]
                    # torch.save 使用 PyTorch 序列化格式保存参数、温度和配置字典。
                    torch.save(checkpoint, output_dir / f"fold{fold}_best_ap.pt")
                if epoch in {106, config.epochs - 1}:
                    torch.save(checkpoint, output_dir / f"fold{fold}_epoch{epoch + 1}.pt")

    # 一个变体的所有折完成后统一落盘，便于后续聚合脚本读取。
    write_csv(output_dir / "metrics.csv", epoch_rows)
    write_csv(output_dir / "graph_metrics.csv", graph_rows)
    (output_dir / "config.json").write_text(
        json.dumps(
            {"variant": variant, "params": params, "experiment": _config_to_json(config)},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return {"variant": variant, "epoch_rows": epoch_rows, "graph_rows": graph_rows}


def main() -> None:
    """解析实验变体与规模参数，依次运行并打印每个变体的产物行数。"""
    parser = argparse.ArgumentParser()
    project_root = Path(__file__).resolve().parents[2]
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=["candidate1"])
    parser.add_argument("--folds", nargs="+", type=int, default=list(range(5)))
    parser.add_argument("--epochs", type=int, default=125)
    parser.add_argument("--warmup-epochs", type=int, default=10)
    parser.add_argument("--graph-refresh", type=int, default=5)
    parser.add_argument("--output-root", type=Path, default=project_root / "research_outputs" / "candidate1")
    parser.add_argument("--no-checkpoints", action="store_true")
    args = parser.parse_args()
    config = ExperimentConfig(
        project_root=project_root,
        output_root=args.output_root,
        folds=tuple(args.folds),
        epochs=args.epochs,
        warmup_epochs=args.warmup_epochs,
        graph_refresh=args.graph_refresh,
        save_checkpoints=not args.no_checkpoints,
    )
    summary = []
    for variant in args.variants:
        result = run_variant(variant, config)
        summary.append(
            {
                "variant": variant,
                "epoch_rows": len(result["epoch_rows"]),
                "graph_rows": len(result["graph_rows"]),
            }
        )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
