"""Candidate 1 加权超图模型、EMA 教师更新与温度校准工具。

该候选模型保留原 HyperAHSF 的“视图特异分支 + 公共分支”结构，但使用
``WeightedHypergraph`` 中的实值成员权重代替二值成员关系。视图预测头用于
为软构图提供教师概率，主分类器仍融合各视图特异节点表示和跨视图超边表示。
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import Linear

from experiments.candidate1.soft_graph import (
    GraphBundle,
    WeightedHypergraph,
    aggregate_edge_to_node,
    aggregate_node_to_edge,
)


class Encoder(nn.Module):
    """由 ``Linear + ReLU`` 堆叠而成的视图编码器。"""

    def __init__(self, input_dim: int, feature_dims: Sequence[int]) -> None:
        """按 ``feature_dims`` 顺序构造各层并记录在 ``nn.Sequential`` 中。

        Args:
            input_dim (int): 原始输入特征维度。
            feature_dims (collections.abc.Sequence[int]): 每层输出维度序列。
        """
        super().__init__()
        layers: List[nn.Module] = []
        previous = input_dim
        for dimension in feature_dims:
            layers.extend((nn.Linear(previous, dimension), nn.ReLU()))
            previous = dimension
        self.encoder = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """编码形状为 ``[N,input_dim]`` 的一个视图特征。

        Args:
            features (torch.Tensor): 一个视图的特征矩阵 ``[N,input_dim]``。
        """
        return self.encoder(features)


class WeightedHGMessagePassing(nn.Module):
    """使用实值成员权重的一层节点→超边→节点传播。

    传播顺序与原始 HyperAHSF 保持一致：原始节点均值得到返回的超边表示，
    线性变换后的节点均值得到传播用超边表示，再由超边聚合回节点。
    """

    def __init__(self, in_dim: int, edge_dim: int, node_dim: int) -> None:
        """创建节点到超边投影与空图时使用的节点残差投影。

        Args:
            in_dim (int): 输入节点维度。
            edge_dim (int): 节点聚合后的超边隐藏维度。
            node_dim (int): 返回节点表示维度。
        """
        super().__init__()
        self.lin_n2e = Linear(in_dim, edge_dim, bias=False, weight_initializer="glorot")
        self.bias_n2e = nn.Parameter(torch.zeros(edge_dim))
        self.res_n = Linear(in_dim, node_dim)
        self.act = nn.PReLU()

    def reset_parameters(self) -> None:
        """重置线性层并把节点到超边偏置清零。"""
        self.lin_n2e.reset_parameters()
        self.res_n.reset_parameters()
        nn.init.zeros_(self.bias_n2e)

    def forward(
        self, features: torch.Tensor, graph: WeightedHypergraph
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """执行一次加权双向聚合。

        Args:
            features (torch.Tensor): 节点特征 ``[N,D]``。
            graph (WeightedHypergraph): 包含 ``edge_index`` 和逐成员权重的稀疏加权超图。

        Returns:
            更新后的节点表示和由原始节点聚合得到的超边表示。
        """
        # 原始节点的加权均值作为对外返回的初始超边特征。
        initial_edges = aggregate_node_to_edge(
            features,
            graph.edge_index,
            graph.incidence_weight,
            graph.num_edges,
        )
        # 节点先映射到 edge_dim，再以相同成员权重聚合到超边。
        transformed = self.lin_n2e(features) + self.bias_n2e
        aggregated_edges = aggregate_node_to_edge(
            transformed,
            graph.edge_index,
            graph.incidence_weight,
            graph.num_edges,
        )
        # 每个节点对其关联超边做加权平均，完成超边到节点传播。
        aggregated_nodes = aggregate_edge_to_node(
            aggregated_edges,
            graph.edge_index,
            graph.incidence_weight,
            graph.num_nodes,
        )
        return self.act(aggregated_nodes), self.act(initial_edges)


class WeightedHyperGNN(nn.Module):
    """在同一张加权超图上堆叠多层 ``WeightedHGMessagePassing``。"""

    def __init__(
        self, in_dim: int, edge_dim: int, node_dim: int, num_layers: int
    ) -> None:
        """创建第一层输入投影及后续 ``node_dim → node_dim`` 传播层。

        Args:
            in_dim (int): 第一层输入节点维度。
            edge_dim (int): 超边隐藏维度。
            node_dim (int): 每层输出节点维度。
            num_layers (int): 消息传播层数。
        """
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be positive")
        layers: List[WeightedHGMessagePassing] = [
            WeightedHGMessagePassing(in_dim, edge_dim, node_dim)
        ]
        for _ in range(num_layers - 1):
            layers.append(WeightedHGMessagePassing(node_dim, edge_dim, node_dim))
        self.layers = nn.ModuleList(layers)

    def forward(
        self, features: torch.Tensor, graph: WeightedHypergraph
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """顺序运行所有传播层，返回最后一层节点和超边表示。

        Args:
            features (torch.Tensor): 输入节点特征 ``[N,in_dim]``。
            graph (WeightedHypergraph): 所有传播层共用的加权超图。
        """
        edge_features = None
        for layer in self.layers:
            features, edge_features = layer(features, graph)
        if edge_features is None:
            raise RuntimeError("weighted HGNN produced no edge features")
        return features, edge_features


@dataclass(frozen=True)
class CandidateModelOutput:
    """Candidate 1 一次前向传播的全部结构化输出。

    Attributes:
        main_logits: 主分类器输出 ``[N,C]``。
        common_views: 公共节点表示按视图拆分后的元组。
        contrastive_representation: 前 N 条跨视图超边表示 ``[N,H]``。
        view_logits: 各视图辅助预测头输出。
        embeddings: 各视图编码器输出，供教师软构图使用。
    """
    main_logits: torch.Tensor
    common_views: Tuple[torch.Tensor, ...]
    contrastive_representation: torch.Tensor
    view_logits: Tuple[torch.Tensor, ...]
    embeddings: Tuple[torch.Tensor, ...]


class Candidate1Model(nn.Module):
    """带软加权超图和视图辅助预测头的候选模型。"""

    def __init__(self, params: dict) -> None:
        """根据数据集配置创建编码器、局部/公共 HGNN 和分类头。

        Args:
            params (dict[str, object]): 当前数据集的模型与训练配置。
        """
        super().__init__()
        self.num_views = int(params["num_views"])
        self.num_classes = int(params["num_class"])
        self.hid_dim = int(params["hid_dim"])
        input_sizes = params["inputs_sizes"]
        feature_dims = params["EDC_feature_matrix"]
        specific_dims = params["specific_dim_matrix"]
        num_layers = int(params["num_layers"])

        # 特异编码器保留各视图自身的最终维度。
        self.encoders = nn.ModuleList(
            Encoder(input_sizes[v], feature_dims[v]) for v in range(self.num_views)
        )
        # 公共映射把所有视图统一到 hid_dim，才能按节点维拼接。
        self.transforms = nn.ModuleList(
            Encoder(input_sizes[v], [self.hid_dim]) for v in range(self.num_views)
        )
        # 辅助视图头产生每个视图的标签 logits，用于教师概率与校准。
        self.view_predictors = nn.ModuleList(
            nn.Linear(feature_dims[v][-1], self.num_classes) for v in range(self.num_views)
        )
        self.view_hgnns = nn.ModuleList(
            WeightedHyperGNN(
                feature_dims[v][-1],
                specific_dims[v][-1],
                specific_dims[v][-1],
                num_layers,
            )
            for v in range(self.num_views)
        )
        self.common_hgnn = WeightedHyperGNN(
            self.hid_dim, self.hid_dim, self.hid_dim, num_layers
        )
        self.view_shared_mlp = nn.Sequential(
            nn.Linear(self.hid_dim, self.hid_dim),
            nn.BatchNorm1d(self.hid_dim),
            nn.ReLU(),
        )
        classifier_dim = sum(x[-1] for x in specific_dims) + self.hid_dim
        self.classifier = nn.Linear(classifier_dim, self.num_classes)

    def encode_for_graph(
        self,
        features: Sequence[torch.Tensor],
        detach_for_view_heads: bool = False,
    ) -> Tuple[Tuple[torch.Tensor, ...], Tuple[torch.Tensor, ...]]:
        """生成构图嵌入和各视图辅助 logits。

        ``detach_for_view_heads=True`` 时，辅助分类损失只更新预测头，不会
        反向影响编码器；主模型前向采用该设置以隔离构图监督。

        Args:
            features (collections.abc.Sequence[torch.Tensor]): 长度 V 的特征序列，第 ``v`` 项为 ``[N,D_v]``。
            detach_for_view_heads (bool): 是否先截断编码器到辅助预测头的梯度。
        """
        embeddings = tuple(self.encoders[v](features[v]) for v in range(self.num_views))
        predictor_inputs = tuple(z.detach() if detach_for_view_heads else z for z in embeddings)
        view_logits = tuple(
            self.view_predictors[v](predictor_inputs[v]) for v in range(self.num_views)
        )
        return embeddings, view_logits

    def forward(
        self, features: Sequence[torch.Tensor], graph_bundle: GraphBundle
    ) -> CandidateModelOutput:
        """在给定软图包上运行特异分支和公共分支。

        Args:
            features (collections.abc.Sequence[torch.Tensor]): 长度 V 的原始视图特征序列。
            graph_bundle (GraphBundle): 与当前样本和视图对应的局部图及公共图。
        """
        embeddings, view_logits = self.encode_for_graph(
            features, detach_for_view_heads=True
        )
        specific_representations: List[torch.Tensor] = []
        common_representations: List[torch.Tensor] = []
        # 每个视图独立在其软局部图上提取特异节点表示。
        for v in range(self.num_views):
            node_features, _ = self.view_hgnns[v](embeddings[v], graph_bundle.view_graphs[v])
            specific_representations.append(node_features)
            transformed = self.transforms[v](features[v])
            common_representations.append(self.view_shared_mlp(transformed))

        # torch.cat(dim=0) 沿已有行维拼接，把 V 个 [N,H] 组成 [V*N,H]，不新增维度。
        common_nodes, common_edges = self.common_hgnn(
            torch.cat(common_representations, dim=0), graph_bundle.common_graph
        )
        num_samples = features[0].shape[0]
        # torch.split 每 N 行切一块；tuple(...) 将返回的张量块序列固定为元组。
        common_views = tuple(torch.split(common_nodes, num_samples, dim=0))
        # 公共图前 N 条边固定为同一样本跨视图超边。
        cross_view_edges = common_edges[:num_samples]
        # cat(dim=1) 沿特征列拼接各特异表示和公共超边表示，输出仍保持 N 行。
        main_logits = self.classifier(
            torch.cat((*specific_representations, cross_view_edges), dim=1)
        )
        return CandidateModelOutput(
            main_logits=main_logits,
            common_views=common_views,
            contrastive_representation=cross_view_edges,
            view_logits=view_logits,
            embeddings=embeddings,
        )


@torch.no_grad()  # 装饰整个函数：EMA 更新不建立反向传播计算图。
def update_ema(teacher: nn.Module, student: nn.Module, decay: float = 0.99) -> None:
    """用学生参数的指数移动平均更新教师模型。

    浮点/整数 buffer（例如 BatchNorm 统计量）直接从学生复制；教师始终保持
    eval 模式且不计算梯度。

    Args:
        teacher (torch.nn.Module): 仅通过 EMA 更新的教师模型。
        student (torch.nn.Module): 已经完成优化器更新的学生模型。
        decay (float): 旧教师参数的保留系数，范围为 ``[0,1)``。
    """
    if not 0.0 <= decay < 1.0:
        raise ValueError("EMA decay must be in [0, 1)")
    teacher_parameters = dict(teacher.named_parameters())
    for name, student_parameter in student.named_parameters():
        teacher_parameters[name].mul_(decay).add_(student_parameter.detach(), alpha=1.0 - decay)
    teacher_buffers = dict(teacher.named_buffers())
    for name, student_buffer in student.named_buffers():
        teacher_buffers[name].copy_(student_buffer.detach())
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)


def make_ema_teacher(student: Candidate1Model) -> Candidate1Model:
    """深拷贝学生模型并冻结全部参数，得到初始 EMA 教师。

    Args:
        student (Candidate1Model): 作为初始参数来源的学生模型。
    """
    teacher = copy.deepcopy(student)
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    return teacher


@torch.no_grad()  # 温度是网格搜索得到的常量，不需要计算梯度。
def fit_temperatures(
    view_logits: Sequence[torch.Tensor], labels: torch.Tensor
) -> torch.Tensor:
    """通过确定性网格搜索为每个视图拟合有界标量温度。

    候选温度覆盖 ``[0.5,5.0]`` 且包含 1，因此除浮点误差外，校准后的训练
    BCE 不会差于未校准结果。返回张量形状为 ``[V]``。

    Args:
        view_logits (collections.abc.Sequence[torch.Tensor]): V 个视图的 logits，每项形状 ``[N,C]``。
        labels (torch.Tensor): float 型二值训练标签 ``[N,C]``。
    """
    # torch.linspace 在 0.5 到 5.0（含端点）之间均匀生成 91 个候选温度。
    candidates = torch.linspace(0.5, 5.0, 91, device=labels.device, dtype=labels.dtype)
    fitted: List[torch.Tensor] = []
    for logits in view_logits:
        # stack 新增候选维，把 91 个标量 BCE 堆成 [91]，便于统一寻找最小值。
        losses = torch.stack(
            [F.binary_cross_entropy_with_logits(logits / temperature, labels) for temperature in candidates]
        )
        # argmin 返回最小损失的位置，再用该位置选出当前视图的最佳温度。
        fitted.append(candidates[torch.argmin(losses)])
    # 再次 stack，把 V 个标量温度堆成形状 [V] 的张量。
    return torch.stack(fitted)
