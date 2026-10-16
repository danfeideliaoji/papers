"""Candidate 1 的稀疏、不确定性感知软超图构造。

软图不把“节点是否属于超边”限制为 0/1，而是为每条节点—超边成员关系
分配实值权重。权重综合特征/嵌入距离、教师标签语义重叠、视图预测熵和
跨视图冲突；随后分别构造每个视图的局部图与统一公共图。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import torch
from experiments.hgnn_ablation.tensor_ops import scatter_add


EPS = 1e-8


@dataclass(frozen=True)
class WeightedHypergraph:
    """一张不可变的稀疏加权超图。

    ``edge_index[:,m]`` 表示第 m 条成员关系，``incidence_weight[m]`` 是其
    权重；二者长度必须一致。
    """
    edge_index: torch.Tensor
    incidence_weight: torch.Tensor
    num_nodes: int
    num_edges: int
    stats: Dict[str, float]


@dataclass(frozen=True)
class GraphBundle:
    """一次构图产生的全部局部图、公共图和诊断量。"""
    view_graphs: Tuple[WeightedHypergraph, ...]
    common_graph: WeightedHypergraph
    reliability: torch.Tensor
    consensus_probabilities: torch.Tensor
    stats: Dict[str, float]

    @property
    def hyperedge_index(self) -> torch.Tensor:
        """兼容旧调用方式，直接暴露公共图成员索引。"""
        return self.common_graph.edge_index

    @property
    def incidence_weight(self) -> torch.Tensor:
        """兼容旧调用方式，直接暴露公共图成员权重。"""
        return self.common_graph.incidence_weight


def aggregate_node_to_edge(
    node_features: torch.Tensor,
    edge_index: torch.Tensor,
    incidence_weight: torch.Tensor,
    num_edges: int,
) -> torch.Tensor:
    """执行按成员权重行归一化的节点→超边聚合。

    对超边 h 计算 ``sum_i(w_ih*x_i) / sum_i(w_ih)``，输出 ``[H,D]``。

    Args:
        node_features (torch.Tensor): 节点特征 ``[N,D]``。
        edge_index (torch.Tensor): long 型成员关系 ``[2,M]``。
        incidence_weight (torch.Tensor): 每条成员关系的实值权重 ``[M]``。
        num_edges (int): 超边数量 H。
    """
    node_idx, edge_idx = edge_index
    weights = incidence_weight.to(dtype=node_features.dtype)
    # scatter_add 按 edge_idx 分组求和，得到每条超边的权重和 [H]；clamp_min 防止除零。
    degree = scatter_add(weights, edge_idx, dim=0, dim_size=num_edges).clamp_min(EPS)
    messages = node_features[node_idx] * weights.unsqueeze(1)
    # 对同一超边收到的 [D] 消息按 edge_idx 分组相加，输出 [H,D]。
    summed = scatter_add(messages, edge_idx, dim=0, dim_size=num_edges)
    return summed / degree.unsqueeze(1)


def aggregate_edge_to_node(
    edge_features: torch.Tensor,
    edge_index: torch.Tensor,
    incidence_weight: torch.Tensor,
    num_nodes: int,
) -> torch.Tensor:
    """执行按成员权重行归一化的超边→节点聚合。

    对节点 i 计算 ``sum_h(w_ih*e_h) / sum_h(w_ih)``，输出 ``[N,D]``。

    Args:
        edge_features (torch.Tensor): 超边特征 ``[H,D]``。
        edge_index (torch.Tensor): long 型成员关系 ``[2,M]``。
        incidence_weight (torch.Tensor): 每条成员关系的实值权重 ``[M]``。
        num_nodes (int): 节点数量 N。
    """
    node_idx, edge_idx = edge_index
    weights = incidence_weight.to(dtype=edge_features.dtype)
    # 此处改按 node_idx 分组，得到每个节点收到的总权重 [N]。
    degree = scatter_add(weights, node_idx, dim=0, dim_size=num_nodes).clamp_min(EPS)
    messages = edge_features[edge_idx] * weights.unsqueeze(1)
    # 按目标节点编号聚合超边消息，输出 [N,D]。
    summed = scatter_add(messages, node_idx, dim=0, dim_size=num_nodes)
    return summed / degree.unsqueeze(1)


def _bernoulli_entropy(probabilities: torch.Tensor) -> torch.Tensor:
    """计算每个样本跨标签平均的归一化 Bernoulli 熵，范围约为 [0,1]。

    Args:
        probabilities (torch.Tensor): 多标签概率矩阵 ``[N,C]``。
    """
    p = probabilities.clamp(EPS, 1.0 - EPS)
    return -(p * p.log() + (1.0 - p) * (1.0 - p).log()).mean(dim=1) / torch.log(
        torch.tensor(2.0, device=p.device, dtype=p.dtype)
    )


def _bernoulli_js(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    """计算两个多标签 Bernoulli 预测的逐样本 Jensen-Shannon 散度。

    Args:
        first (torch.Tensor): 第一组概率 ``[N,C]``。
        second (torch.Tensor): 第二组概率 ``[N,C]``。
    """
    p = first.clamp(EPS, 1.0 - EPS)
    q = second.clamp(EPS, 1.0 - EPS)
    midpoint = ((p + q) / 2.0).clamp(EPS, 1.0 - EPS)

    def kl(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """逐标签 Bernoulli KL 散度，不在标签维求平均。

        Args:
            source (torch.Tensor): 源 Bernoulli 概率 ``[N,C]``。
            target (torch.Tensor): 目标 Bernoulli 概率 ``[N,C]``。
        """
        return source * (source.log() - target.log()) + (1.0 - source) * (
            (1.0 - source).log() - (1.0 - target).log()
        )

    return 0.5 * (kl(p, midpoint) + kl(q, midpoint)).mean(dim=1)


def _calibrate_probabilities(probabilities: torch.Tensor, temperature: torch.Tensor) -> torch.Tensor:
    """把概率还原为 logits，除以温度后再映射回概率。

    Args:
        probabilities (torch.Tensor): 待校准概率 ``[N,C]``。
        temperature (torch.Tensor): 当前视图的标量温度。
    """
    p = probabilities.clamp(1e-6, 1.0 - 1e-6)
    # torch.logit 是 sigmoid 的反函数，把 (0,1) 概率还原为任意实数 logits。
    logits = torch.logit(p)
    # sigmoid 再把温度缩放后的 logits 压回 (0,1) 概率。
    return torch.sigmoid(logits / temperature.clamp(0.5, 5.0))


def _soft_jaccard(anchor: torch.Tensor, candidates: torch.Tensor) -> torch.Tensor:
    """计算概率向量之间可微的软 Jaccard 相似度。

    Args:
        anchor (torch.Tensor): 广播后的锚点概率 ``[N,K+1,C]``。
        candidates (torch.Tensor): 候选成员概率 ``[N,K+1,C]``。
    """
    intersection = (anchor * candidates).sum(dim=-1)
    union = (anchor + candidates - anchor * candidates).sum(dim=-1)
    return intersection / union.clamp_min(EPS)


def _candidate_knn(features: torch.Tensor, k: int) -> torch.Tensor:
    """按欧氏距离返回每个样本最多 k 个非自身候选近邻。

    Args:
        features (torch.Tensor): 一个视图的原始特征 ``[N,D]``。
        k (int): 最大近邻数量。
    """
    num_samples = features.shape[0]
    if num_samples <= 1:
        return torch.empty((num_samples, 0), dtype=torch.long, device=features.device)
    effective_k = min(k, num_samples - 1)
    # cdist([N,D],[N,D]) 返回全部样本对的 [N,N] 欧氏距离矩阵。
    distances = torch.cdist(features, features, p=2)
    distances.fill_diagonal_(float("inf"))
    # topk(..., largest=False) 取每行距离最小的 K 项；.indices 只保留邻居下标 [N,K]。
    return torch.topk(distances, k=effective_k, dim=1, largest=False).indices


def _graph_stats(weights: torch.Tensor, width: int) -> Dict[str, float]:
    """将每条锚点超边的权重恢复为矩阵并计算稀疏性诊断统计。

    Args:
        weights (torch.Tensor): 展平后的全部成员权重 ``[N*width]``。
        width (int): 每条锚点超边的固定成员槽位数 ``K+1``。
    """
    if weights.numel() == 0:
        return {
            "mean_anchor_mass": 1.0,
            "mean_nonself_mass": 0.0,
            "mean_effective_size": 1.0,
            "min_weight": 0.0,
            "max_weight": 0.0,
        }
    matrix = weights.reshape(-1, width)
    entropy = -(matrix.clamp_min(EPS) * matrix.clamp_min(EPS).log()).sum(dim=1)
    return {
        "mean_anchor_mass": float(matrix[:, 0].mean().detach().cpu()),
        "mean_nonself_mass": float((1.0 - matrix[:, 0]).mean().detach().cpu()),
        "mean_effective_size": float(entropy.exp().mean().detach().cpu()),
        "min_weight": float(matrix.min().detach().cpu()),
        "max_weight": float(matrix.max().detach().cpu()),
    }


class SoftHypergraphBuilder:
    """在不使用验证真实标签的前提下构造局部和公共软超图。"""

    def __init__(
        self,
        k: int = 6,
        kappa: float = 1.0,
        tau_r: float = 1.0,
        tau_x: float = 1.0,
        tau_y: float = 1.0,
        tau_h: float = 1.0,
        semantic_weight: float = 1.0,
        use_reliability: bool = True,
        use_conflict: bool = True,
    ) -> None:
        """保存软图各温度、语义权重和消融开关。

        ``tau_x/tau_y/tau_h`` 分别控制距离、语义和最终 softmax 的尺度；
        ``kappa`` 控制冲突惩罚，``tau_r`` 控制可靠性衰减速度。

        Args:
            k (int): 每个锚点的最大 KNN 候选数。
            kappa (float): 跨视图冲突惩罚强度。
            tau_r (float): 可靠性指数衰减温度。
            tau_x (float): 几何距离项温度。
            tau_y (float): 语义重叠项温度。
            tau_h (float): 最终成员 softmax 温度。
            semantic_weight (float): 软 Jaccard 语义项系数。
            use_reliability (bool): 是否启用可靠性加权。
            use_conflict (bool): 是否将跨视图冲突纳入可靠性。
        """
        if k <= 0:
            raise ValueError("k must be positive")
        self.k = int(k)
        self.kappa = float(kappa)
        self.tau_r = float(tau_r)
        self.tau_x = float(tau_x)
        self.tau_y = float(tau_y)
        self.tau_h = float(tau_h)
        self.semantic_weight = float(semantic_weight)
        self.use_reliability = bool(use_reliability)
        self.use_conflict = bool(use_conflict)

    def build(
        self,
        features: Sequence[torch.Tensor],
        teacher_embeddings: Sequence[torch.Tensor],
        teacher_probabilities: Sequence[torch.Tensor],
        temperatures: torch.Tensor,
    ) -> GraphBundle:
        """根据教师嵌入与概率构造一次完整图包。

        Args:
            features (collections.abc.Sequence[torch.Tensor]): 原始多视图特征，每项 ``[N,D_v]``。
            teacher_embeddings (collections.abc.Sequence[torch.Tensor]): EMA 教师嵌入，每项 ``[N,H_v]``。
            teacher_probabilities (collections.abc.Sequence[torch.Tensor]): 各视图多标签概率，每项 ``[N,C]``。
            temperatures (torch.Tensor): 每个视图一个校准温度 ``[V]``。
        """
        if not (len(features) == len(teacher_embeddings) == len(teacher_probabilities)):
            raise ValueError("features, embeddings and probabilities must have equal view counts")
        if not features:
            raise ValueError("at least one view is required")
        num_views = len(features)
        num_samples = features[0].shape[0]
        if any(x.shape[0] != num_samples for x in features):
            raise ValueError("all views must contain the same number of samples")
        device = features[0].device
        dtype = teacher_probabilities[0].dtype
        # as_tensor 统一设备/类型；reshape(-1) 自动推断长度并展平成一维 [V]。
        temperatures = torch.as_tensor(temperatures, device=device, dtype=dtype).reshape(-1)
        if temperatures.numel() != num_views:
            raise ValueError("one temperature is required per view")

        # 1. 每个视图先做温度校准，降低过度自信对构图的影响。
        calibrated = tuple(
            _calibrate_probabilities(prob.to(device), temperatures[v])
            for v, prob in enumerate(teacher_probabilities)
        )
        # entropy[n,v] 衡量样本 n 在视图 v 上的预测不确定性。
        # stack 会新增视图维：V 个 [N] 熵向量沿 dim=1 堆成 [N,V]。
        entropy = torch.stack([_bernoulli_entropy(prob) for prob in calibrated], dim=1)

        # 2. agreement 使用一个视图与其他视图的平均 JS 一致性。
        if num_views == 1 or not self.use_conflict:
            agreement = torch.ones_like(entropy)
        else:
            agreement_columns: List[torch.Tensor] = []
            log_two = torch.log(torch.tensor(2.0, device=device, dtype=dtype))
            for v in range(num_views):
                others = [
                    1.0 - _bernoulli_js(calibrated[v], calibrated[r]) / log_two
                    for r in range(num_views)
                    if r != v
                ]
                # 将“当前视图与每个其他视图”的 [N] 一致性堆成 [N,V-1] 后按视图求均值。
                agreement_columns.append(torch.stack(others, dim=1).mean(dim=1))
            # 再新增视图维，把各列 [N] 组成最终 [N,V] 一致性矩阵。
            agreement = torch.stack(agreement_columns, dim=1).clamp(0.0, 1.0)

        # 3. 熵越高或跨视图分歧越大，可靠性越低。
        if self.use_reliability:
            reliability = torch.exp(
                -(entropy + self.kappa * (1.0 - agreement)) / max(self.tau_r, EPS)
            )
        else:
            reliability = torch.ones_like(entropy)

        # 4. 按逐样本、逐视图可靠性加权得到共识标签概率。
        # stack(calibrated, dim=1)：V 个 [N,C] 概率张量组成 [N,V,C]，新增的是视图维。
        stacked_probabilities = torch.stack(calibrated, dim=1)
        consensus = (stacked_probabilities * reliability.unsqueeze(-1)).sum(dim=1)
        consensus = consensus / reliability.sum(dim=1, keepdim=True).clamp_min(EPS)

        # 5. 各视图分别构造局部软图，再合成公共图。
        view_graphs: List[WeightedHypergraph] = []
        view_stats: Dict[str, float] = {}
        for v in range(num_views):
            graph = self._build_view_graph(
                features[v], teacher_embeddings[v], consensus, reliability[:, v]
            )
            view_graphs.append(graph)
            for key, value in graph.stats.items():
                view_stats[f"view_{v}_{key}"] = value

        common_graph = self._build_common_graph(tuple(view_graphs), reliability)
        stats = dict(view_stats)
        stats.update(
            {
                "mean_reliability": float(reliability.mean().detach().cpu()),
                "min_reliability": float(reliability.min().detach().cpu()),
                "mean_entropy": float(entropy.mean().detach().cpu()),
                "mean_agreement": float(agreement.mean().detach().cpu()),
            }
        )
        return GraphBundle(
            view_graphs=tuple(view_graphs),
            common_graph=common_graph,
            reliability=reliability,
            consensus_probabilities=consensus,
            stats=stats,
        )

    def _build_view_graph(
        self,
        features: torch.Tensor,
        embeddings: torch.Tensor,
        consensus: torch.Tensor,
        reliability: torch.Tensor,
    ) -> WeightedHypergraph:
        """为一个视图构造“每个样本一条锚点超边”的软图。

        每条边包含锚点及最多 k 个原始特征近邻；成员 logits 由教师嵌入
        距离、共识概率软 Jaccard 和成对可靠性组成，最终沿成员维 softmax。

        Args:
            features (torch.Tensor): 当前视图原始特征 ``[N,D_v]``。
            embeddings (torch.Tensor): 当前视图教师嵌入 ``[N,H_v]``。
            consensus (torch.Tensor): 跨视图共识多标签概率 ``[N,C]``。
            reliability (torch.Tensor): 当前视图逐样本可靠性 ``[N]``。
        """
        num_samples = features.shape[0]
        # 原始特征只负责限定稀疏候选集，避免构造完全图。
        candidates = _candidate_knn(features, self.k)
        effective_k = candidates.shape[1]
        # arange 生成锚点号 [N]；unsqueeze(1) 新增列维，变为 [N,1]。
        anchors = torch.arange(num_samples, device=features.device).unsqueeze(1)
        # cat 沿已有成员列拼接 [N,1] 和 [N,K]，得到 [N,K+1]，不会新增维度。
        members = torch.cat((anchors, candidates), dim=1)
        width = effective_k + 1

        # 计算锚点与各成员在教师嵌入空间中的平方距离。
        anchor_embeddings = embeddings.unsqueeze(1)
        member_embeddings = embeddings[members]
        squared_distance = ((anchor_embeddings - member_embeddings) ** 2).sum(dim=-1)
        # 用非零邻居距离中位数做尺度归一化，避免不同视图量纲差异。
        positive = squared_distance[:, 1:][squared_distance[:, 1:] > 0]
        scale = positive.median() if positive.numel() else torch.tensor(
            1.0, device=features.device, dtype=embeddings.dtype
        )
        normalized_distance = squared_distance / scale.clamp_min(EPS)

        anchor_probabilities = consensus.unsqueeze(1).expand(-1, width, -1)
        member_probabilities = consensus[members]
        # 语义重叠越高、两端越可靠，成员权重应越大。
        overlap = _soft_jaccard(anchor_probabilities, member_probabilities)
        pair_reliability = reliability.unsqueeze(1) * reliability[members]
        logits = -normalized_distance / max(self.tau_x, EPS)
        if self.semantic_weight != 0.0:
            logits = logits + self.semantic_weight * torch.log(overlap.clamp_min(EPS)) / max(
                self.tau_y, EPS
            )
        if self.use_reliability:
            logits = logits + 0.5 * torch.log(pair_reliability.clamp_min(EPS))
        # 每条锚点超边内部归一化，成员权重和为 1。
        # softmax 沿每条超边的成员维 dim=1 归一化，使一行成员权重之和为 1。
        weights = torch.softmax(logits / max(self.tau_h, EPS), dim=1)

        # repeat_interleave 将每个锚点边号连续重复 width 次，与 members 展平顺序一致。
        edge_ids = torch.arange(num_samples, device=features.device).repeat_interleave(width)
        # stack 新增第 0 维，把节点号 [M] 和边号 [M] 堆成 edge_index [2,M]。
        edge_index = torch.stack((members.reshape(-1), edge_ids), dim=0)
        flat_weights = weights.reshape(-1)
        return WeightedHypergraph(
            edge_index=edge_index,
            incidence_weight=flat_weights,
            num_nodes=num_samples,
            num_edges=num_samples,
            stats=_graph_stats(flat_weights, width),
        )

    @staticmethod
    def _build_common_graph(
        view_graphs: Tuple[WeightedHypergraph, ...], reliability: torch.Tensor
    ) -> WeightedHypergraph:
        """合并局部软图，并添加同一样本跨视图的可靠性加权超边。

        Args:
            view_graphs (tuple[WeightedHypergraph, ...]): V 张视图内局部软图。
            reliability (torch.Tensor): 每个样本、每个视图的可靠性 ``[N,V]``。
        """
        num_views = len(view_graphs)
        num_samples = view_graphs[0].num_nodes
        device = reliability.device

        # 对每个样本沿视图维归一化，作为跨视图超边的成员权重。
        normalized_reliability = reliability / reliability.sum(dim=1, keepdim=True).clamp_min(EPS)
        samples = torch.arange(num_samples, device=device)
        views = torch.arange(num_views, device=device)
        cross_nodes = (samples[:, None] + views[None, :] * num_samples).reshape(-1)
        cross_edges = samples.repeat_interleave(num_views)
        # stack 新增行维，形成跨视图成员关系索引 [2,N*V]。
        cross_index = torch.stack((cross_nodes, cross_edges), dim=0)
        cross_weights = normalized_reliability.reshape(-1)

        # 公共图前 N 条边是跨视图边，随后依次追加各视图局部软图。
        indices = [cross_index]
        weights = [cross_weights]
        edge_offset = num_samples
        for v, graph in enumerate(view_graphs):
            shifted = graph.edge_index.clone()
            # 局部节点编号偏移到视图 v 的全局节点区间。
            shifted[0] += v * num_samples
            shifted[1] += edge_offset
            indices.append(shifted)
            weights.append(graph.incidence_weight)
            edge_offset += graph.num_edges

        # cat 不新增维度：索引沿成员列连接，权重沿一维成员顺序连接。
        edge_index = torch.cat(indices, dim=1)
        incidence_weight = torch.cat(weights)
        return WeightedHypergraph(
            edge_index=edge_index,
            incidence_weight=incidence_weight,
            num_nodes=num_samples * num_views,
            num_edges=edge_offset,
            stats={
                "mean_cross_view_reliability_entropy": float(
                    (-(normalized_reliability.clamp_min(EPS) * normalized_reliability.clamp_min(EPS).log())
                     .sum(dim=1)
                     .mean())
                    .detach()
                    .cpu()
                )
            },
        )
