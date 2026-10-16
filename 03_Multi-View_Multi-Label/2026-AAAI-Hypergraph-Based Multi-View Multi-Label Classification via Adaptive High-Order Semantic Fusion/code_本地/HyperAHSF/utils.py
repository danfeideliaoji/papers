"""HyperAHSF 的超图合并与对比损失工具。

``construct_hyperedge`` 将多个视图的局部超图转换到统一的全局编号空间，
并为同一样本的不同视图表示建立跨视图超边；``PrecomputedContrastiveLoss``
则根据完全相同的多标签组合预先建立正样本组。
"""

import torch.nn as nn
import torch
import torch.nn.functional as F
from collections import defaultdict

def jaccard_unique_values(tensor):
    """打印二值标签矩阵中两两 Jaccard 相似度的全部不同取值。

    Args:
        tensor (torch.Tensor): 二值标签矩阵，形状为 ``[N, C]``。

    Notes:
        此函数用于探索阈值候选，不返回计算结果。
    """
    # torch.mm 是二维矩阵乘法；[N,C]×[C,N] 一次得到 [N,N] 的标签交集数。
    intersection = torch.mm(tensor, tensor.T)

    # 每个样本拥有的正标签数量。
    row_sum = tensor.sum(dim=1)

    # |A∪B| = |A| + |B| - |A∩B|，通过广播得到 [N, N] 矩阵。
    union = row_sum.unsqueeze(1) + row_sum.unsqueeze(0) - intersection

    # 两个样本都没有标签时 union=0，这里约定它们的相似度为 1。
    jaccard = torch.where(
        union != 0,
        intersection / union,
        torch.ones_like(intersection)
    )

    # torch.unique 去重并排序，只输出不同取值，便于观察相似度的离散分布。
    unique_values = torch.unique(jaccard)
    print(unique_values)

def construct_hyperedge(view_tensors, view_hyperedge_index):
    """把各视图局部超图合并为带跨视图连接的统一超图。

    Args:
        view_tensors (list[torch.Tensor] | tuple[torch.Tensor, ...]): 长度为 ``V`` 的特征序列。每项至少具有相同样本数
            ``N``；本函数只读取 ``V``、``N`` 和设备，不使用特征值。
        view_hyperedge_index (list[torch.Tensor] | tuple[torch.Tensor, ...]): 长度为 ``V`` 的局部超图序列。第 ``v`` 项形状
            为 ``[2, M_v]``，第一行是局部节点编号 ``0..N-1``，第二行是
            该视图内部的超边编号。

    Returns:
        全局 ``hyperedge_index``，形状为 ``[2, M]``。全局节点按视图连续
        编号为 ``v*N + sample_id``；前 ``N`` 条超边固定连接同一样本在
        不同视图中的节点，其余超边是偏移后的各视图局部超边。
    """
    V = len(view_tensors)
    N = view_tensors[0].size(0)
    device = view_tensors[0].device
    # 为每个样本创建一条跨视图超边：超边 i 包含所有视图中的样本 i。
    # torch.arange(V) 生成整数序列 0..V-1；在同一 device 创建可避免 CPU/GPU 混用。
    views = torch.arange(V, device=device)  # (V,)
    samples = torch.arange(N, device=device)  # (N,)
    cross_nodes = (views[:, None] * N + samples[None, :]).T.reshape(-1)  # (V*N,)
    # repeat_interleave 把每个样本号连续重复 V 次，与 cross_nodes 的展平顺序对齐。
    cross_edges = torch.repeat_interleave(torch.arange(N, device=device), V)
    # stack 新增第 0 维：两个 [V*N] 向量组成 [2,V*N]；不同于 cat，它不会首尾相接。
    cross_hyperedge = torch.stack([cross_nodes, cross_edges])  # (2, V*N)
    # 每个视图必须提供且只提供一张局部超图。
    if V != len(view_hyperedge_index):
        raise ValueError("View count doesn't match view_hyperedge_index length")
    # 统计每张局部超图的超边数，为后续全局编号偏移做准备。
    e_counts = []
    for v in range(V):
        h_edge = view_hyperedge_index[v]
        if h_edge.numel() == 0:
            e_counts.append(0)
        else:
            max_e = h_edge[1].max()
            e_counts.append(max_e.item() + 1)
    # 前 N 个编号留给跨视图超边；各视图局部超边依次排在其后。
    # torch.cumsum 计算前缀累加和，用于确定每个视图超边编号的起始偏移。
    prefix_sums = torch.cumsum(torch.tensor([0] + e_counts, device=device), dim=0)
    start_indices = N + prefix_sums[:-1]  # (V,)
    # 将局部节点/超边编号转换成互不冲突的全局编号。
    view_hyperedges = []
    for v in range(V):
        h_edge = view_hyperedge_index[v]
        if h_edge.numel() == 0:
            continue
        # 节点 v:N 范围偏移：local_node → local_node + v*N。
        nodes = h_edge[0] + v * N
        # 超边按 start_indices[v] 偏移。
        edges = h_edge[1] + start_indices[v]
        # stack 把全局节点号和全局边号堆为该视图的 [2,M_v] 索引。
        view_hyperedges.append(torch.stack([nodes, edges]))
    # 第一部分始终是跨视图超边，第二部分是所有视图内部超边。
    hyperedges = [cross_hyperedge]
    if view_hyperedges:
        # cat(dim=1) 沿成员列连接多个 [2,M_v]，维数仍为二维。
        hyperedges.append(torch.cat(view_hyperedges, dim=1))
    # 最后把跨视图边与全部局部边沿成员列合为一张 [2,M] 全局索引。
    hyperedge_index = torch.cat(hyperedges, dim=1)

    return hyperedge_index

class PrecomputedContrastiveLoss(nn.Module):
    """基于完全相同标签组合的高性能向量化原型对比损失。

    初始化时把拥有相同多热标签向量的样本归为一组，只保留至少包含两个
    样本的组。前向传播时，每组以组内归一化特征均值作为原型，拉近组内
    样本与原型的余弦相似度，并压低组外样本与该原型的平均相似度。
    使用向量化 scatter_add 与矩阵乘法实现，避免顺序循环与显存碎片化。
    """

    def __init__(self, Y, w_intra):
        """预计算有效标签组。

        Args:
            Y (torch.Tensor): 二值多标签矩阵 ``[N, C]``。
            w_intra (float): 对比损失在总损失中的缩放系数。
        """
        super().__init__()
        self.w_intra = w_intra

        # 1. 标签向量转为可哈希元组，以完整标签组合为键聚合样本。
        label_dict = defaultdict(list)
        for i, label_vec in enumerate(Y):
            key = tuple(label_vec.tolist())
            label_dict[key].append(i)

        # 2. 单样本组无法形成组内正对，因此只保留大小不少于 2 的组。
        valid_groups = [indices for indices in label_dict.values() if len(indices) >= 2]
        self.num_groups = len(valid_groups)

        # 3. 没有有效组时，前向传播将返回同设备上的标量 0。
        if self.num_groups == 0:
            print("Warning: No valid groups found (all groups have size < 2).")
            return

        group_sizes = [len(g) for g in valid_groups]
        self.register_buffer('group_sizes', torch.tensor(group_sizes, dtype=torch.float32))

        flat_indices = []
        flat_group_ids = []
        for g, indices in enumerate(valid_groups):
            flat_indices.extend(indices)
            flat_group_ids.extend([g] * len(indices))

        self.register_buffer('flat_indices', torch.tensor(flat_indices, dtype=torch.long))
        self.register_buffer('flat_group_ids', torch.tensor(flat_group_ids, dtype=torch.long))

    def forward(self, X):
        """计算所有有效标签组的平均原型对比损失。

        Args:
            X (torch.Tensor): 样本表示矩阵 ``[N, D]``。

        Returns:
            标量损失 ``w_intra * mean(group_loss)``。
        """
        if self.num_groups == 0:
            return X.sum() * 0.0

        N, D = X.size()
        G = self.num_groups
        X_norm = F.normalize(X, p=2, dim=1)

        group_sizes = self.group_sizes.to(device=X.device, dtype=X.dtype)
        flat_indices = self.flat_indices.to(X.device)
        flat_group_ids = self.flat_group_ids.to(X.device)

        # 向量化累加每组样本特征以计算组原型
        group_sums = X.new_zeros(G, D)
        group_sums.scatter_add_(0, flat_group_ids.unsqueeze(1).expand(-1, D), X_norm[flat_indices])
        prototypes = F.normalize(group_sums / group_sizes.unsqueeze(1), p=2, dim=1)

        # 一次矩阵乘法得到所有样本与所有原型的相似度矩阵 [N, G]
        S = torch.mm(X_norm, prototypes.t())
        all_sum = S.sum(dim=0)

        # 提取正对相似度并对每组求和
        pos_sims = S[flat_indices, flat_group_ids]
        pos_sum = X.new_zeros(G).scatter_add_(0, flat_group_ids, pos_sims)

        # 计算正样本拉近项与负样本推远项
        pos_term = -pos_sum / group_sizes
        num_neg = N - group_sizes
        # 与旧循环一致：组外没有样本时负项为零，避免 0/0 及 NaN 梯度。
        neg_term = torch.where(num_neg > 0,
                               (all_sum - pos_sum) / num_neg.clamp_min(1),
                               torch.zeros_like(all_sum))
        loss = self.w_intra * (pos_term + neg_term).mean()
        return loss
