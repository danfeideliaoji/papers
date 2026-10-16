"""HyperAHSF 多视图超图神经网络。

模型包含两条并行表示路径：每个视图各自使用一张标签引导/特征引导的
局部超图提取视图特异表示；所有视图映射到统一维度后，再使用包含跨视图
同一样本连接的公共超图提取共享表示。底层 ``HG_messagepassing`` 完成一次
“节点→超边→节点”传播，``HyperGNN`` 负责堆叠多层传播。

张量约定：``N`` 为样本数，``V`` 为视图数，``C`` 为标签数；局部图节点
编号为 ``0..N-1``，公共图节点编号为 ``0..V*N-1``。``hyperedge_index``
形状为 ``[2, M]``，每列表示 ``[节点编号, 超边编号]``。
"""

import torch.nn as nn
import torch
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, Linear
from experiments.hgnn_ablation.tensor_ops import scatter_add, scatter_mean
from typing import Optional, Callable
from utils import construct_hyperedge

class Encoder(nn.Module):
    """由若干 ``Linear + ReLU`` 组成的前馈特征编码器。"""

    def __init__(self, input_dim, feature_dim_list): # feature_dim
        """构造编码器。

        Args:
            input_dim (int): 原始视图特征维度。
            feature_dim_list (list[int] | tuple[int, ...]): 每层输出维度；最终输出维度为最后一个元素。
        """
        super(Encoder, self).__init__()
        layers = []
        prev_dim = input_dim
        for dim in feature_dim_list:
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(nn.ReLU())
            prev_dim = dim
        self.encoder = nn.Sequential(*layers)

    def forward(self, x):
        """将 ``[N, input_dim]`` 编码为 ``[N, output_dim]``。

        Args:
            x (torch.Tensor): 一个视图的节点/样本特征矩阵 ``[N,input_dim]``。
        """
        y = self.encoder(x)
        return y

class HGMVMLC_Model(nn.Module):
    """HyperAHSF 主模型：联合学习视图特异表示与跨视图公共表示。"""

    def __init__(self,dataset, params, device):
        """根据训练折和数据集配置构造模型与训练超图。

        Args:
            dataset (MyDataset_k_fold): 当前训练折；包含多视图特征列表和真实标签。
            params (dict[str, object]): ``config.yaml`` 中当前数据集的配置字典。
            device (str | torch.device): 模型和超图所在的 CPU/GPU 设备。

        Notes:
            训练超图在初始化时使用真实标签构造；验证超图不能使用验证标签，
            因而延迟到 ``evaluate`` 首次调用时根据特征构造并缓存。
        """
        super(HGMVMLC_Model, self).__init__()
        self.device = device
        self.k = params['k']
        self.thr = params['thr']
        self.test_k = params['test_k']
        self.test_thr = params['test_thr']
        self.num_views = params['num_views']
        self.num_class = params['num_class']
        self.num_layers = params['num_layers']
        self.input_sizes = params['inputs_sizes']
        self.feature_dim_matrix = params['EDC_feature_matrix']

        # 每个视图一张局部训练超图，节点编号均为该视图内部的 0..N-1。
        self.View_hyperedge_index = self.construct_label_hyperedge_index(dataset.features, dataset.labels,self.k, self.thr)
        # 合并局部图，并增加“同一样本的不同视图表示”之间的跨视图超边。
        self.common_hyperedge_index = construct_hyperedge(dataset.features, self.View_hyperedge_index)
        # 验证图采用惰性构造：None 表示尚未根据验证特征建立/缓存。
        self.test_hyperedge_index = None  # 长度 V 的视图内验证超图列表。
        self.eval_hyperedge_index = None  # 合并后的全局验证超图。
        self._evaluation_graph_key = None
        # 保留原代码在同一模型各 HGNN 间共享激活的结构，但不能跨折共享参数。
        graph_activation = nn.PReLU().to(self.device)

        # encoders 生成视图特异分支输入；transforms 把各视图映射到公共维度。
        self.encoders = []
        self.transforms = []
        # 每个视图拥有独立的 HyperGNN，参数不共享。
        self.View_HGNNs = nn.ModuleList([HyperGNN(in_dim=self.feature_dim_matrix[v][-1],
                                                  edge_dim=params['specific_dim_matrix'][v][-1],
                                                  node_dim=params['specific_dim_matrix'][v][-1],
                                                  num_layers=params['num_layers'],
                                                  hyperedge_weight = None,
                                                  act=graph_activation
                                                  ).to(self.device)
                                         for v in range(params['num_views'])])

        # 分别实例化特异编码器和公共映射器。
        for v in range(self.num_views):
            self.encoders.append(Encoder(self.input_sizes[v],
                                         self.feature_dim_matrix[v]).to(self.device))
            self.transforms.append(Encoder(self.input_sizes[v], [params['hid_dim']]).to(self.device))
        # 转为 ModuleList 后，内部参数才能被 optimizer/state_dict 正确发现。
        self.encoders = nn.ModuleList(self.encoders)
        self.transforms = nn.ModuleList(self.transforms)
        # 公共 HGNN 在包含 V*N 个节点的统一超图上进行传播。
        self.common_HGNN = HyperGNN(in_dim=params['hid_dim'],
                                    edge_dim=params['hid_dim'],
                                    node_dim=params['hid_dim'],
                                    num_layers=params['num_layers'],
                                    hyperedge_weight = None,
                                    act=graph_activation
                                    ).to(self.device)
        # 分类器输入 = 各视图特异节点表示拼接 + 每个样本的跨视图超边表示。
        dim = sum([params['specific_dim_matrix'][v][-1]
                   for v in range(params['num_views'])]) + params['hid_dim']
        self.classifier = nn.Linear(dim, self.num_class).to(self.device)
        self.view_shared_mlp = nn.Sequential(nn.Linear(params['hid_dim'], params['hid_dim']),
                                             nn.BatchNorm1d(params['hid_dim']),
                                             nn.ReLU(),
                                             ).to(self.device)

    def forward(self, features):
        """使用训练超图执行前向传播。

        Args:
            features (list[torch.Tensor] | tuple[torch.Tensor, ...]): 长度为 ``V`` 的特征序列，第 ``v`` 项形状为 ``[N,D_v]``。

        Returns:
            logits: 多标签分类分数 ``[N,C]``。
            commons_view: 公共 HGNN 输出按视图拆分后的元组，每项 ``[N,H]``。
            common_edge_repr: 前 ``N`` 条跨视图超边的表示 ``[N,H]``。
        """
        common_representations = []
        specific_representations = []
        num_samples = features[0].shape[0]
        for v in range(self.num_views):
            input_view = features[v]
            # 特异分支：原始视图特征 → 编码器 → 当前视图局部 HGNN。
            embedding_view = self.encoders[v](input_view)
            num_edges_s = self.View_hyperedge_index[v][1].max() + 1 if len(self.View_hyperedge_index[v][1]) > 0 \
                else torch.tensor(0, dtype=torch.int64).to(self.device)
            assert num_edges_s > 0, num_edges_s
            n_s, _ = self.View_HGNNs[v](x=embedding_view,
                                        hyperedge_index=self.View_hyperedge_index[v],
                                        num_nodes=num_samples,
                                        num_edges=num_edges_s)
            specific_representations.append(n_s)
            # 公共分支：不同输入维度都映射到同一个 hid_dim。
            transform = self.transforms[v](input_view)
            common_representations.append(self.view_shared_mlp(transform))
        assert len(common_representations) == self.num_views, 'common_representations length mismatch'

        # torch.cat 沿“已有维度”拼接；dim=0 把 V 个 [N,H] 接成 [V*N,H]，不会新增维度。
        num_edges_c = self.common_hyperedge_index[1].max() + 1 if len(self.common_hyperedge_index[1]) > 0 \
            else torch.tensor(0, dtype=torch.int64).to(self.device)
        n_c, e_c = self.common_HGNN(x=torch.cat(common_representations, dim=0),
                                     hyperedge_index=self.common_hyperedge_index,
                                     num_nodes=num_samples * self.num_views,
                                     num_edges=num_edges_c)
        # torch.split 按 dim=0 每 N 行切一块，把 [V*N,H] 还原为 V 个 [N,H]。
        commons_view = torch.split(n_c, num_samples, dim=0)
        # construct_hyperedge 将前 N 条边固定为跨视图同样本超边。
        specific_representations.append(e_c[:num_samples])
        # dim=1 表示沿特征列拼接：若各分支为 [N,D_v]，结果为 [N,sum(D_v)]。
        logits = self.classifier(torch.cat(specific_representations, dim=1))
        specific_representations.pop()
        assert len(specific_representations) == self.num_views, 'specific_representations length mismatch'
        return logits, commons_view, e_c[:num_samples]

    def construct_test_hyperedge_index(self, features, k, thr):
        """仅根据验证/测试特征构造每个视图的局部超图。

        对每个样本先按欧氏距离选取 ``k`` 个最近邻，再用余弦相似度阈值
        ``thr`` 过滤。每个样本作为一条候选超边的锚点；没有保留邻居的
        单元素候选边会被删除，最后额外为每个节点添加一条自环超边。

        Args:
            features (list[torch.Tensor] | tuple[torch.Tensor, ...]): 多视图特征序列，第 ``v`` 项为 ``[N,D_v]``。
            k (int): 每个样本参与距离排序的最近邻数量。
            thr (float): 保留候选邻居所需的最小余弦相似度。

        Returns:
            长度为 ``V`` 的列表；每项是形状 ``[2,M_v]`` 的局部
            ``hyperedge_index``。
        """
        intra_top = []
        for i in range(self.num_views):
            features_i = features[i]
            num_samples = features_i.size(0)
            assert k > 0, "k must be a positive integer"
            # F.normalize(..., dim=1) 把每一行除以自身 L2 范数，使每个样本成为单位向量。
            data_norm = F.normalize(features_i, p=2, dim=1)
            # torch.cdist 计算两组行向量的两两距离；这里得到 [N,N] 欧氏距离矩阵。
            dist = torch.cdist(features_i, features_i, p=2)
            dist.fill_diagonal_(float('inf'))  # 排除样本自身。
            # torch.mm 是二维矩阵乘法；单位向量矩阵乘其转置得到 [N,N] 余弦相似度。
            cos_sim = torch.mm(data_norm, data_norm.T)
            # torch.topk 返回“值和下标”；largest=False 表示取每行最小的 k 项，这里只使用下标。
            _, topk_indices = torch.topk(dist, k=k, dim=1, largest=False)
            # torch.gather 按 topk_indices 从 cos_sim 每一行取值，输出仍是 [N,k]。
            neighbor_cos_sim = torch.gather(cos_sim, 1, topk_indices)
            invalid_mask = neighbor_cos_sim < thr
            topk_indices[invalid_mask] = -1
            topk_indices = topk_indices.to(torch.long)
            # 每个样本 i 定义一条候选超边：[锚点 i, 最多 k 个有效近邻]。
            # arange 生成 0..N-1；unsqueeze(1) 新增列维，把 [N] 变为 [N,1]。
            row_indices = torch.arange(num_samples, device=self.device).unsqueeze(1)
            # cat 沿已有的列维拼接，把锚点列 [N,1] 与邻居 [N,k] 合成 [N,k+1]。
            combined = torch.cat([row_indices, topk_indices], dim=1)
            all_nodes = combined.view(-1)
            # repeat_interleave 将每个边编号连续重复 k+1 次，与展平后的成员顺序对齐。
            hyperedge_ids = torch.arange(num_samples, device=self.device).repeat_interleave(k + 1)
            mask = (all_nodes != -1)
            all_nodes = all_nodes[mask]
            hyperedge_ids = hyperedge_ids[mask]
            # torch.stack 会“新增一个维度”；两个 [M] 向量堆成 [2,M] 的节点—超边索引。
            hyperedge_index = torch.stack([all_nodes, hyperedge_ids], dim=0)
            # 删除只剩锚点的候选边（成员数小于 2），并将边编号压缩为连续值。
            if hyperedge_index.size(1) > 0:
                # bincount 统计每个超边编号出现次数，即每条候选超边包含多少成员。
                counts = torch.bincount(hyperedge_index[1], minlength=num_samples)
                valid_mask = counts >= 2
                # where(condition)[0] 返回条件为 True 的位置，这里得到有效超边编号。
                valid_ids = torch.where(valid_mask)[0]
                # isin 逐元素判断成员边编号是否属于 valid_ids，生成成员过滤掩码。
                edge_mask = torch.isin(hyperedge_index[1], valid_ids)
                hyperedge_index = hyperedge_index[:, edge_mask]
                if hyperedge_index.size(1) > 0:
                    # unique 找出不同边号；inverse 给出每个原边号对应的新连续编号 0..H-1。
                    _, inverse = torch.unique(hyperedge_index[1], return_inverse=True)
                    hyperedge_index[1] = inverse
                else:
                    hyperedge_index = torch.zeros((2, 0), dtype=torch.long, device=self.device)
            else:
                hyperedge_index = torch.zeros((2, 0), dtype=torch.long, device=self.device)
            # 前 N 条边为单节点自环，保证每个节点至少属于一条超边。
            self_loop_nodes = torch.arange(num_samples, device=self.device)
            # stack 新增第 0 维：[节点号] 与 [同号超边号] 组成 [2,N] 自环成员索引。
            self_loop_edges = torch.stack([self_loop_nodes, self_loop_nodes], dim=0)
            if hyperedge_index.size(1) > 0:
                hyperedge_index[1] += num_samples
            # cat 不新增维度，沿成员列 dim=1 把自环关系与动态关系接在一起。
            hyperedge_index = torch.cat([self_loop_edges, hyperedge_index], dim=1)
            intra_top.append(hyperedge_index)
        return intra_top

    def construct_label_hyperedge_index(self, features, labels, k, thr):
        """利用训练标签为每个视图构造局部训练超图。

        先按当前视图特征的欧氏距离寻找 ``k`` 个近邻，再计算锚点与近邻
        多标签集合的 Jaccard 相似度，只保留相似度不低于 ``thr`` 的近邻。
        训练阶段允许使用真实标签；验证阶段必须改用
        ``construct_test_hyperedge_index``，避免标签泄漏。

        Args:
            features (list[torch.Tensor] | tuple[torch.Tensor, ...]): 多视图特征序列，第 ``v`` 项为 ``[N,D_v]``。
            labels (torch.Tensor): 二值多标签矩阵 ``[N,C]``。
            k (int): 每个样本参与距离排序的最近邻数量。
            thr (float): 保留候选邻居所需的最小标签 Jaccard 相似度。

        Returns:
            长度为 ``V`` 的局部超图列表，每项形状为 ``[2,M_v]``。
        """
        intra_top = []
        for i in range(self.num_views):
            features_i = features[i]
            num_samples = features_i.size(0)
            assert k > 0, "k must be a positive integer"
            # cdist 的两项均为 [N,D]，因此返回每对样本欧氏距离组成的 [N,N]。
            dists = torch.cdist(features_i, features_i, p=2)
            dists.fill_diagonal_(float('inf'))  # except self
            # topk(..., largest=False) 取每行最小的 k 个距离及位置；这里只保留位置。
            topk_indices = torch.topk(dists, k=k, dim=1, largest=False).indices
            # 3. 将锚点/邻居索引展平，以向量化方式计算标签 Jaccard。
            # torch.arange(num_samples) 生成样本编号 [0,1,...,N-1]，形状为 [N]。
            # unsqueeze(1) 在第 1 维插入一个维度，将 [N] 变成列向量 [N,1]。
            # expand(-1,k) 将每个样本编号扩展 k 份得到 [N,k]；-1 表示第 0 维仍保持 N。
            # expand 只是创建广播视图，不会像 repeat 那样真正复制 k 份底层数据。
            row_indices = torch.arange(num_samples, device=self.device).unsqueeze(1).expand(-1, k)
            # flatten() 将 [N,k] 按行展开成 [N*k]：同一锚点编号会连续出现 k 次。
            anchors = row_indices.flatten()
            # 同样把邻居下标 [N,k] 展平；neighbors[p] 始终是 anchors[p] 对应的那个邻居。
            neighbors = topk_indices.flatten()
            # anchor_labels 与 neighbor_labels 均为 [N*k,C]。
            anchor_labels = labels[anchors.long()]
            neighbor_labels = labels[neighbors.long()]
            intersection = (anchor_labels & neighbor_labels).sum(dim=1)
            union = (anchor_labels | neighbor_labels).sum(dim=1)
            # 并集为正整数时不加 epsilon，否则恰等于 thr 的邻居会被错误过滤。
            # 空并集保持原代码 Jaccard=0 的约定，并避免除零。
            jaccard = intersection / union.clamp_min(1)
            valid_mask = (jaccard >= thr) & (neighbors != -1)
            neighbors[~valid_mask] = -1  # 使用 -1 标记未通过标签阈值的邻居。
            # neighbors 此时是展平的一维邻居序列 [N*k]；其中未通过 Jaccard 阈值的位置已被改成 -1。
            # view(num_samples,k) 不改变元素顺序，只把它恢复为 [N,k]：第 i 行仍是样本 i 的 k 个候选邻居。
            topk_indices = neighbors.view(num_samples, k)
            # arange(num_samples) 生成锚点编号 [0,1,...,N-1]，形状为 [N]。
            # unsqueeze(1) 新增第 1 维，将其变成列向量 [N,1]，保证每行只有当前样本自己的编号。
            row_indices = torch.arange(num_samples, device=self.device).unsqueeze(1)
            # 随后 cat(dim=1) 把锚点列放在邻居前面，每行形成 [锚点,邻居1,...,邻居k]。
            combined = torch.cat([row_indices, topk_indices], dim=1)
            # 展平节点—超边成员关系，并移除 -1 占位项。
            all_nodes = combined.view(-1)
            hyperedge_ids = torch.arange(num_samples, device=self.device).repeat_interleave(k + 1)
            mask = (all_nodes != -1)
            all_nodes = all_nodes[mask]
            hyperedge_ids = hyperedge_ids[mask]
            # stack 新增行维，把两个 [M] 向量变为标准 [2,M] hyperedge_index。
            hyperedge_index = torch.stack([all_nodes, hyperedge_ids], dim=0)
            # 删除无有效邻居的单元素候选超边，并连续化剩余边编号。
            if hyperedge_index.size(1) > 0:
                # bincount 按超边编号统计成员数量。
                counts = torch.bincount(hyperedge_index[1], minlength=num_samples)
                valid_mask = counts >= 2
                valid_ids = torch.where(valid_mask)[0]
                edge_mask = torch.isin(hyperedge_index[1], valid_ids)
                hyperedge_index = hyperedge_index[:, edge_mask]
                if hyperedge_index.size(1) > 0:
                    # unique 的 inverse 映射可把删除边后的编号重新压缩为连续整数。
                    _, inverse = torch.unique(hyperedge_index[1], return_inverse=True)
                    hyperedge_index[1] = inverse
                else:
                    hyperedge_index = torch.zeros((2, 0), dtype=torch.long, device=self.device)
            else:
                hyperedge_index = torch.zeros((2, 0), dtype=torch.long, device=self.device)
            # 5. 为每个样本加入一条自环超边；动态边编号整体后移 N。
            self_loop_nodes = torch.arange(num_samples, device=self.device)
            # stack 将节点编号和对应自环边编号堆成 [2,N]。
            self_loop_edges = torch.stack([self_loop_nodes, self_loop_nodes], dim=0)
            # 让动态超边编号从 N 开始，避免与 0..N-1 的自环编号冲突。
            if hyperedge_index.size(1) > 0:
                hyperedge_index[1] += num_samples
            # 合并后，每个视图得到一张独立的局部训练超图。
            # cat 沿成员列合并两组 [2,*] 索引。
            hyperedge_index = torch.cat([self_loop_edges, hyperedge_index], dim=1)
            intra_top.append(hyperedge_index)
        return intra_top

    def evaluate(self, features):
        """使用不依赖真实标签的验证超图执行前向传播。

        首次调用时构造并缓存局部验证图和公共验证图；相同特征复用缓存。
        更换特征、原地修改特征或改变构图参数时自动重建缓存。
        返回值与 ``forward`` 相同。

        Args:
            features (list[torch.Tensor] | tuple[torch.Tensor, ...]): 验证/测试多视图特征序列，第 ``v`` 项为 ``[N,D_v]``。
        """
        # 惰性构图避免在模型初始化时要求提供验证特征。
        graph_key = (self.test_k, self.test_thr, tuple(
            (f.data_ptr(), tuple(f.shape), f._version, str(f.device), str(f.dtype))
            for f in features))
        if graph_key != self._evaluation_graph_key:
            self.test_hyperedge_index = None
            self.eval_hyperedge_index = None
            self._evaluation_graph_key = graph_key
        if self.test_hyperedge_index is None:
            self.test_hyperedge_index = self.construct_test_hyperedge_index(features, self.test_k, self.test_thr)
        if self.eval_hyperedge_index is None:
            self.eval_hyperedge_index = construct_hyperedge(features, self.test_hyperedge_index)
        common_representations = []
        specific_representations = []
        num_samples = features[0].shape[0]
        for v in range(self.num_views):
            input_view = features[v]
            embedding_view = self.encoders[v](input_view)
            num_edges_s = self.test_hyperedge_index[v][1].max() + 1 if len(self.test_hyperedge_index[v][1]) > 0 \
                else torch.tensor(0, dtype=torch.int64).to(self.device)
            assert num_edges_s > 0, num_edges_s
            n_s, _ = self.View_HGNNs[v](x=embedding_view,
                                        hyperedge_index=self.test_hyperedge_index[v],
                                        num_nodes=num_samples,
                                        num_edges=num_edges_s)
            specific_representations.append(n_s)
            transform = self.transforms[v](input_view)
            common_representations.append(self.view_shared_mlp(transform))
        assert len(common_representations) == self.num_views, 'common_representations length mismatch'

        num_edges_c = self.eval_hyperedge_index[1].max() + 1 if len(self.eval_hyperedge_index[1]) > 0 \
            else torch.tensor(0, dtype=torch.int64).to(self.device)
        # cat(dim=0) 把各视图样本行接成公共图的 [V*N,H] 节点矩阵。
        n_c, e_c = self.common_HGNN(x=torch.cat(common_representations, dim=0),
                                    hyperedge_index=self.eval_hyperedge_index,
                                    num_nodes=num_samples * self.num_views,
                                    num_edges=num_edges_c)
        # split 每 N 行切分一次，恢复为各视图的 [N,H] 表示。
        commons_view = torch.split(n_c, num_samples, dim=0)
        specific_representations.append(e_c[:num_samples])
        # cat(dim=1) 沿特征维融合 V 个特异表示和一个公共超边表示。
        logits = self.classifier(torch.cat(specific_representations, dim=1))
        specific_representations.pop()
        assert len(specific_representations) == self.num_views, 'specific_representations length mismatch'
        return logits, commons_view, e_c[:num_samples]

class HG_messagepassing(MessagePassing):
    """一层“节点→超边→节点”的二部图消息传播。

    在默认 ``row_norm=True``、``use_attention=False`` 时，节点到超边阶段
    对一条超边内变换后的节点特征取平均；超边到节点阶段再对一个节点关联
    的超边表示取平均。该类保留了注意力、残差、融合和超边权重接口，但
    当前前向路径中有多处未启用，相关位置均在注释中明确标出。
    """

    def __init__(self, in_dim: int, hid_dim: int, out_dim: int, hyperedge_weight, dropout: float = 0.0,
                 act: Optional[Callable] = None, bias: bool = True, row_norm: bool = True,
                 use_attention: bool = False, **kwargs):
        """配置一层超图消息传播。

        Args:
            in_dim (int): 输入节点维度。
            hid_dim (int): 节点聚合到超边前的隐藏维度（设计上的超边维度）。
            out_dim (int): 超边传播回节点后的目标维度。
            hyperedge_weight (torch.Tensor | None): 可选超边权重；当前 ``compute_norm`` 计算了对应
                值但没有应用到返回权重，因而默认传 ``None``。
            dropout (float): 预留参数，当前前向路径未使用。
            act (collections.abc.Callable): 激活函数或可调用 PyTorch 模块。
            bias (bool): 是否创建两方向偏置；当前调用均使用默认 True。
            row_norm (bool): True 使用行平均，False 使用对称度归一化。
            use_attention (bool): 是否对每条成员消息启用多头权重，默认关闭。
            **kwargs (object): 继续传给 PyG ``MessagePassing`` 的关键字参数。
        """
        # PyG 对发往同一目标的消息做求和；配合 norm 后可实现平均聚合。
        kwargs.setdefault('aggr', 'add')
        # hyperedge_index 第一行为源节点、第二行为目标超边。
        kwargs.setdefault('flow', 'source_to_target')
        super().__init__(node_dim=0, **kwargs)
        self.in_dim = in_dim
        self.hid_dim = hid_dim
        self.out_dim = out_dim
        self.hyperedge_weight = hyperedge_weight
        self.dropout = dropout
        self.act = nn.PReLU() if act is None else act
        self.row_norm = row_norm
        self.use_attention = use_attention
        self.mode = 'n2e'  # 'n2e'：节点到边；'e2n'：边到节点。
        # 真实执行的节点→超边线性变换：in_dim → hid_dim。
        self.lin_n2e = Linear(in_dim, hid_dim, bias=False, weight_initializer='glorot')
        # 设计上的超边→节点变换；当前 forward 中调用被注释，暂未生效。
        self.lin_e2n = Linear(hid_dim, out_dim, bias=False, weight_initializer='glorot')
        # 以下融合层和 BatchNorm 是可选结构，当前主路径未调用 fuse_features。
        self.fuse_e = Linear(hid_dim, hid_dim, weight_initializer='glorot')
        self.fuse_n = Linear(out_dim, out_dim, weight_initializer='glorot')
        # 注意力固定拆成 4 个头，因此 hid_dim 必须能被 4 整除。
        self.bn_e = nn.BatchNorm1d(hid_dim)
        self.bn_n = nn.BatchNorm1d(hid_dim)
        # 为被注释的残差路径准备维度投影层。
        self.num_heads = 4
        assert hid_dim % self.num_heads == 0, "hid_dim/num_heads"
        self.head_dim = hid_dim // self.num_heads
        if use_attention:
            self.att_n2e = nn.Parameter(torch.Tensor(self.num_heads, self.head_dim))
            self.att_e2n = nn.Parameter(torch.Tensor(self.num_heads, self.head_dim))
            nn.init.xavier_uniform_(self.att_n2e)
            nn.init.xavier_uniform_(self.att_e2n)
        # 线性层自身不带 bias，偏置作为独立可训练参数维护。
        self.res_e = Linear(in_dim, hid_dim)
        self.res_n = Linear(in_dim, out_dim)
        # optional
        if bias:
            self.bias_n2e = nn.Parameter(torch.Tensor(hid_dim))
            self.bias_e2n = nn.Parameter(torch.Tensor(out_dim))
            nn.init.zeros_(self.bias_n2e)
            nn.init.zeros_(self.bias_e2n)
        else:
            self.register_parameter('bias_n2e', None)
            self.register_parameter('bias_e2n', None)

    def reset_parameters(self):
        """重置当前层拥有的线性映射参数。"""
        self.fuse_e.reset_parameters()
        self.fuse_n.reset_parameters()
        self.res_e.reset_parameters()
        self.res_n.reset_parameters()
        self.lin_n2e.reset_parameters()
        self.lin_e2n.reset_parameters()

    def forward(self, x: torch.Tensor, hyperedge_index: torch.Tensor,
                num_nodes: Optional[int] = None, num_edges: Optional[int] = None):
        """完成一次节点→超边→节点传播。

        Args:
            x (torch.Tensor): 输入节点特征 ``[N,in_dim]``。
            hyperedge_index (torch.Tensor): long 型成员关系 ``[2,M]``；每列为 ``[node_id,edge_id]``。
            num_nodes (int | None): 节点数 N。
            num_edges (int | torch.Tensor | None): 超边数 H。

        Returns:
            n: 更新后的节点表示。当前由于 ``lin_e2n`` 未启用，实际末维
                保持为 ``hid_dim``，配置中 ``hid_dim == out_dim``。
            e: 当前代码返回由原始 ``x`` 均值得到的初始超边表示，而不是
                ``e_agg``；这是现有行为，注释不对其做修复。
        """
        if num_edges != 0:
            # 初始超边表示：对每条超边内的原始节点 x 取均值，[H,in_dim]。
            e = self.initialize_hyperedge_features(x, hyperedge_index, num_edges)
            x_res = x
            self.mode = 'n2e'
            # 节点先投影到超边隐藏维度：[N,in_dim] → [N,hid_dim]。
            x_trans = self.lin_n2e(x) + self.bias_n2e
            # 对每条超边中的 x_trans 做归一化求和，得到 [H,hid_dim]。
            e_agg = self.node_to_hyperedge(x_trans, hyperedge_index, e, num_nodes, num_edges)
            # 注意：当前激活的是初始 e，而非刚聚合得到的 e_agg；保持原实现。
            e = self.act(e)
            self.mode = 'e2n'
            # 为兼容已有 checkpoint，保留当前未接入前向路径的可选层参数。
            # 每个节点聚合其关联的全部超边表示，得到 [N,hid_dim]。
            n_agg = self.hyperedge_to_node(e_agg, hyperedge_index, x_res, num_edges, num_nodes)
            n = self.act(n_agg)
            return n, e
        else:
            # 空图退化为仅对节点做残差投影，不产生超边表示。
            return self.act(self.res_n(x)), None

    def initialize_hyperedge_features(self, x, hyperedge_index, num_edges):
        """按超边编号对原始节点特征取均值，生成初始超边表示。

        Args:
            x (torch.Tensor): 原始节点特征 ``[N,D]``。
            hyperedge_index (torch.Tensor): long 型成员关系 ``[2,M]``。
            num_edges (int | torch.Tensor): 超边数量 H。
        """
        node_idx, edge_idx = hyperedge_index
        # scatter_mean 按 edge_idx 分组，对属于同一超边的 x[node_idx] 求平均，输出 [H,D]。
        return scatter_mean(x[node_idx], edge_idx, dim=0, dim_size=num_edges)

    def node_to_hyperedge(self, x, hyperedge_index, e, num_nodes, num_edges):
        """沿 ``node_id → edge_id`` 方向传播并聚合节点消息。

        Args:
            x (torch.Tensor): 已变换节点特征 ``[N,hid_dim]``。
            hyperedge_index (torch.Tensor): long 型成员关系 ``[2,M]``。
            e (torch.Tensor): 目标超边占位特征 ``[H,D]``。
            num_nodes (int): 节点数量 N。
            num_edges (int | torch.Tensor): 超边数量 H。
        """
        self.mode = 'n2e'
        norm = self.compute_norm(hyperedge_index, num_nodes, num_edges)
        return self.propagate(
            hyperedge_index,
            # 二元组分别提供源节点特征与目标超边特征；当前 message 不使用 x_i。
            x=(x, e),
            norm=norm,
            size=(num_nodes, num_edges)
        )

    def hyperedge_to_node(self, e, hyperedge_index, x, num_edges, num_nodes):
        """翻转成员索引，沿 ``edge_id → node_id`` 方向传播超边消息。

        Args:
            e (torch.Tensor): 超边特征 ``[H,D]``。
            hyperedge_index (torch.Tensor): long 型成员关系 ``[2,M]``。
            x (torch.Tensor): 目标节点占位特征 ``[N,D_x]``。
            num_edges (int | torch.Tensor): 超边数量 H。
            num_nodes (int): 节点数量 N。
        """
        self.mode = 'e2n'
        norm = self.compute_norm(hyperedge_index, num_nodes, num_edges)
        return self.propagate(
            hyperedge_index.flip([0]),  # [node,edge] → [edge,node]
            x=(e, x),
            norm=norm,
            size=(num_edges, num_nodes)
        )

    def compute_norm(self, hyperedge_index, num_nodes, num_edges):
        """为每条节点—超边成员关系计算传播归一化系数。

        ``Dn[i]`` 是节点 i 关联的超边数，``De[h]`` 是超边 h 包含的节点数。
        行归一化时，n2e 使用 ``1/De``，e2n 使用 ``1/Dn``；关闭行归一化
        时，两方向都使用 ``1/sqrt(Dn*De)``。

        Args:
            hyperedge_index (torch.Tensor): long 型成员关系 ``[2,M]``。
            num_nodes (int): 节点数量 N。
            num_edges (int | torch.Tensor): 超边数量 H。
        """
        node_idx, edge_idx = hyperedge_index
        # ones_like 生成全 1 且形状/设备相同的张量；scatter_add 按编号分组求和，即统计度数。
        Dn = scatter_add(torch.ones_like(node_idx), node_idx, dim=0, dim_size=num_nodes)
        De = scatter_add(torch.ones_like(edge_idx), edge_idx, dim=0, dim_size=num_edges)
        if self.mode == 'n2e':
            if self.row_norm:
                return 1.0 / De[edge_idx]
            else:
                return 1.0 / torch.sqrt(Dn[node_idx] * De[edge_idx])
        else:
            if self.row_norm:
                return 1.0 / Dn[node_idx]
            else:
                return 1.0 / torch.sqrt(Dn[node_idx] * De[edge_idx])

    def fuse_features(self, new):
        """可选的激活、线性融合与批归一化路径（当前 forward 未调用）。

        Args:
            new (torch.Tensor): 待融合特征；n2e 模式为 ``[H,hid_dim]``，e2n 模式为 ``[N,out_dim]``。
        """
        if self.mode == 'n2e':
            new = self.act(new)
            fused = self.fuse_e(new)
            fused = self.bn_e(fused)
            return fused
        else:
            new = self.act(new)
            fused = self.fuse_n(new)
            fused = self.bn_n(fused)
            return fused

    def message(self, x_j: torch.Tensor, x_i: torch.Tensor, norm: torch.Tensor):
        """为每条成员关系生成从源 ``j`` 发往目标 ``i`` 的消息。

        默认直接返回 ``norm * x_j``；PyG 随后按目标编号求和。开启注意力
        时，将每条消息拆为 4 个头并对头维做 softmax 加权。

        Args:
            x_j (torch.Tensor): 每条成员关系复制后的源特征 ``[M,D]``。
            x_i (torch.Tensor): 每条成员关系对应的目标特征 ``[M,D_i]``；当前未使用。
            norm (torch.Tensor): 每条成员关系的归一化系数 ``[M]``。
        """
        if self.use_attention:
            # 根据当前传播方向选择对应的可训练头权重。
            if self.mode == 'n2e':
                att_weights = self.att_n2e  # [num_heads, head_dim]
            else:
                att_weights = self.att_e2n  # [num_heads, head_dim]
            # [M,hid_dim] → [M,num_heads,head_dim]。
            x_j_multihead = x_j.view(-1, self.num_heads, self.head_dim)  # [E, num_heads, head_dim]
            # 每条成员关系、每个头得到一个标量分数。
            # einsum 按字符串规则做乘法并对 d 维求和：每条消息、每个头得到一个分数 [E,heads]。
            att_scores = torch.einsum('ehd,hd->eh', x_j_multihead, att_weights)  # [E, num_heads]
            # 当前 softmax 沿“头”维进行，而不是沿同一目标的邻居维进行。
            # softmax 沿 dim=1 把每条消息的各头分数变成和为 1 的权重。
            att_weights_per_head = torch.softmax(att_scores, dim=1)  # [E, num_heads]
            # 用头权重缩放对应子空间，再恢复为完整 hid_dim。
            weighted = x_j_multihead * att_weights_per_head.unsqueeze(-1)  # [E, num_heads, head_dim]
            aggregated = weighted.view(-1, self.hid_dim)  # [E, hid_dim]
            return norm.view(-1, 1) * aggregated
        else:
            return norm.view(-1, 1) * x_j

class HyperGNN(nn.Module):
    """堆叠多层 ``HG_messagepassing`` 的超图神经网络。"""

    def __init__(self, in_dim, edge_dim, node_dim, num_layers, hyperedge_weight, act: Optional[Callable] = None):
        """构造指定深度的超图网络。

        第一层接收 ``in_dim``；后续层接收上一层的 ``node_dim``。每层共享
        同一张 ``hyperedge_index``，但拥有独立线性参数。

        Args:
            in_dim (int): 第一层输入节点特征维度。
            edge_dim (int): 节点聚合到超边后的隐藏维度。
            node_dim (int): 每层输出节点维度。
            num_layers (int): 消息传播层数。
            hyperedge_weight (torch.Tensor | None): 可选超边权重。
            act (collections.abc.Callable): 每层使用的激活函数。
        """
        super(HyperGNN, self).__init__()
        self.in_dim = in_dim
        self.edge_dim = edge_dim
        self.node_dim = node_dim
        self.num_layers = num_layers
        self.act = nn.PReLU() if act is None else act
        self.edge_embeddings = []  # 保存最近一次前向传播中每层返回的超边表示。
        self.layers = nn.ModuleList()
        if num_layers == 1:
            self.layers.append(HG_messagepassing(self.in_dim, self.edge_dim, self.node_dim,
                                                 hyperedge_weight, act=self.act))
        else:
            self.layers.append(HG_messagepassing(self.in_dim, self.edge_dim, self.node_dim,
                                                 hyperedge_weight, act=self.act))
            for _ in range(self.num_layers - 2):
                self.layers.append(HG_messagepassing(self.node_dim, self.edge_dim, self.node_dim,
                                                     hyperedge_weight, act=self.act))
            self.layers.append(HG_messagepassing(self.node_dim, self.edge_dim, self.node_dim,
                                                 hyperedge_weight, act=self.act))
        self.reset_parameters()

    def reset_parameters(self):
        """依次重置所有消息传播层。"""
        for layer in self.layers:
            layer.reset_parameters()

    def forward(self, x: torch.Tensor, hyperedge_index: torch.Tensor, num_nodes: int, num_edges: int):
        """在同一张超图上顺序执行 ``num_layers`` 层传播。

        Args:
            x (torch.Tensor): 输入节点特征 ``[N,in_dim]``。
            hyperedge_index (torch.Tensor): long 型成员关系 ``[2,M]``。
            num_nodes (int): 节点数量 N。
            num_edges (int | torch.Tensor): 超边数量 H。
        """
        self.edge_embeddings = []
        for i in range(self.num_layers):
            x, e = self.layers[i](x, hyperedge_index, num_nodes, num_edges)
            self.edge_embeddings.append(e)
        assert len(self.edge_embeddings) == self.num_layers
        return x, e

    def get_edge_embeddings(self):
        """返回最近一次前向传播保存的逐层超边表示列表。"""
        return self.edge_embeddings
