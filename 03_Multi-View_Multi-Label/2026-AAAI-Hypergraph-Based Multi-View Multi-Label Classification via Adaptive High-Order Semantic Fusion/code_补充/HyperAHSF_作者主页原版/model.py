import torch.nn as nn
import torch
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, Linear
from torch_scatter import scatter_add, scatter_mean
from typing import Optional, Callable
from utils import construct_hyperedge

class Encoder(nn.Module):
    def __init__(self, input_dim, feature_dim_list): # feature_dim
        super(Encoder, self).__init__()
        layers = []
        prev_dim = input_dim
        for dim in feature_dim_list:
            layers.append(nn.Linear(prev_dim, dim))
            # layers.append(nn.BatchNorm1d(dim))
            layers.append(nn.ReLU())
            # layers.append(nn.Dropout(0.2))
            prev_dim = dim
        # layers = layers[:-2]
        self.encoder = nn.Sequential(*layers)
        # self._initialize_weights()  # optional

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=nn.init.calculate_gain('relu'))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        y = self.encoder(x)
        return y

class HGMVMLC_Model(nn.Module):
    def __init__(self,dataset, params, device):
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

        # View_hyperedge_index is a list
        self.View_hyperedge_index = self.construct_label_hyperedge_index(dataset.features, dataset.labels,self.k, self.thr)
        self.common_hyperedge_index = construct_hyperedge(dataset.features, self.View_hyperedge_index)
        # self.hyperedge_weights = self.get_hyperedge_weights(self.View_hyperedge_index, dataset.labels)
        # self.common_hyperedge_weight = compute_hyperedge_weights(self.common_hyperedge_index,
        #                                torch.cat([dataset.labels for v in range(self.num_views)], dim=0))
        # print(f"train common hg: ", self.common_hyperedge_index)
        self.test_hyperedge_index = None
        self.eval_hyperedge_index = None

        # print(self.common_hyperedge_weight)

        # initial
        self.encoders = []
        self.transforms = []
        self.View_HGNNs = nn.ModuleList([HyperGNN(in_dim=self.feature_dim_matrix[v][-1],
                                                  edge_dim=params['specific_dim_matrix'][v][-1],
                                                  node_dim=params['specific_dim_matrix'][v][-1],
                                                  num_layers=params['num_layers'],
                                                  hyperedge_weight = None
                                                  ).to(self.device)
                                         for v in range(params['num_views'])])

        # Instantiation
        for v in range(self.num_views):
            self.encoders.append(Encoder(self.input_sizes[v],
                                         self.feature_dim_matrix[v]).to(self.device))
            self.transforms.append(Encoder(self.input_sizes[v], [params['hid_dim']]).to(self.device))
        self.encoders = nn.ModuleList(self.encoders)
        self.transforms = nn.ModuleList(self.transforms)
        self.common_HGNN = HyperGNN(in_dim=params['hid_dim'],
                                    edge_dim=params['hid_dim'],
                                    node_dim=params['hid_dim'],
                                    num_layers=params['num_layers'],
                                    hyperedge_weight = None
                                    ).to(self.device)
        dim = sum([params['specific_dim_matrix'][v][-1]
                   for v in range(params['num_views'])]) + params['hid_dim']
        self.classifier = nn.Linear(dim, self.num_class).to(self.device)
        self.view_shared_mlp = nn.Sequential(nn.Linear(params['hid_dim'], params['hid_dim']),
                                             nn.BatchNorm1d(params['hid_dim']),
                                             nn.ReLU(),
                                             # nn.Dropout(0.1)
                                             ).to(self.device)
        # self._initialize_weights()  # optional

    def _initialize_weights(self):
        nn.init.xavier_normal_(self.classifier.weight, gain=nn.init.calculate_gain('relu'))
        if self.classifier.bias is not None:
            nn.init.zeros_(self.classifier.bias)
        for layer in self.view_shared_mlp:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight, gain=nn.init.calculate_gain('relu'))
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def forward(self, features):
        # print('training mode')
        common_representations = []
        specific_representations = []
        num_samples = features[0].shape[0]
        for v in range(self.num_views):
            input_view = features[v]
            embedding_view = self.encoders[v](input_view)
            num_edges_s = self.View_hyperedge_index[v][1].max() + 1 if len(self.View_hyperedge_index[v][1]) > 0 \
                else torch.tensor(0, dtype=torch.int64).to(self.device)
            assert num_edges_s > 0, num_edges_s
            n_s, e_s = self.View_HGNNs[v](x=embedding_view,
                                        hyperedge_index=self.View_hyperedge_index[v],
                                        num_nodes=num_samples,
                                        num_edges=num_edges_s)
            specific_representations.append(n_s)
            transform = self.transforms[v](input_view)
            common_representations.append(self.view_shared_mlp(transform))
        assert len(common_representations) == self.num_views, 'common_representations length mismatch'

        num_edges_c = self.common_hyperedge_index[1].max() + 1 if len(self.common_hyperedge_index[1]) > 0 \
            else torch.tensor(0, dtype=torch.int64).to(self.device)
        n_c, e_c = self.common_HGNN(x=torch.cat(common_representations, dim=0),
                                     hyperedge_index=self.common_hyperedge_index,
                                     num_nodes=num_samples * self.num_views,
                                     num_edges=num_edges_c)
        commons_view = torch.split(n_c, num_samples, dim=0)
        common = torch.stack(commons_view).mean(dim=0)
        specific_representations.append(e_c[:num_samples])
        logits = self.classifier(torch.cat(specific_representations, dim=1))
        specific_representations.pop()
        assert len(specific_representations) == self.num_views, 'specific_representations length mismatch'
        return logits, commons_view, e_c[:num_samples]

    def construct_test_hyperedge_index(self, features, k, thr):
        intra_top = []
        for i in range(self.num_views):
            features_i = features[i]
            num_samples = features_i.size(0)
            assert k > 0, "k must be a positive integer"
            data_norm = F.normalize(features_i, p=2, dim=1)
            dist = torch.cdist(features_i, features_i, p=2)
            dist.fill_diagonal_(float('inf'))  # self
            # cosine
            cos_sim = torch.mm(data_norm, data_norm.T)
            # knn
            topk_dist, topk_indices = torch.topk(dist, k=k, dim=1, largest=False)
            # select
            neighbor_cos_sim = torch.gather(cos_sim, 1, topk_indices)
            invalid_mask = neighbor_cos_sim < thr
            topk_indices[invalid_mask] = -1
            topk_indices = topk_indices.to(torch.long)
            # print('view', i, topk_indices)
            # Hyperedge Index
            row_indices = torch.arange(num_samples, device=self.device).unsqueeze(1)
            combined = torch.cat([row_indices, topk_indices], dim=1)
            all_nodes = combined.view(-1)
            hyperedge_ids = torch.arange(num_samples, device=self.device).repeat_interleave(k + 1)
            mask = (all_nodes != -1)
            all_nodes = all_nodes[mask]
            hyperedge_ids = hyperedge_ids[mask]
            hyperedge_index = torch.stack([all_nodes, hyperedge_ids], dim=0)
            # filter
            if hyperedge_index.size(1) > 0:
                counts = torch.bincount(hyperedge_index[1], minlength=num_samples)
                valid_mask = counts >= 2
                valid_ids = torch.where(valid_mask)[0]
                edge_mask = torch.isin(hyperedge_index[1], valid_ids)
                hyperedge_index = hyperedge_index[:, edge_mask]
                if hyperedge_index.size(1) > 0:
                    _, inverse = torch.unique(hyperedge_index[1], return_inverse=True)
                    hyperedge_index[1] = inverse
                else:
                    hyperedge_index = torch.zeros((2, 0), dtype=torch.long, device=self.device)
            else:
                hyperedge_index = torch.zeros((2, 0), dtype=torch.long, device=self.device)
            # self loop
            self_loop_nodes = torch.arange(num_samples, device=self.device)
            self_loop_edges = torch.stack([self_loop_nodes, self_loop_nodes], dim=0)
            if hyperedge_index.size(1) > 0:
                hyperedge_index[1] += num_samples
            hyperedge_index = torch.cat([self_loop_edges, hyperedge_index], dim=1)
            intra_top.append(hyperedge_index)
            # print(hyperedge_index[:,-50:])
        return intra_top

    def construct_label_hyperedge_index(self, features, labels, k, thr):
        intra_top = []
        for i in range(self.num_views):
            features_i = features[i]
            num_samples = features_i.size(0)
            assert k > 0, "k must be a positive integer"
            # Euclidean distance
            dists = torch.cdist(features_i, features_i, p=2)
            dists.fill_diagonal_(float('inf'))  # except self
            # knn
            topk_values, topk_indices = torch.topk(dists, k=k, dim=1, largest=False)
            # Jaccard
            row_indices = torch.arange(num_samples, device=self.device).unsqueeze(1).expand(-1, k)
            anchors = row_indices.flatten()
            neighbors = topk_indices.flatten()
            # (anchor, neighbor)
            anchor_labels = labels[anchors.long()]
            neighbor_labels = labels[neighbors.long()]
            intersection = (anchor_labels & neighbor_labels).sum(dim=1)
            union = (anchor_labels | neighbor_labels).sum(dim=1)
            jaccard = intersection / (union + 1e-6)  # 0
            # print(jaccard.reshape(num_samples, -1))
            # print(jaccard.view(num_samples, k))
            valid_mask = (jaccard >= thr) & (neighbors != -1)
            neighbors[~valid_mask] = -1  # select
            # construct hyperedge
            topk_indices = neighbors.view(num_samples, k)
            # print(f"{i}: ", topk_indices)
            row_indices = torch.arange(num_samples, device=self.device).unsqueeze(1)
            combined = torch.cat([row_indices, topk_indices], dim=1)
            # filter
            all_nodes = combined.view(-1)
            hyperedge_ids = torch.arange(num_samples, device=self.device).repeat_interleave(k + 1)
            mask = (all_nodes != -1)
            all_nodes = all_nodes[mask]
            hyperedge_ids = hyperedge_ids[mask]
            hyperedge_index = torch.stack([all_nodes, hyperedge_ids], dim=0)
            # index
            if hyperedge_index.size(1) > 0:
                counts = torch.bincount(hyperedge_index[1], minlength=num_samples)
                valid_mask = counts >= 2
                valid_ids = torch.where(valid_mask)[0]
                edge_mask = torch.isin(hyperedge_index[1], valid_ids)
                hyperedge_index = hyperedge_index[:, edge_mask]
                if hyperedge_index.size(1) > 0:
                    _, inverse = torch.unique(hyperedge_index[1], return_inverse=True)
                    hyperedge_index[1] = inverse
                else:
                    hyperedge_index = torch.zeros((2, 0), dtype=torch.long, device=self.device)
            else:
                hyperedge_index = torch.zeros((2, 0), dtype=torch.long, device=self.device)
            # self loop
            self_loop_nodes = torch.arange(num_samples, device=self.device)
            self_loop_edges = torch.stack([self_loop_nodes, self_loop_nodes], dim=0)
            # adjust index
            if hyperedge_index.size(1) > 0:
                hyperedge_index[1] += num_samples
            # union
            hyperedge_index = torch.cat([self_loop_edges, hyperedge_index], dim=1)
            intra_top.append(hyperedge_index)
            # print(f"View {i} hyperedge index:", hyperedge_index)
        return intra_top

    def evaluate(self, features):
        if self.test_hyperedge_index is None:
            self.test_hyperedge_index = self.construct_test_hyperedge_index(features, self.test_k, self.test_thr)
        if self.eval_hyperedge_index is None:
            self.eval_hyperedge_index = construct_hyperedge(features, self.test_hyperedge_index)
            # print(f"val common hg: ", self.eval_hyperedge_index)
        common_representations = []
        specific_representations = []
        num_samples = features[0].shape[0]
        for v in range(self.num_views):
            input_view = features[v]
            embedding_view = self.encoders[v](input_view)
            num_edges_s = self.test_hyperedge_index[v][1].max() + 1 if len(self.test_hyperedge_index[v][1]) > 0 \
                else torch.tensor(0, dtype=torch.int64).to(self.device)
            assert num_edges_s > 0, num_edges_s
            n_s, e_s = self.View_HGNNs[v](x=embedding_view,
                                          hyperedge_index=self.test_hyperedge_index[v],
                                          num_nodes=num_samples,
                                          num_edges=num_edges_s)
            specific_representations.append(n_s)
            transform = self.transforms[v](input_view)
            common_representations.append(self.view_shared_mlp(transform))
        assert len(common_representations) == self.num_views, 'common_representations length mismatch'

        num_edges_c = self.eval_hyperedge_index[1].max() + 1 if len(self.eval_hyperedge_index[1]) > 0 \
            else torch.tensor(0, dtype=torch.int64).to(self.device)
        n_c, e_c = self.common_HGNN(x=torch.cat(common_representations, dim=0),
                                    hyperedge_index=self.eval_hyperedge_index,
                                    num_nodes=num_samples * self.num_views,
                                    num_edges=num_edges_c)
        commons_view = torch.split(n_c, num_samples, dim=0)
        common = torch.stack(commons_view).mean(dim=0)
        specific_representations.append(e_c[:num_samples])
        logits = self.classifier(torch.cat(specific_representations, dim=1))
        specific_representations.pop()
        assert len(specific_representations) == self.num_views, 'specific_representations length mismatch'
        return logits, commons_view, e_c[:num_samples]

class HG_messagepassing(MessagePassing):
    def __init__(self, in_dim: int, hid_dim: int, out_dim: int, hyperedge_weight, dropout: float = 0.0,
                 act: Callable = nn.PReLU(), bias: bool = True, row_norm: bool = True,
                 use_attention: bool = False, **kwargs):
        kwargs.setdefault('aggr', 'add')
        kwargs.setdefault('flow', 'source_to_target')
        super().__init__(node_dim=0, **kwargs)
        self.in_dim = in_dim
        self.hid_dim = hid_dim
        self.out_dim = out_dim
        self.hyperedge_weight = hyperedge_weight
        self.dropout = dropout
        self.act = act
        self.row_norm = row_norm
        self.use_attention = use_attention
        self.mode = 'n2e'
        self.lin_n2e = Linear(in_dim, hid_dim, bias=False, weight_initializer='glorot')
        self.lin_e2n = Linear(hid_dim, out_dim, bias=False, weight_initializer='glorot')
        # optional
        self.fuse_e = Linear(hid_dim, hid_dim, weight_initializer='glorot')
        self.fuse_n = Linear(out_dim, out_dim, weight_initializer='glorot')
        # optional
        self.bn_e = nn.BatchNorm1d(hid_dim)
        self.bn_n = nn.BatchNorm1d(hid_dim)
        # optional
        self.num_heads = 4
        assert hid_dim % self.num_heads == 0, "hid_dim/num_heads"
        self.head_dim = hid_dim // self.num_heads
        if use_attention:
            self.att_n2e = nn.Parameter(torch.Tensor(self.num_heads, self.head_dim))
            self.att_e2n = nn.Parameter(torch.Tensor(self.num_heads, self.head_dim))
            nn.init.xavier_uniform_(self.att_n2e)
            nn.init.xavier_uniform_(self.att_e2n)
        # optional
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
        self.fuse_e.reset_parameters()
        self.fuse_n.reset_parameters()
        self.res_e.reset_parameters()
        self.res_n.reset_parameters()
        self.lin_n2e.reset_parameters()
        self.lin_e2n.reset_parameters()

    def forward(self, x: torch.Tensor, hyperedge_index: torch.Tensor,
                num_nodes: Optional[int] = None, num_edges: Optional[int] = None):
        if num_edges != 0:
            # initial
            e = self.initialize_hyperedge_features(x, hyperedge_index, num_edges)
            # optional
            x_res = x
            e_res = e
            self.mode = 'n2e'
            x_trans = self.lin_n2e(x) + self.bias_n2e
            # x_trans = self.bn_n(x_trans)
            # print('x_trans shape:', x_trans.shape, 'e shape', e.shape)
            e_agg = self.node_to_hyperedge(x_trans, hyperedge_index, e, num_nodes, num_edges)
            # e = self.fuse_features(e_agg)
            # e = e_agg + self.res_e(e_res)
            e = self.act(e)
            self.mode = 'e2n'
            # e_trans = self.lin_e2n(e)
            n_agg = self.hyperedge_to_node(e_agg, hyperedge_index, x_res, num_edges, num_nodes)
            # n = self.fuse_features(n_agg)
            # n = n_agg + self.res_n(x_res)
            n = self.act(n_agg)
            return n, e
        else:
            return self.act(self.res_n(x)), None

    def initialize_hyperedge_features(self, x, hyperedge_index, num_edges):
        node_idx, edge_idx = hyperedge_index
        # x_trans = self.lin_n2e(x) + self.bias_n2e
        # if self.lin_n2e(x).shape[1] == self.in_dim:
        #     x = self.bn_n(x)
        return scatter_mean(x[node_idx], edge_idx, dim=0, dim_size=num_edges)

    def node_to_hyperedge(self, x, hyperedge_index, e, num_nodes, num_edges):
        self.mode = 'n2e'
        norm = self.compute_norm(hyperedge_index, num_nodes, num_edges)
        return self.propagate(
            hyperedge_index,
            x=(x, e),
            norm=norm,
            size=(num_nodes, num_edges)
        )

    def hyperedge_to_node(self, e, hyperedge_index, x, num_edges, num_nodes):
        self.mode = 'e2n'
        norm = self.compute_norm(hyperedge_index, num_nodes, num_edges)
        return self.propagate(
            hyperedge_index.flip([0]),
            x=(e, x),
            norm=norm,
            size=(num_edges, num_nodes)
        )

    def compute_norm(self, hyperedge_index, num_nodes, num_edges):
        node_idx, edge_idx = hyperedge_index
        if self.hyperedge_weight == None or self.training == False:
            node_weight_values = torch.ones_like(node_idx)
        else:
            node_weight_values = self.hyperedge_weight[edge_idx]
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
        # optional
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
        if self.use_attention:
            # optional
            if self.mode == 'n2e':
                att_weights = self.att_n2e  # [num_heads, head_dim]
            else:
                att_weights = self.att_e2n  # [num_heads, head_dim]
            # optional
            x_j_multihead = x_j.view(-1, self.num_heads, self.head_dim)  # [E, num_heads, head_dim]
            # optional
            att_scores = torch.einsum('ehd,hd->eh', x_j_multihead, att_weights)  # [E, num_heads]
            # optional
            att_weights_per_head = torch.softmax(att_scores, dim=1)  # [E, num_heads]
            # optional
            weighted = x_j_multihead * att_weights_per_head.unsqueeze(-1)  # [E, num_heads, head_dim]
            # optional
            aggregated = weighted.view(-1, self.hid_dim)  # [E, hid_dim]
            # optional
            return norm.view(-1, 1) * aggregated
        else:
            return norm.view(-1, 1) * x_j

class HyperGNN(nn.Module):
    def __init__(self, in_dim, edge_dim, node_dim, num_layers, hyperedge_weight, act: Callable = nn.PReLU()):
        super(HyperGNN, self).__init__()
        self.in_dim = in_dim
        self.edge_dim = edge_dim
        self.node_dim = node_dim
        self.num_layers = num_layers
        self.act = act
        self.edge_embeddings = []
        self.layers = nn.ModuleList()
        if num_layers == 1:
            self.layers.append(HG_messagepassing(self.in_dim, self.edge_dim, self.node_dim,
                                                 hyperedge_weight, act=act))
        else:
            self.layers.append(HG_messagepassing(self.in_dim, self.edge_dim, self.node_dim,
                                                 hyperedge_weight, act=act))
            for _ in range(self.num_layers - 2):
                self.layers.append(HG_messagepassing(self.node_dim, self.edge_dim, self.node_dim,
                                                     hyperedge_weight, act=act))
            self.layers.append(HG_messagepassing(self.node_dim, self.edge_dim, self.node_dim,
                                                 hyperedge_weight, act=act))
        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()

    def forward(self, x: torch.Tensor, hyperedge_index: torch.Tensor, num_nodes: int, num_edges: int):
        self.edge_embeddings = []
        for i in range(self.num_layers):
            x, e = self.layers[i](x, hyperedge_index, num_nodes, num_edges)
            self.edge_embeddings.append(e)
        assert len(self.edge_embeddings) == self.num_layers
        return x, e

    # optional
    def get_edge_embeddings(self):
        return self.edge_embeddings