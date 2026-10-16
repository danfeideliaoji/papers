
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import degree, dense_to_sparse, softmax
import random

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

set_seed(42)

def spmm(graph, x, actnn=False, fast_spmm=None, fast_spmm_cpu=None):
    if fast_spmm is None:
        initialize_spmm()
        fast_spmm = CONFIGS["fast_spmm"]
    if fast_spmm_cpu is None:
        initialize_spmm_cpu()
        fast_spmm_cpu = CONFIGS["fast_spmm_cpu"]
    if fast_spmm is not None and str(x.device) != "cpu":
        if graph.out_norm is not None:
            x = graph.out_norm * x

        row_ptr, col_indices = graph.row_indptr, graph.col_indices
        csr_data = graph.raw_edge_weight
        if x.dtype == torch.half:
            csr_data = csr_data.half()
        x = fast_spmm(row_ptr.int(), col_indices.int(), x, csr_data, graph.is_symmetric(), actnn=actnn)

        if graph.in_norm is not None:
            x = graph.in_norm * x
    elif fast_spmm_cpu is not None and str(x.device) == "cpu" and x.requires_grad is False:
        if graph.out_norm is not None:
            x = graph.out_norm * x

        row_ptr, col_indices = graph.row_indptr, graph.col_indices
        csr_data = graph.raw_edge_weight
        x = fast_spmm_cpu(row_ptr.int(), col_indices.int(), csr_data, x)

        if graph.in_norm is not None:
            x = graph.in_norm * x
    else:
        row, col = graph.edge_index
        x = spmm_scatter(row, col, graph.edge_weight, x)
    return x


def spmm_scatter(row, col, values, b):
    r"""
    Args:
        (row, col): Tensor, shape=(2, E)
        values : Tensor, shape=(E,)
        b : Tensor, shape=(N, d)
    """
    output = b.index_select(0, col) * values.unsqueeze(-1).to(b.dtype)
    output = torch.zeros_like(b).scatter_add_(0, row.unsqueeze(-1).expand_as(output), output)
    return output


def initialize_spmm_cpu():
    if CONFIGS["spmm_cpu_flag"]:
        return
    CONFIGS["spmm_cpu_flag"] = True

    from cogdl.operators.spmm import spmm_cpu

    CONFIGS["fast_spmm_cpu"] = spmm_cpu


def initialize_spmm():
    if CONFIGS["spmm_flag"]:
        return
    CONFIGS["spmm_flag"] = True
    if torch.cuda.is_available():
        from cogdl.operators.spmm import csrspmm

        CONFIGS["fast_spmm"] = csrspmm

def glorot_init(input_dim, output_dim):
    init_range = np.sqrt(6.0/(input_dim + output_dim))
    initial = torch.rand(input_dim, output_dim)*2*init_range - init_range
    return nn.Parameter(initial)

class GraphConvSparse(nn.Module):
    def __init__(self, input_dim, output_dim, num_views, n, activation=None):
        super(GraphConvSparse, self).__init__()
        self.weight = glorot_init(input_dim, output_dim)
        self.ortho_weight = torch.zeros_like(self.weight)
        self.activation = activation
        self.num_views = num_views
        self.bias = nn.Parameter(torch.randn(output_dim,), requires_grad=True)


    def forward(self, inputs, adj, fea_sp=False):
        x = inputs
        if fea_sp:
            x = torch.spmm(x, self.weight)
        else:
            x = torch.mm(x, self.weight)
        x = torch.spmm(adj, x) #+ self.bias
        if self.activation is None:
            return x
        else:
            return self.activation(x)


class MvGCN(nn.Module):
    def __init__(self, device, hidden_dims, num_views, dropout, n):
        super(MvGCN, self).__init__()
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.device = device
        self.num_views = num_views
        self.gc1 = GraphConvSparse(self.hidden_dims[0], self.hidden_dims[1], self.num_views, n)
        self.gc2 = GraphConvSparse(self.hidden_dims[1], self.hidden_dims[1], self.num_views, n)

    def forward(self, input, adj):
        # adj_I = torch.eye(adj.shape[0]).to(self.device)
        hidden = F.relu(self.gc1(input, adj.to(self.device)))
        hidden = F.dropout(hidden, self.dropout, training=self.training)
        output = self.gc2(hidden, adj.to(self.device))
        output = F.dropout(output, self.dropout, training=self.training)
        return output

class GCN(nn.Module):
    def __init__(self, device, hidden_dims, num_views, dropout, n):
        super(GCN, self).__init__()
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.device = device
        self.num_views = num_views
        self.gc1 = GraphConvSparse(self.hidden_dims[0], self.hidden_dims[1], self.num_views, n)
        self.gc2 = GraphConvSparse(self.hidden_dims[1], self.hidden_dims[2], self.num_views, n)

    def forward(self, input, adj):
        hidden = F.relu(self.gc1(input, adj.to(self.device)))
        hidden = F.dropout(hidden, self.dropout, training=self.training)
        output = self.gc2(hidden, adj.to(self.device))
        output = F.dropout(output, self.dropout, training=self.training)
        return output

class MLP(torch.nn.Module):
    def __init__(self, n_feature, n_hidden):
        super(MLP, self).__init__()
        # self.dropout = dropout
        self.bias_1 = nn.Parameter(torch.randn(n_hidden, ), requires_grad=True)
        # self.bias_2 = nn.Parameter(torch.randn(n_output, ), requires_grad=True)
        self.hidden = torch.nn.Linear(n_feature, n_hidden)
        # self.predict = torch.nn.Linear(n_hidden, n_output)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.hidden(x)
        return x

class IGLayer(MessagePassing):
    def __init__(self, args, num_relations):
        super(IGLayer, self).__init__(aggr='add')
        self.args = args
        self.dropout = nn.Dropout(args.dropout)
        self.drop_prob = 0.3
        self.gate = nn.Linear(2 * args.hdim[0] // 2, 1)
        self.a = nn.Parameter(torch.zeros(size=(2 * args.hdim[0] // 2, 1)))
        self.num_relations = num_relations
        self.leakrelu = nn.LeakyReLU(0.2)
        nn.init.xavier_normal_(self.gate.weight, gain=1.414)

    def forward(self, h, adj_list, num_nodes):
        # Update embedding
        f_out = torch.zeros_like(h)
        for r in range(len(adj_list)):
            out = self.mp(adj_list[r], h, num_nodes)
            f_out = f_out + out * (1 /len(adj_list))
        return f_out

    def mp(self, adj, h, num_nodes):
        edge_index = dense_to_sparse(adj)[0]
        row, col = edge_index
        norm_degree = degree(row, num_nodes=num_nodes).clamp(min=1)
        deg_inv_sqrt = torch.pow(norm_degree, -0.5)
        deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        norm = self.dropout(norm)
        out = self.propagate(edge_index, size=(h.size(0), h.size(0)), x=h, norm=norm)
        return out

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def update(self, aggr_out):
        return aggr_out

class TGNN(nn.Module):
    def __init__(self, args, num_view, n, input_dims, num_classes, dropout, hdim, device, adj_list):
        super(TGNN, self).__init__()
        self.args = args
        self.bias = self.args.bias
        self.layer_num = self.args.layers
        self.device = device
        self.input_dims = input_dims
        self.hdim = hdim
        self.num_views = num_view
        self.Inv_rep = nn.Parameter(torch.randn(n, hdim[0] // 2, dtype=torch.float))
        self.dropout = args.dropout

        self.featurizer = nn.ModuleList()
        self.intervention = nn.ModuleList()
        self.layers = nn.ModuleList()
        self.Invariant_rep = MvGCN(self.device, hidden_dims=[hdim[0] // 2, hdim[0] // 2], num_views=self.num_views, dropout=dropout, n=n)
        self.MLP = MLP(hdim[0] // 2, num_classes)
        for j in range(len(input_dims)):
            temp_dims = []
            temp_dims.append(input_dims[j])
            temp_dims.append(hdim[0])
            temp_dims.append(num_classes)
            self.featurizer.append(MLP(input_dims[j], hdim[0] // 2))
        for i in range(self.num_views):
            self.intervention.append(MvGCN(device, hidden_dims=[hdim[0] // 2, hdim[0] // 2], num_views=self.num_views, dropout=dropout, n=n))

        self.MLP_classifier = MLP(hdim[0] // 2, num_classes)
        self.MLP_cat = MLP((hdim[0] // 2) * num_view, hdim[0] // 2)
        self.mse = torch.nn.MSELoss()

        for i in range(self.layer_num):
            self.layers.append(IGLayer(self.args, num_view))

    def Classifier(self, m, concat_, adj_list):
        raw = m
        h = concat_
        for i in range(self.layer_num):
            out = self.layers[i](h, adj_list, adj_list[0][0].shape[0])
            h = 0.5 * raw + 0.5 * out

        Output = self.MLP_classifier(h)

        return Output

    def sample_gaussian(self, m, v):
        sample = torch.randn(m.size()).to(self.device)
        z = m + (v**0.5) * sample
        return z
    def targets_loss(self, y_pred, int_y_pred):
        y2 = torch.argmax(int_y_pred, axis=1).clone().detach()
        prob_c = F.softmax(y_pred, dim=1)
        dissimilar_prob = 1.0 - prob_c
        dissimilar_prob = dissimilar_prob / dissimilar_prob.sum(dim=1, keepdim=True)
        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(dissimilar_prob, y2)
        return loss

    def cal_cl_loss(self, feature, avgfeature, labels):
        logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07)).exp()
        logits = logit_scale * feature @ avgfeature.t()
        loss_i = F.cross_entropy(logits, labels)
        loss_t = F.cross_entropy(logits.T, labels)
        ret_loss = (loss_i + loss_t) / 2
        return ret_loss

    def intervention_loss(self, intervention):
        return torch.clamp(torch.norm(torch.pow(intervention, 2)-self.bias+1e-8), min=0.0)

    def kl_normal(self, qm, qv, pm, pv):
        """
        Computes the elem-wise KL divergence between two normal distributions KL(q || p) and
        sum over the last dimension

        Args:
            qm: tensor: (batch, dim): q mean
            qv: tensor: (batch, dim): q variance
            pm: tensor: (batch, dim): p mean
            pv: tensor: (batch, dim): p variance

        Return:
            kl: tensor: (batch,): kl between each sample
        """
        element_wise = (qm - pm).pow(2)
        kl = element_wise.mean()
        return kl

    def condition_prior(self, scale, label, dim):
        mean = ((label-scale[0])/(scale[1])).reshape(-1, 1).repeat(1, dim)
        var = torch.ones(label.size()[0], dim)
        return mean.to(self.device), var.to(self.device)

    def kl_loss(self, m, v, y):
        pm, pv = self.condition_prior([0., 1.], y, m.size()[1])
        return self.kl_normal(m, pv * 0.0001, pm, pv * 0.0001)

    def forward(self, feature, adj_list, labels, mask):
        L_sha = 0
        L_sep = 0
        inter_norm = 0
        L_kl = 0
        Invariant_rep_list = []
        int_z_list = []
        z_intervention_result_list = []
        feature_noise_list = []
        concat_ = torch.Tensor().to(self.args.device)

        m = self.featurizer[0](feature[0])
        v = torch.zeros_like(m)
        feature_noise = self.sample_gaussian(m, v)
        feature_noise_list.append(feature_noise)
        for i in range(self.num_views):
            Invariant_rep = self.Invariant_rep(feature_noise, adj_list[i])
            Invariant_rep_list.append(Invariant_rep)
            concat_ = torch.cat((concat_, Invariant_rep), dim=1)
            concat_ = concat_.to(self.args.device)

        concat_ = self.MLP_cat(concat_)
        for j in range(self.num_views):
            int_z = self.intervention[j](feature_noise_list[0], adj_list[j])
            int_z_list.append(int_z)
            z_intervention_pred = self.MLP(int_z)
            z_intervention_result_list.append(z_intervention_pred)

        Invariant_pred = self.MLP(concat_)
        v = torch.zeros_like(Invariant_pred)
        for j in range(self.num_views):
            L_sha += self.cal_cl_loss(concat_, Invariant_rep_list[j], torch.arange(Invariant_rep_list[j].shape[0]).to(str(self.args.device)))
            L_sep += self.targets_loss(Invariant_pred, z_intervention_result_list[j]).mean()
            inter_norm += self.intervention_loss((int_z_list[j] - Invariant_rep_list[j])).mean()
            L_kl += self.kl_loss(int_z_list[j][mask], v[mask], labels[torch.randperm(labels.size(0))]).mean()
        L_1 = self.kl_loss(concat_[mask], v[mask], labels).mean() + L_kl

        Output = self.Classifier(m, concat_, adj_list)
        Loss = self.args.alpha * (L_sha + L_sep + L_1) + 0.001 * inter_norm


        return Loss, Output


