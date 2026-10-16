
from sklearn import metrics
import torch.nn as nn
import torch.nn.functional as F
import random
import torch
import numpy as np
import scipy.sparse as sp
from sklearn.neighbors import NearestNeighbors



def get_evaluation_results(labels_true, labels_pred):
    ACC = metrics.accuracy_score(labels_true, labels_pred)
    P = metrics.precision_score(labels_true, labels_pred, average='macro')
    R = metrics.recall_score(labels_true, labels_pred, average='macro')
    F1_ma = metrics.f1_score(labels_true, labels_pred, average='macro')
    F1_mi = metrics.f1_score(labels_true, labels_pred, average='micro')

    return ACC, P, R, F1_ma, F1_mi

def set_seed(seed):

    np.random.seed(seed)
    random.seed(seed)
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        # torch.cuda.manual_seed_all(seed)
    return seed

class MLP(nn.Module):
    def __init__(self, input_d, hidden_d, output_d, num_layers, dropout, norm, init_activate):
        super().__init__()

        self.init_activate = init_activate
        self.norm = norm
        self.dropout = dropout
        self.layers = nn.ModuleList([])
        self.num_layers = num_layers

        if num_layers == 1:
            self.layers.append(nn.Linear(input_d, output_d))
        elif num_layers > 1:
            self.layers.append(nn.Linear(input_d, hidden_d))
            for k in range(num_layers - 2):
                self.layers.append(nn.Linear(hidden_d, hidden_d))
            self.layers.append(nn.Linear(hidden_d, output_d))

        self.norm_cnt = num_layers - 1 + int(init_activate)  # how many norm layers we have
        if norm == "batch":
            self.norms = nn.ModuleList([nn.BatchNorm1d(hidden_d) for _ in range(self.norm_cnt)])
        elif norm == "layer":
            self.norms = nn.ModuleList([nn.LayerNorm(hidden_d) for _ in range(self.norm_cnt)])

        self.reset_params()

    def reset_params(self):
        for layer in self.layers:
            nn.init.xavier_normal_(layer.weight.data)
            # nn.init.constant_(layer.bias.data, 0)


    def activate(self, x):
        if self.norm != "none":
            x = self.norms[self.cur_norm_idx](x)  # use the last norm layer
            self.cur_norm_idx += 1
        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        return x

    def forward(self, x):
        self.cur_norm_idx = 0

        if self.init_activate:
            x = self.activate(x)

        for i, layer in enumerate(self.layers):
            x = layer(x)
            if self.num_layers == 1:
                x = F.relu(x)
            if i != len(self.layers) - 1:  # do not activate in the last layer
                x = self.activate(x)

        return x


def construct_adjacency_matrix(features, k_nearest_neighobrs, prunning_one, prunning_two, common_neighbors):
    nbrs = NearestNeighbors(n_neighbors=k_nearest_neighobrs+1, algorithm='ball_tree').fit(features)
    adj_wave = nbrs.kneighbors_graph(features)  # <class 'scipy.sparse.csr.csr_matrix'>

    if prunning_one:
        # Pruning strategy 1
        original_adj_wave = adj_wave.A
        judges_matrix = original_adj_wave == original_adj_wave.T
        np_adj_wave = original_adj_wave * judges_matrix
        adj_wave = sp.csc_matrix(np_adj_wave)
    else:
        # transform the matrix to be symmetric (Instead of Pruning strategy 1)
        np_adj_wave = construct_symmetric_matrix(adj_wave.A)
        adj_wave = sp.csc_matrix(np_adj_wave)

    # obtain the adjacency matrix without self-connection
    adj = sp.csc_matrix(np_adj_wave)
    adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
    adj.eliminate_zeros()

    if prunning_two:
        # Pruning strategy 2
        adj = adj.A
        b = np.nonzero(adj)
        rows = b[0]
        cols = b[1]
        dic = {}
        for row, col in zip(rows, cols):
            if row in dic.keys():
                dic[row].append(col)
            else:
                dic[row] = []
                dic[row].append(col)
        for row, col in zip(rows, cols):
            if len(set(dic[row]) & set(dic[col])) < common_neighbors:
                adj[row][col] = 0
        adj = sp.csc_matrix(adj)
        adj.eliminate_zeros()

    # construct the adjacency hat matrix
    adj_hat = construct_adjacency_hat(adj)

    return adj, adj_wave, adj_hat


def construct_adjacency_hat(adj1):
    """
    :param adj: original adjacency matrix  <class 'scipy.sparse.csr.csr_matrix'>
    :return:
    """
    adj  = adj1.cpu().detach().numpy()
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))  # <class 'numpy.ndarray'> (n_samples, 1)
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    return adj_normalized

def construct_symmetric_matrix(original_matrix):
    """
        transform a matrix (n*n) to be symmetric
    :param np_matrix: <class 'numpy.ndarray'>
    :return: result_matrix: <class 'numpy.ndarray'>
    """
    result_matrix = np.zeros(original_matrix.shape, dtype=float)
    num = original_matrix.shape[0]
    for i in range(num):
        for j in range(num):
            if original_matrix[i][j] == 0:
                continue
            elif original_matrix[i][j] == 1:
                result_matrix[i][j] = 1
                result_matrix[j][i] = 1
            else:
                print("The value in the original matrix is illegal!")
    assert (result_matrix == result_matrix.T).all() == True

    if ~(np.sum(result_matrix, axis=1) > 1).all():
        print("There existing a outlier!")

    return result_matrix