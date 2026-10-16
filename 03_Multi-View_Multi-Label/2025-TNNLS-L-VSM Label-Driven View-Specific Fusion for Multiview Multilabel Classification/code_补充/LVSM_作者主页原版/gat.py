import torch
import scipy.io as scio
from sklearn.metrics.pairwise import pairwise_distances
import numpy as np
from mymodel import GAT, GAT_by_torch, Net
from RGCN import RGCN
import torch.nn.functional as F
import dgl
import argparse
from dgl.nn.pytorch import RelGraphConv
import scipy.io as sio
import pandas as pd
# from torch_geometric.data import Data
import time
import math
import torch.nn as nn
import os
import matplotlib.pyplot as plt
import sys
import h5py
from dgl import DGLGraph
from dgl.data import citation_graph as citegrh
import time


class GATLayer(nn.Module):
    def __init__(self, g, in_dim, out_dim):
        super(GATLayer, self).__init__()
        self.g = g
        # 公式 (1)
        self.fc = nn.Linear(in_dim, out_dim, bias=False)
        # 公式 (2)
        self.attn_fc = nn.Linear(2 * out_dim, 1, bias=False)

    def edge_attention(self, edges):
        # 公式 (2) 所需，边上的用户定义函数
        z2 = torch.cat([edges.src['z'], edges.dst['z']], dim=1)
        a = self.attn_fc(z2)
        return {'e': F.leaky_relu(a)}

    def message_func(self, edges):
        # 公式 (3), (4)所需，传递消息用的用户定义函数
        return {'z': edges.src['z'], 'e': edges.data['e']}

    def reduce_func(self, nodes):
        # 公式 (3), (4)所需, 归约用的用户定义函数
        # 公式 (3)
        alpha = F.softmax(nodes.mailbox['e'], dim=1)
        # 公式 (4)
        h = torch.sum(alpha * nodes.mailbox['z'], dim=1)
        return {'h': h}

    def forward(self, h):
        # 公式 (1)
        z = self.fc(h)
        self.g.ndata['z'] = z
        # 公式 (2)
        self.g.apply_edges(self.edge_attention)
        # 公式 (3) & (4)
        self.g.update_all(self.message_func, self.reduce_func)
        return self.g.ndata.pop('h')


class MultiHeadGATLayer(nn.Module):
    def __init__(self, g, in_dim, out_dim, num_heads, merge='cat'):
        super(MultiHeadGATLayer, self).__init__()
        self.heads = nn.ModuleList()
        for i in range(num_heads):
            self.heads.append(GATLayer(g, in_dim, out_dim))
        self.merge = merge

    def forward(self, h):
        head_outs = [attn_head(h) for attn_head in self.heads]
        if self.merge == 'cat':
            # 对输出特征维度（第1维）做拼接
            return torch.cat(head_outs, dim=1)
        else:
            # 用求平均整合多头结果
            return torch.mean(torch.stack(head_outs))


class GAT(nn.Module):
    def __init__(self, g, in_dim, hidden_dim, out_dim, num_heads):
        super(GAT, self).__init__()
        self.layer1 = MultiHeadGATLayer(g, in_dim, hidden_dim, num_heads)
        # 注意输入的维度是 hidden_dim * num_heads 因为多头的结果都被拼接在了
        # 一起。 此外输出层只有一个头。
        self.layer2 = MultiHeadGATLayer(g, hidden_dim * num_heads, out_dim, 1)

    def forward(self, h):
        h = self.layer1(h)
        h = F.elu(h)
        h = self.layer2(h)
        return h


def load_cora_data():
    data = citegrh.load_cora()
    features = torch.FloatTensor(data.features)
    labels = torch.LongTensor(data.labels)
    mask = torch.ByteTensor(data.train_mask)
    g = DGLGraph(data.graph)
    return g, features, labels, mask


def train():
    g, features, labels, mask = load_cora_data()
    print('g:{} feature:{} labels:{} mask:{}'.format(g, features.shape, labels.shape, mask.shape))
    print('feature0:', features.sum(dim=0))
    print('feature1:', features.sum(dim=1).shape)
    xpp = input()
    # 创建模型
    net = GAT(g,
              in_dim=features.size()[1],
              hidden_dim=8,
              out_dim=7,
              num_heads=8)
    print(net)

    # 创建优化器
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)

    # 主流程
    dur = []
    for epoch in range(30):
        if epoch >= 3:
            t0 = time.time()

        logits = net(features)
        logp = F.log_softmax(logits, 1)
        loss = F.nll_loss(logp[mask], labels[mask])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch >= 3:
            dur.append(time.time() - t0)

        print("Epoch {:05d} | Loss {:.4f} | Time(s) {:.4f}".format(
            epoch, loss.item(), np.mean(dur)))


if __name__ == '__main__':
    train()
