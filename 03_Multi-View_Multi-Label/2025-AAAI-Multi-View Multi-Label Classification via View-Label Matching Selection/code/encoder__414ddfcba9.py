# 原作者保留的图卷积与自编码器组件；当前 demo.py 入口使用 MLP.merge_views。
# util.encoder() 提供备用的自编码器预训练路径，当前完整 GPU 运行未调用该路径。

import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    import dgl
    import dgl.nn.functional as fn
    from dgl.nn.pytorch.conv import RelGraphConv, GATConv
except Exception:
    dgl = None
import copy
from torch.nn import Parameter, LayerNorm
import math
from sklearn.preprocessing import StandardScaler

# 基础图卷积组件，计算邻接矩阵聚合后的线性变换。
class GraphConvolution(nn.Module):

    # 保存输入/输出维度，并建立权重与可选偏置。
    def __init__(self, in_features, out_features, bias=False):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.Tensor(1, 1, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    # 根据输出维度设置均匀分布初始化范围。
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    # input 为节点特征，adj 为邻接矩阵；主要计算 adj @ (input @ weight)。
    def forward(self, input, adj):
        #support = torch.matmul(input, self.weight)
        output = torch.matmul(adj, torch.matmul(input, self.weight))
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    # 返回组件维度，便于打印模型结构。
    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


latent_dim = 128
# 自编码器：编码到共同隐藏空间，再解码重建原始特征。
class Encoder(nn.Module):
    # 按输入维度选择编码网络，解码层输出与原始输入相同的维度。
    def __init__(self, in_feat, n_hidden,n_class, n_layers, dropout):
        super(Encoder, self).__init__()
        self.in_feat = in_feat
        self.n_hidden = n_hidden
        self.n_class = n_class
        self.n_layers = n_layers
        self.dropout = dropout

        self.drop = nn.Dropout(p=self.dropout)

        self.relu = nn.LeakyReLU(0.2, inplace=True)

        if in_feat <=512:
            self.encoder = nn.Sequential(
                nn.Linear(in_feat, 256),
                nn.LeakyReLU(0.2),
                # LayerNorm(256),
                # nn.Dropout(p=self.dropout),
                nn.Linear(256, n_hidden)
            )
        elif in_feat <= 1024:
            self.encoder = nn.Sequential(
                nn.Linear(in_feat, 512),
                nn.LeakyReLU(0.2),
                # LayerNorm(512),
                # nn.Dropout(p=self.dropout),
                nn.Linear(512, 256),
                nn.LeakyReLU(0.2),
                # LayerNorm(256),
                # nn.Dropout(p=self.dropout),
                nn.Linear(256, n_hidden)
            )
        else:
            self.encoder = nn.Sequential(
                nn.Linear(in_feat, 2048),
                nn.LeakyReLU(0.2),
                # LayerNorm(2048),
                # nn.Dropout(p=self.dropout),
                nn.Linear(2048, 1024),
                nn.LeakyReLU(0.2),
                # LayerNorm(1024),
                # nn.Dropout(p=self.dropout),
                nn.Linear(1024, 512),
                nn.LeakyReLU(0.2),
                # LayerNorm(512),
                # nn.Dropout(p=self.dropout),
                nn.Linear(512, 256),
                nn.LeakyReLU(0.2),
                # LayerNorm(256),
                # nn.Dropout(p=self.dropout),
                nn.Linear(256, n_hidden)
            )

        self.decoder = nn.Sequential(
            nn.Linear(n_hidden, in_feat),
        )
        self.n_hidden = n_hidden

        # 保留的 Transformer 模块仅在此处创建，下面的 forward() 没有使用它。
        end = nn.TransformerEncoderLayer(self.n_hidden, 4, self.n_hidden)
        self.te = nn.TransformerEncoder(end, 2)
    # 同时返回低维 encoded 和重建 decoded，供重建损失使用。
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded,decoded