import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.nn.functional as fn
from dgl.nn.pytorch.conv import RelGraphConv, GATConv
import copy
from torch.nn import Parameter, LayerNorm
import math
from sklearn.preprocessing import StandardScaler

class GraphConvolution(nn.Module):

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

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        #support = torch.matmul(input, self.weight)
        output = torch.matmul(adj, torch.matmul(input, self.weight))
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


latent_dim = 128
class Encoder(nn.Module):
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

        end = nn.TransformerEncoderLayer(self.n_hidden, 4, self.n_hidden)
        self.te = nn.TransformerEncoder(end, 2)
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded,decoded