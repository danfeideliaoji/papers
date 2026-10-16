import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.nn.functional as fn
from dgl.nn.pytorch.conv import RelGraphConv, GATConv
import copy
from torch.nn import Parameter
import math
from util import *
class GATLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(GATLayer, self).__init__()
        # self.g = g
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

    def forward(self, g, h):
        # 公式 (1)
        z = self.fc(h)
        g.ndata['z'] = z
        # 公式 (2)
        g.apply_edges(self.edge_attention)
        # 公式 (3) & (4)
        g.update_all(self.message_func, self.reduce_func)
        return g.ndata.pop('h')


class MultiHeadGATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads, merge='cat'):
        super(MultiHeadGATLayer, self).__init__()
        self.heads = nn.ModuleList()
        for i in range(num_heads):
            self.heads.append(GATLayer(in_dim, out_dim))
        self.merge = merge

    def forward(self, g, h):
        head_outs = [attn_head(g, h) for attn_head in self.heads]
        if self.merge == 'cat':
            # 对输出特征维度（第1维）做拼接
            return torch.cat(head_outs, dim=1)
        else:
            # 用求平均整合多头结果
            return torch.mean(torch.stack(head_outs))


class GAT_simple(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_heads):
        super(GAT_simple, self).__init__()
        self.layer1 = MultiHeadGATLayer(in_dim, hidden_dim, num_heads)
        # 注意输入的维度是 hidden_dim * num_heads 因为多头的结果都被拼接在了
        # 一起。 此外输出层只有一个头。
        self.layer2 = MultiHeadGATLayer(hidden_dim * num_heads, out_dim, 1)

    def forward(self, g, h):
        h = self.layer1(g, h)
        h = F.elu(h)
        h = self.layer2(g, h)
        return h


class GAT(nn.Module):
    def __init__(self,
                 num_layers,
                 in_dim,
                 num_hidden,
                 num_classes,
                 heads,
                 activation=F.elu,
                 feat_drop=0,
                 attn_drop=0,
                 negative_slope=0.2,
                 residual=False):
        super(GAT, self).__init__()
        self.num_layers = num_layers
        self.gat_layers = nn.ModuleList()
        self.activation = activation
        # input projection (no residual)
        self.gat_layers.append(GATConv(
            in_dim, num_hidden, heads[0],
            feat_drop, attn_drop, negative_slope, False, self.activation))
        # hidden layers
        for l in range(1, num_layers):
            # due to multi-head, the in_dim = num_hidden * num_heads
            self.gat_layers.append(GATConv(
                num_hidden * heads[l - 1], num_hidden, heads[l],
                feat_drop, attn_drop, negative_slope, residual, self.activation))
        # output projection
        self.gat_layers.append(GATConv(
            num_hidden * heads[-2], num_classes, heads[-1],
            feat_drop, attn_drop, negative_slope, residual, None))

    def forward(self, g, inputs):
        h = inputs
        for l in range(self.num_layers):
            h = self.gat_layers[l](g, h).flatten(1)
        # output projection
        logits = self.gat_layers[-1](g, h).mean(1)
        return logits

class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

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


class RGCN(nn.Module):
    def __init__(self, in_feat, n_hidden, n_class, n_layers, dropout, e_type):
        # model = RGCN(dim_view, 128, target.shape[0], 2, .2, 2)
        super(RGCN, self).__init__()
        self.in_feat = in_feat
        self.n_hidden = n_hidden
        self.n_class = n_class
        self.n_layers = n_layers
        self.dropout = dropout
        self.etype = e_type
        self.encoders = nn.ModuleList()
        self.drop = nn.Dropout(p=self.dropout)
        self.gcs = []
        self.adj = None
        self.embeddings = None
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.trans = []
        self.fcs = []
        #self.gat1 = nn.ModuleList()
        #self.gat2 = nn.ModuleList()
        #self.gat3 = GAT(2, n_hidden, n_hidden, n_hidden, [2,2])

        for i in range(len(in_feat)):
            #self.gat1.append(GAT(2, in_feat[i], in_feat[i], in_feat[i], [2,2]))
            #self.gat2.append(GAT(2, n_hidden, n_hidden, n_hidden, [2,2]))
            if in_feat[i] <= 512:
                encoder = nn.Sequential(
                    nn.Linear(in_feat[i], 256),
                    nn.LeakyReLU(0.2),
                    nn.Dropout(p=self.dropout),
                    nn.Linear(256, n_hidden)
                )
            elif in_feat[i] <= 1024:
                encoder = nn.Sequential(
                    nn.Linear(in_feat[i], 512),
                    nn.LeakyReLU(0.2),
                    nn.Dropout(p=self.dropout),
                    nn.Linear(512, 256),
                    nn.LeakyReLU(0.2),
                    nn.Dropout(p=self.dropout),
                    nn.Linear(256, n_hidden)
                )
            else:
                encoder = nn.Sequential(
                    nn.Linear(in_feat[i], 2048),
                    nn.LeakyReLU(0.2),
                    nn.Dropout(p=self.dropout),
                    nn.Linear(2048, 1024),
                    nn.LeakyReLU(0.2),
                    nn.Dropout(p=self.dropout),
                    nn.Linear(1024, 512),
                    nn.LeakyReLU(0.2),
                    nn.Dropout(p=self.dropout),
                    nn.Linear(512, 256),
                    nn.LeakyReLU(0.2),
                    nn.Dropout(p=self.dropout),
                    nn.Linear(256, n_hidden)
                )
            self.encoders.append(encoder)

        self.RGCNs = nn.ModuleList()
        for i in range(n_layers):
            RGCNLayer = RelGraphConv(n_hidden, n_hidden, e_type)
            self.RGCNs.append(RGCNLayer)
        self.decoder = nn.Sequential(
            nn.Linear(n_hidden, n_class),
            #nn.LeakyReLU(0.2),
            #nn.Dropout(p=self.dropout),
            #nn.Linear(128, n_class)
        )
        self.len_gc = 2
        self.n_hidden = n_hidden

        end = nn.TransformerEncoderLayer(self.n_hidden, 4, self.n_hidden)
        self.te = nn.TransformerEncoder(end, 2)


    def forward(self, feat_list, g, etypes, g_list):
        with g.local_scope():
            feat = None

            ndim = len(feat_list)

            for i in range(len(feat_list)):
                #flag = self.gat1[i](g_list[i], feat_list[i])
                flag = self.encoders[i](feat_list[i])
                #flag = self.encoders[i](feat_list[i])
                if i == 0:
                    feat = flag
                else:
                    feat = torch.cat((feat, flag), dim=0)
            # feat = self.common(feat)
            feat = self.drop(feat)
            # rel = nn.LeakyReLU(0.2)
            for layer in self.RGCNs:
                feat = layer(g, feat, etypes.to(torch.int64))
                # feat=self.drop(feat)
                # feat = rel(feat)
            '''
            le = feat.shape[0] // ndim
            feat2 = torch.zeros(feat.shape[0], feat.shape[1])
            for i in range(ndim):
                lp = i * le
                rp = (i + 1) * le
                feat2[lp:rp] = self.gat2[i](g_list[i], feat[lp:rp])
            # feat = self.gat2(g, feat)
            #feat2 = feat
            #feat2 = self.gat3(g, feat2)
            
            '''
            feat = self.drop(feat)
            #feat = self.decoder(feat)
            #print('feat:', feat.shape)
            res = torch.zeros(feat.shape[0] // ndim, self.n_class)



            for i in range(len(self.embeddings)):
                #break
                #print('i:{} embed:{}'.format(i, self.embeddings[i].shape))
                l = i * res.shape[0]
                r = (i + 1) * res.shape[0]
                x = self.fcs[i](self.embeddings[i])
                x = self.trans[i](x.view(x.shape[0], 1, -1)).squeeze()
                # print('i:{} x1:{}'.format(i, x.shape))

                # print('i:{} x2:{}'.format(i, x.shape))
                '''
                x = self.trans[i](self.embeddings[i].view(self.embeddings[i].shape[0], 1, -1))
                #print('i:{} x1:{}'.format(i, x.shape))
                x = self.fcs[i](x).squeeze()
                #print('i:{} x2:{}'.format(i, x.shape))
                '''
                '''
                
                for j in range(self.len_gc):
                    if j == 0:
                        x = self.gcs[i][j](self.embeddings[i], self.adj)
                    else:
                        x = self.relu(x)
                        x = self.gcs[i][j](x, self.adj)
                '''
                
                x = x.transpose(0, 1)
                #print('x:', x.shape)
                #xp = torch.matmul(feat[l:r], x)
                #print('xp:{} res:{}'.format(xp.shape, res.shape))
                res += torch.matmul(feat[l:r], x)





            #print('res:{} feat:{}'.format(res.shape, feat.shape))
            #for i in range(ndim):
                #res += feat[i * res.shape[0]: (i + 1) * res.shape[0]]
            res /= ndim
            # for i in range(res.shape[0]):
            # for j in range(ndim):
            # res[i] += feat[j * res.shape[0] + i]
            # res[i] /= ndim
            # sig = nn.Sigmoid()
            # res = sig(res)
            # soft = nn.Softmax(dim=1)
            # res = soft(res)
            # print(feat.mean())
            # print(res.mean())
            # print(res.sum())
            # print('res:',res.shape)
            return res


    def get_adj(self, adj, t=0.4):
        #adj = gen_A(self.n_class, t, adj).float()
        #print('adj:', adj)
        for i in range(adj.shape[0]):
            adj[i] /= adj[i].sum()
        #self.adj = gen_adja(adj)
        self.adj = adj
        for i in range(self.adj.shape[0]):
            self.adj[i][i] = 1

        self.adj = torch.eye(self.n_class)

        print('adj2:', self.adj)
        '''
        for i in range(self.adj.shape[0]):
            print('i:{} sum:{}'.format(i, self.adj[i].sum()))
        '''

    def get_embeddings(self, embeddings):
        self.embeddings = embeddings

    def get_gc(self):
        for i in range(len(self.embeddings)):
            gc = []
            '''
            end = nn.TransformerEncoderLayer(self.embeddings[i].shape[1], 1, self.n_hidden)
            self.trans.append(nn.TransformerEncoder(end, 2))
            self.fcs.append(nn.Linear(self.embeddings[i].shape[1], self.n_hidden))
            '''
            end = nn.TransformerEncoderLayer(self.n_hidden, 1, self.n_hidden)
            self.trans.append(nn.TransformerEncoder(end, 2))
            self.fcs.append(nn.Linear(self.embeddings[i].shape[1], self.n_hidden))
            for j in range(self.len_gc):
                if j == 0:
                    gc.append(GraphConvolution(self.embeddings[i].shape[1], self.n_hidden))
                else:
                    gc.append(GraphConvolution(self.n_hidden, self.n_hidden))
            self.gcs.append(gc)