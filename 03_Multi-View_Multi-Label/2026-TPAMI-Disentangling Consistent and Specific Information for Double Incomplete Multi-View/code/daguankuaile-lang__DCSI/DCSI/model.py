import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear
import torch.nn.init as init
from random import sample
import copy
import math
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

class GCNLayer(nn.Module):
    def __init__(self, in_dim, out_dims,use_bias = True):
        super(GCNLayer, self).__init__()
        self.weight = nn.Parameter(torch.randn(in_dim,out_dims))
        self.use_bias = use_bias
        if self.use_bias:
            self.bias = nn.Parameter(torch.randn(out_dims))
        else:
            self.register_parameter('bias', None)
        # self.reset_parameters()
    def reset_parameters(self):
        init.kaiming_uniform_(self.weight)
        if self.use_bias:
            init.zeros_(self.bias)
    def forward(self, A, X):
        support = torch.mm(X, self.weight)
        out = torch.mm(A,support)
        if self.use_bias:
            out = out + self.bias
        return out

class GCNNet(nn.Module):
    """
    定义一个包含两层GraphConvolution的模型
    """
    def __init__(self, input_dim=1024,output_dim=1024,num_layers=2):
        super(GCNNet, self).__init__()
        dims = [(input_dim,input_dim) for n in range(num_layers-1)]
        dims.append((input_dim,output_dim))
        self.gcnList = nn.ModuleList([GCNLayer(input_dim,output_dim) for (input_dim,output_dim) in dims])
        self.gcn1 = GCNLayer(input_dim, input_dim)
        self.gcn2 = GCNLayer(input_dim, output_dim)
    
    def forward(self, adjacency, x):
        # for i in range(len(self.gcnList)-1):
        #     x = F.relu(self.gcnList[i](adjacency, x))
        # logits = self.gcnList[-1](adjacency, x)
        x = self.gcn1(adjacency, x)
        logits = self.gcn2(adjacency, x)
        return logits


class encoder(nn.Module):
    def __init__(self, n_dim, dims, n_z):
        super(encoder, self).__init__()
        # print(n_dim,dims[0])
        self.enc_1 = Linear(n_dim, dims[0])
        self.enc_2 = Linear(dims[0], dims[1])
        self.enc_3 = Linear(dims[1], dims[2])
        self.z_layer = Linear(dims[2], n_z)
        self.z_b0 = nn.BatchNorm1d(n_z)
        
    def forward(self, x):
        enc_h1 = F.relu(self.enc_1(x))
        enc_h2 = F.relu(self.enc_2(enc_h1))
        enc_h3 = F.relu(self.enc_3(enc_h2))
        z = self.z_b0(self.z_layer(enc_h3))
        return z


class decoder(nn.Module):
    def __init__(self, n_dim, dims, n_z):
        super(decoder, self).__init__()
        self.dec_0 = Linear(n_z, n_z)
        self.dec_1 = Linear(n_z, dims[2])
        self.dec_2 = Linear(dims[2], dims[1])
        self.dec_3 = Linear(dims[1], dims[0])
        self.x_bar_layer = Linear(dims[0], n_dim)

    def forward(self, z):
        r = F.relu(self.dec_0(z))
        dec_h1 = F.relu(self.dec_1(r))
        dec_h2 = F.relu(self.dec_2(dec_h1))
        dec_h3 = F.relu(self.dec_3(dec_h2))
        x_bar = self.x_bar_layer(dec_h3)
        return x_bar
    
class MLP(nn.Module):
    def __init__(self,input_dim,hid_dimen1,hid_dimen2,hid_dimen3):
        super(MLP,self).__init__()
        self.Mlp =  nn.Sequential(
        nn.Linear(input_dim,hid_dimen1),
        nn.Tanh(),
        nn.Linear(hid_dimen1, hid_dimen2),
        nn.Tanh(),
        nn.Linear(hid_dimen2, hid_dimen3),
        nn.Tanh(),
        )

    def forward(self, x):
        mlp = self.Mlp(x)
        return mlp


class net(nn.Module):

    def __init__(self, n_stacks, n_input, n_z, nLabel):
        super(net, self).__init__()


        dims = []
        for n_dim in n_input:

            linshidims = []
            for idim in range(n_stacks - 2):
                linshidim = round(n_dim * 0.8)
                linshidim = int(linshidim)
                linshidims.append(linshidim)
            linshidims.append(1500)
            dims.append(linshidims)

        self.encoder_list = nn.ModuleList([encoder(n_input[i], dims[i], n_z) for i in range(len(n_input))])
        self.decoder_list = nn.ModuleList([decoder(n_input[i], dims[i], 1*n_z) for i in range(len(n_input))])
        self.encoder2_list = nn.ModuleList([encoder(n_input[i], dims[i], n_z) for i in range(len(n_input))])
        # self.decoder2_list = nn.ModuleList([decoder(n_input[i], dims[i], n_z) for i in range(len(n_input))])
        
        self.regression = Linear(1*n_z, nLabel)
        self.act = nn.Sigmoid()
        
        self.nLabel = nLabel
        self.BN = nn.BatchNorm1d(n_z)
        
        self.regression1 = Linear(1*n_z, nLabel)
        self.act1 = nn.Sigmoid()
        
        self.classifier_view = MLP(1*n_z, nLabel, nLabel, len(n_input)+1)
        self.vv = nn.BatchNorm1d(n_z)
       
        
    def forward(self, mul_X, we, mode, sigma, epoch):
        # dep_graph = torch.eye(self.nLabel,device=we.device).float()
        batch_size = mul_X[0].shape[0]
        summ = 0
        prop = sigma
        share_zs = []
        conf = []
        p = []
        if mode =='train':
            for i,X in enumerate(mul_X):
                mask_len = int(prop*X.size(-1))

                # st = torch.randint(low=0,high=X.size(-1)-mask_len-1,size=(X.size(0),))
                # # print(st,st+mask_len)
                # mask = torch.ones_like(X)
                # for j,e in enumerate(mask): 
                #     mask[j,st[j]:st[j]+mask_len] = 0
                # mul_X[i] = mul_X[i].mul(mask)

                # for s in range(mul_X[i].size(0)):
                #     mask = sample(range(X.size(-1)),mask_len)
                #     mul_X[i][s,mask] = 0
                
                #随机元素缺失
                mask = torch.ones_like(X)
                for j in range(mask.shape[0]):
                    zero_indices = torch.randperm(mask.shape[1])[:mask_len]
                    mask[j, zero_indices] = 0
                mul_X[i] = mul_X[i].mul(mask)
                
        # for enc_i, enc in enumerate(self.encoder_list):
        #     z_i = enc(mul_X[enc_i])
        #     share_zs.append(z_i)
            
        #     p_i = self.act1(self.regression1(F.relu(z_i)))
        #     p.append(p_i)
              
        #    summ += torch.diag(we[:, enc_i]).mm(z_i)
        # wei = 1 / torch.sum(we, 1)
        # s_z = torch.diag(wei).mm(summ)
        
        # for enc_i, enc in enumerate(self.encoder_list):
        #     z_i = enc(mul_X[enc_i])
        #     share_zs.append(z_i)
        #     summ += torch.diag(we[:, enc_i]).mm(z_i)
        # wei = 1 / torch.sum(we, 1)
        # s_z = torch.diag(wei).mm(summ)
            
        for enc_i, enc in enumerate(self.encoder_list):
            z_i = enc(mul_X[enc_i])
            share_zs.append(z_i)
            
            p_i = self.act1(self.regression1(F.relu(z_i)))
            p.append(p_i)
            alpha = 3
            alpha = torch.tensor(alpha, dtype=torch.float32)
            numerator = 1 - torch.exp(-alpha * torch.abs(2 * p_i - 1))
            denominator = 1 - torch.exp(-alpha)
            confidence_matrix = numerator / denominator
            column_vector = confidence_matrix.mean(dim=1)
            conf_i = column_vector.view(-1, 1)
            s_z = torch.zeros_like(z_i)
            conf_i_clone = conf_i.clone()
            conf.append(conf_i_clone)
        conf_matrix = torch.cat(conf, dim=1) 
        conf_matrix = conf_matrix * we   
        row_sums = torch.sum(conf_matrix, dim=1, keepdim=True)
        conf_matrix1= conf_matrix/ row_sums 
        for i in range(conf_matrix1.shape[0]):  # 对每个样本
            for j in range(conf_matrix1.shape[1]):  # 对每个视图
              # aa = individual_zs[j].data.cpu().numpy()
              # bb = conf_matrix1[i, j].data.cpu().numpy()
              # cc = individual_zs[j][i,:].data.cpu().numpy()
              s_z[i] += conf_matrix1[i, j] * share_zs[j][i,:] * we[i,j]  # 第i行的加权融合
            
        
        
        #     confidence_matrix = torch.abs(2 * p_i - 1)
        #     cc = confidence_matrix.data.cpu().numpy()
            
        #     confidence_matrix = torch.abs(2 * p_i - 1)
        #     confidence_matrix = confidence_matrix.softmax(dim=-1)
        #     cc = confidence_matrix.data.cpu().numpy()
        #     计算每一行的平均值，得到一个一维张量 (行均值)
        #     column_vector = confidence_matrix.mean(dim=1)
        #     c = column_vector.data.cpu().numpy()
        #     将一维张量转换为列向量 (n, 1)
        #     conf_i = column_vector.view(-1, 1)
        #     print(conf_i.shape)
        #     print("hhh")
            
        #     s_z = torch.zeros_like(z_i)
        #     conf_i_clone = conf_i.clone()
        #     conf.append(conf_i_clone)
        #     ci = conf_i_clone.data.cpu().numpy()
            
        # conf_matrix = torch.cat(conf, dim=1)  # PyTorch 的拼接操作
        # print(conf_matrix.shape)
        # cc1 = conf_matrix.data.cpu().numpy()
        # 对拼接后的张量进行 Softmax (在 dim=1 维度上应用)
        # conf_matrix = conf_matrix * we
        # conf_matrix1 = F.softmax(conf_matrix, dim=1)
        # row_sums = torch.sum(conf_matrix, dim=1, keepdim=True)
        # conf_matrix1= conf_matrix/ row_sums
        # print(conf_matrix1.shape)
        # ccc1 = conf_matrix1.data.cpu().numpy()
        
        # for i in range(conf_matrix1.shape[0]):  # 对每个样本
        #     for j in range(conf_matrix1.shape[1]):  # 对每个视图
        #       # aa = individual_zs[j].data.cpu().numpy()
        #       # bb = conf_matrix1[i, j].data.cpu().numpy()
        #       # cc = individual_zs[j][i,:].data.cpu().numpy()
        #       s_z[i] += conf_matrix1[i, j] * share_zs[j][i,:] * we[i,j]  # 第i行的加权融合
        
        # summvz = 0
        # viewsp_zs = []
        # for enc_i, enc in enumerate(self.encoder2_list):
        #     z_i = enc(mul_X[enc_i])
        #     viewsp_zs.append(z_i)
        #     summvz += torch.diag(we[:, enc_i]).mm(z_i)
        # wei = 1 / torch.sum(we, 1)
        # v_z = torch.diag(wei).mm(summvz)
        
        #self.vv = nn.BatchNorm1d(n_z)
        viewsp_zs = []
        # view_s_v = []
        for enc_i, enc in enumerate(self.encoder2_list):
            # 使用 clone() 进行复制
            mul_X_i = mul_X[enc_i].clone()
            z_i = enc(mul_X_i)
            z_i = z_i * we[:, enc_i].unsqueeze(-1)
            viewsp_zs.append(z_i)   
        #v_z_i = torch.cat(viewsp_zs, dim=1) 
        v_z_i = torch.stack(viewsp_zs, dim=0)
        v_z = torch.sum(v_z_i, dim=0)
        #v_z = F.normalize(v_z, dim=1)
        v_z = self.vv(v_z)
        
       
        # 创建一个 n x 1 的列向量，所有值为 1
        ones_column = torch.ones(we.shape[0], 1)
        #将 matrix 和 ones_column 沿着列（维度1）拼接
        result_matrix = torch.cat((we, ones_column.to('cuda:0')), dim=1)
        
        #view classfier begin
        view_class_specific_res=[]
        for i in range(we.shape[1]):
            tmp = self.classifier_view(viewsp_zs[i])
            view_class_specific_res.append(tmp)
        sz_tmp = self.classifier_view(s_z)
        view_class_specific_res.append(sz_tmp)
        
        for v in range(result_matrix.shape[1]):
            # print(view_class_specific_res[v].shape)
            # print(result_matrix[:,v].shape)
            view_class_specific_res[v] = view_class_specific_res[v]*result_matrix[:,v].unsqueeze(1)
        
        
        # z = torch.cat((s_z,v_z),-1)
        z = s_z.mul(v_z.sigmoid_())
        # z = self.BN(z)
        z = F.relu(z)
        # z = s_z+v_z
        
        # x_bar_list = []
        # for dec_i, dec in enumerate(self.decoder_list):

        #     x_bar_list.append(dec(share_zs[dec_i]+viewsp_zs[dec_i]))
        
        
        logi = self.regression(z) #[n c]
        yLable = self.act(logi)
        
        # logi = self.regression(s_z) #[n c]
        # yLable = self.act(logi)
        
        return yLable, z, share_zs, viewsp_zs, p, view_class_specific_res
        #return yLable, z, share_zs, viewsp_zs, p, view_class_specific_res,x_bar_list, 


def get_model(n_stacks,n_input,n_z,Nlabel,device):
    model = net(n_stacks=n_stacks,n_input=n_input,n_z=n_z,nLabel=Nlabel).to(device)
    return model