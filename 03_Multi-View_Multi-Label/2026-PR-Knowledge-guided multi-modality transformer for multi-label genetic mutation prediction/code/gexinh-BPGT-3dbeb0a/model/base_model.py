from tkinter import E
from typing import Sequence
from matplotlib.lines import Line2D
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from utils.utils import initialize_weights
import math
from torch.nn.parameter import Parameter

def size_selection(size_arg):
    size_setting = {"1024_small": [1024, 512, 256], "1024_big": [1024, 512, 384], "1024_min": [1024, 256,128], 
                             "dino": [384, 256, 128], "dino_b": [384, 300, 256], "dino_ss": [384, 64, 32], "dino_s": [384, 128, 64],
                             "hipt": [192, 128, 64], "hipt_att": [192,192,192]
                             }

    size = size_setting[size_arg]
    return size


"""
Attention Network with Sigmoid Gating (3 fc layers)
args:
    L: input feature dimension
    D: hidden layer dimension
    dropout: whether to use dropout (p = 0.25)
    n_classes: number of classes 
"""
class Attn_Net_Gated(nn.Module):
    def __init__(self, L = 1024, D = 256, dropout = False, n_classes = 1):
        super(Attn_Net_Gated, self).__init__()
        self.query = [
            nn.Linear(L, D),
            nn.Tanh()]
        
        self.keys = [nn.Linear(L, D),
                            nn.Sigmoid()]
        if dropout:
            self.query.append(nn.Dropout(0.25))
            self.keys.append(nn.Dropout(0.25))

        self.query = nn.Sequential(*self.query)
        self.keys = nn.Sequential(*self.keys)
        
        self.additional_transform = nn.Linear(D, n_classes)

    def forward(self, x):
        a = self.query(x)
        b = self.keys(x)
        A = a.mul(b) # inner product
        A = self.additional_transform(A)  # N x n_classes

        return A, x


class Sparse_Net(nn.Module):
    def __init__(self, input_dim, mapp):
        super().__init__()
        # self.mapp = mapp.astype(np.float32).to_tensor()
        if str(type(mapp)) == "<class 'numpy.ndarray'>":
            self.mapp = mapp.astype(np.float32).to_tensor()
        else:
            self.mapp = mapp

        self.output_shape = np.shape(self.mapp)[-1]
        self.kernel_shape = (input_dim, self.output_shape)

        # numpy 1.17 use the row index, output shape is [indices, axis] instead of [axis, indices]
        self.nonzero_ind = np.array(np.nonzero(self.mapp.T))
        # self.nonzero_ind = np.array(np.nonzero(self.mapp)) 

        self.no_zero_count = np.shape(self.nonzero_ind)[0]

        self.weight = torch.nn.Parameter(data= torch.zeros((self.no_zero_count,1), dtype=torch.float32), requires_grad=True)
        nn.init.xavier_uniform_(self.weight)
        # self.weight.squeeze()

        self.use_bias = True
        self.bias = torch.nn.Parameter(data=torch.zeros(self.output_shape, dtype=torch.float32) ,requires_grad=True)
        nn.init.constant_(self.bias, 0)

    def forward(self, x):
        device = self.weight.device

        W = torch.zeros(self.kernel_shape, dtype=torch.float32).to(device).T # modified on 3/19

        for i, ind in enumerate(self.nonzero_ind):
            W[ind[0],ind[1]] = self.weight[i]
        
        W = W.T # modified on 3/19

        x = x.type(torch.float32)
        out = torch.matmul(x, W)

        if self.use_bias:
            out = out + self.bias

        return out

class Slide_Net(nn.Module):
    def __init__(self, L = 1024, output_dim = 256, dropout = False):
        super(Slide_Net, self).__init__()
        fc = [nn.Linear(1024, 512), nn.ReLU()]
        # fc = [nn.Linear(1024,512), nn.ReLU()]
        if dropout:
            fc.append(nn.Dropout(0.25))

        fc.append(nn.Linear(512, 256))
        fc.append(nn.ReLU())
        if dropout:
            fc.append(nn.Dropout(0.25))

        fc.append(nn.Linear(256, 128))
        fc.append(nn.ReLU())
        if dropout:
            fc.append(nn.Dropout(0.25))

        # fc.append(nn.Linear(128, 64))
        # fc.append(nn.ReLU())
        # if dropout:
        #     fc.append(nn.Dropout(0.25))

        # # fc.append(nn.Linear(256, 1024))
        # # fc.append(nn.ReLU())
        # fc.append(nn.Linear(64, 128))
        # fc.append(nn.ReLU())
        # if dropout:
        #     fc.append(nn.Dropout(0.25))

        fc.append(nn.Linear(128, output_dim))
        # fc.append(nn.ReLU())

        # fc.append(nn.Linear(1604, 2048), nn.ReLU())

        self.slide_net = nn.Sequential(*fc)

    def forward(self, x):
        h = self.slide_net(x)

        return h

class base_net(nn.Module):
    """ End-to-End model
    
        Args
            mapp: the sparse mapping obtained from gene reactome pathway
            k: the number of random samples
            attn: if use the attention block
            dropout: if use the dropout between each transformation
    """
    def __init__(self, mapp, k = [5, 20, 100], attn = True, dropout = False,):        
        super(base_net, self).__init__()
        self.k_sample = k 
        """ Bottleneck Net 
        """

        # fc.append(nn.Linear(128, mapp.size(0)))
        # fc.append(nn.ReLU())

        # fc.append(nn.Linear(1604, 2048), nn.ReLU())
        
        # self.slide_net = nn.Sequential(*fc)
        self.output_dim = mapp.size(1)
        # self.slide_net = Slide_Net(output_dim=self.output_dim,dropout=True)
        self.slide_net = Slide_Net(output_dim=mapp.size(0),dropout=True)

        self.sparse_net = Sparse_Net(input_dim=mapp.size(0), mapp=mapp)

        self.classifiers = nn.Sigmoid()
        initialize_weights(self)
        # if attn == True:
        #     self.atten_net = Attn_Net(input_dim=1024)

    def forward(self, x, region=None, label=None, label_ind = None):
        # feature extraction: feature embedding -> pathway 
        device = x.device    
        x = x.type(torch.float32)

        h = self.slide_net(x)

        # transpose sparse net: pathway -> gene
        h = self.sparse_net(h)

        # update partial nodes
        """The more general approach will be to copy the weights you want frozen, 
        call opt.step() and then rewrite the frozen weights back into your param.
        ref: https://discuss.pytorch.org/t/how-do-i-freeze-the-specific-weights-in-a-layer/104722/3
        """

        h = self.classifiers(h)

        if self.training:
            k = int(np.random.choice(self.k_sample))
            h = self.forward_fixed_k(h, k)
            lasso_loss= self.compute_l1_loss(h)
            return h, lasso_loss
        else:
            pred = 0
            for k in self.k_sample:
                pred += self.forward_fixed_k(h, int(k)) / len(self.k_sample)
            lasso_loss= self.compute_l1_loss(pred)
            return pred, lasso_loss

        """ patch-level learning based on clustering
        """

        # return logits, Y_hat

    def weighted_sum(self, h, region=None, label=None, label_ind = None):
                # Randomly sampling Top K
        # num = len(self.k_sample)
        weighted_y = []
        patch_size = h.size(-2)

        if self.training == True:

            # label_ind = torch.tensor(label_ind,dtype=torch.int)
            y_pre = torch.index_select(input=h, dim=-1, index=label_ind)
            y_tar = torch.index_select(input=label, dim=-1, index=label_ind).T
            scores = torch.matmul(y_pre, y_tar)  # (?, patch, label) * (label, 1) = (?, patch, 1)
            scores = torch.squeeze(scores)
            _, ranks_ind = torch.sort(scores, descending=True) 


        for i, num in enumerate(self.k_sample):
            temp_h = torch.index_select(input=h, dim=-2, index=ranks_ind[0:num])
            temp_h = torch.mean(temp_h,dim=-2,keepdim=True)  # (?, 1, gene)                

            if i == 0:
                weighted_y = temp_h # (?, 1, gene))
            else:
                weighted_y = torch.cat((weighted_y,temp_h),dim=-2) # (?, i, gene))

        # else:
        #     for i, num in enumerate(self.k_sample):            
        #         # for i in range(num):
        #         # TODO: debug this part
        #         rand_ind = torch.randint(0, patch_size, (num,)).to(device)
        #         temp_h = torch.index_select(input=h, dim=-2, index=rand_ind)
        #         temp_h = torch.mean(temp_h,dim=-2,keepdim=True)  # (?, 1, gene)
                
        #         if i == 0:
        #             weighted_y = temp_h # (?, 1, gene))
        #         else:
        #             weighted_y = torch.cat((weighted_y,temp_h),dim=-2) # (?, i, gene))
            Y_hat = torch.mean(weighted_y, dim=-2, keepdim=False, dtype=torch.float32) # (?, gene))
          # Y_hat = self.weighted_sum(weighted_y, dim=-2, keepdim=False)
        return Y_hat


    def forward_fixed_k(self, x, k):
        mask, _ = torch.max(x, dim=2, keepdim=True) 
        mask = (mask > 0).float()   #相当于把预测是负值的patch给去除了

        x = x * mask
        t, _ = torch.topk(x, k, dim=1, largest=True, sorted=True)
        x = torch.sum(t * mask[:, :k, :], dim=1) / torch.sum(mask[:, :k, :], dim=1)
        return x

    def compute_l1_loss(self, w):
          return torch.abs(w).sum()

    def relocate(self):
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # TODO: modify this part
        self.slide_net = self.slide_net.to(device)
        self.classifiers = self.classifiers.to(device)
        self.sparse_net = self.sparse_net.to(device)





""" Basic model:
        DeepHist:        
        1) MLP + self-attention + average pooling  -> binary classification

        Genemutation:
        2) MLP + average pooling  -> binary classification

        BCOMIL:
        3)
""" 

class Self_Attn(nn.Module):
    """ Self attention Layer"""
    def __init__(self, in_dim):
        super(Self_Attn, self).__init__()
        self.query_conv = nn.Conv1d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.key_conv = nn.Conv1d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.ones(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( 1 X C X N)
            returns :
                out : (1 X C X N) self attention value + input feature
                attention:  1 X N X N  (N is number of patches)
        """
        proj_query = self.query_conv(x).permute(0, 2, 1)  # 1 X N X C
        proj_key = self.key_conv(x)  # 1 X C x N
        energy = torch.bmm(proj_query, proj_key)  # 1 X N X N
        attention = self.softmax(energy)  # 1 X N X N

        out = torch.bmm(x, attention.permute(0, 2, 1))  # 1 X C X N
        out = self.gamma * out + x

        return out

class AttnClassifier(nn.Module):
    def __init__(self, aggregation='att',size_arg = "small", num_classes=2, k_sample=-1, dropout=True, dropout_rate=0.1):
        super(AttnClassifier, self).__init__()
        size = size_selection(size_arg)

        self.agg = aggregation
        self.fc1 = [nn.Linear(size[0], size[1]), nn.ReLU()]
        self.fc2 = [nn.Linear(size[1], size[2]), nn.ReLU()]

        self.attn = Self_Attn(size[2])

        self.fc3 = nn.Linear(size[2], num_classes)
        self.num_classes= num_classes
        self.k_sample = k_sample

        if dropout:
            self.fc1.append(nn.Dropout(dropout_rate))
            self.fc2.append(nn.Dropout(dropout_rate))

        self.fc1 = nn.Sequential(*self.fc1)
        self.fc2 = nn.Sequential(*self.fc2)

        # initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.0)

    def forward(self, x):
        """
        Inputs:
            x: (1, patch, features)
        Outputs: 
            x: (1, features)
        """
        # ramdomly select
        if self.k_sample != -1:
            total_patch = x.size()[1]
            row_sequence = np.arange(total_patch)
            np.random.shuffle(row_sequence)
            # k = int(np.random.choice(self.k_sample))
            k = row_sequence[0:self.k_sample]
            x = x[:, k, :]

        # forward
        x = self.fc1(x)
        x = self.fc2(x)

        if self.agg == 'att':
            x = self.attn(x.permute(0, 2, 1))  # X: 1, features, patches
            x2 = x.mean(dim=-1) # 1, features
            x3 = x.permute(0, 2, 1)

        elif self.agg == 'mean':
            x3 = x
            x2 = x.permute(0, 2, 1).mean(dim=-1)

        elif self.agg == 'max':
            x3 = x
            x2 = x.permute(0, 2, 1).max(dim=-1)

        x = self.fc3(x2) 
        
        # x3 is the tensor without pooling
        if self.num_classes == 2:
            return x
        else:
            # x: (1, k), x2: (1,d), x3: (1, p, d)
            return x, x2, x3

    def relocate(self):
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # TODO: modify this part
        self.fc1 = self.fc1.to(device)
        self.fc2 = self.fc2.to(device)
        self.attn = self.attn.to(device)
        self.fc3 = self.fc3.to(device)


try:
    from kmeans_pytorch import kmeans
except ImportError:
    kmeans = None


def _require_kmeans():
    if kmeans is None:
        raise ImportError(
            "kmeans_pytorch is required only for legacy cluster-based MIL branches. "
            "Install it or choose a non-cluster visual extractor."
        )


# 
class Cluster_MIL(nn.Module):
    def __init__(self, aggregation='att',size_arg = "small", num_classes=2, num_clusters=4, k_sample=-1):
        super(Cluster_MIL, self).__init__()         
        size = size_selection(size_arg)

        self.agg = aggregation
        self.fc1 = nn.Linear(size[0], size[1])
        self.fc2 = nn.Linear(size[1], size[2])
        self.attn = Self_Attn(size[2])
        self.fc3 = nn.Linear(size[2], num_classes)

        self.num_classes = num_classes
        self.num_clusters = num_clusters
        self.k_sample = k_sample
        # initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.0)

    def forward(self, x):
        """
        Inputs:
            x: (patch, features)
        """
        #
        if len(x.size()) > 2:
            x = x.squeeze(0)
        
        # randomly select
        if self.k_sample != -1:
            total_patch = x.size()[1]
            row_sequence = np.arange(total_patch)
            np.random.shuffle(row_sequence)
            # k = int(np.random.choice(self.k_sample))
            k = row_sequence[0:self.k_sample]
            x = x[:, k, :]
        
        # cluster
        _require_kmeans()
        cluster_ids_x, cluster_centers = kmeans(
                    X=x, num_clusters=self.num_clusters, distance='euclidean', device=x.device
                )
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        if self.agg == 'att':
            x = self.attn(x.permute(0, 2, 1))  # X: 1, features, patches
            x2 = x.mean(dim=-1) # 1, features

        elif self.agg == 'mean':
            x2 = x.permute(0, 2, 1).mean(dim=-1)
            
        elif self.agg == 'max':
            x2 = x.permute(0, 2, 1).max(dim=-1)
        x = self.fc3(x2) 

        return x

    def relocate(self):
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # TODO: modify this part
        self.fc1 = self.fc1.to(device)
        self.fc2 = self.fc2.to(device)
        self.attn = self.attn.to(device)
        self.fc3 = self.fc3.to(device)


""" 3th model: CLAM
"""
class Attn_Net(nn.Module):
    
    def __init__(self, L = 1024, D = 256, dropout = False, n_classes = 1):
        super(Attn_Net, self).__init__()
        self.module = [
            nn.Linear(L, D),
            nn.Tanh()]

        if dropout:
            self.module.append(nn.Dropout(0.25))

        self.module.append(nn.Linear(D, n_classes))
        
        self.module = nn.Sequential(*self.module)
    
    def forward(self, x):
        return self.module(x), x # N x n_classes

class Attn_Net_Gated(nn.Module):
    """Attention Network with Sigmoid Gating (3 fc layers)
    
    args:
        L: input feature dimension
        D: hidden layer dimension
        dropout: whether to use dropout (p = 0.25)
        n_classes: number of classes 
    """
    def __init__(self, L = 1024, D = 256, dropout = False, n_classes = 1, dropout_rate = 0.5):
        super(Attn_Net_Gated, self).__init__()
        self.attention_a = [
            nn.Linear(L, D),
            nn.Tanh()]
        
        self.attention_b = [nn.Linear(L, D),
                            nn.Sigmoid()]
        if dropout:
            self.attention_a.append(nn.Dropout(dropout_rate))
            self.attention_b.append(nn.Dropout(dropout_rate))

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)
        
        self.attention_c = nn.Linear(D, n_classes)

        self.gamma = nn.Parameter(torch.ones(1))
        initialize_weights(self)
        
    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b) # inner product
        A = self.attention_c(A)  # N x 1

        return A, x

class CLAM_SB(nn.Module):
    """ 
        model: 'wsl'- weakly supervised learning, 'attention' - attention module
    """
    def __init__(self, model='wsl', aggregation='att', gate = True, size_arg = "small", dropout = False, k_sample=8, n_classes=2,
        instance_loss_fn=nn.CrossEntropyLoss(), subtyping=False, dropout_rate=0.5):
        super(CLAM_SB, self).__init__()
        size = size_selection(size_arg)
        
        self.model = model
        if self.model == 'attention' or self.model == 'geneh' or self.model == 'deephis':
            self.agg = aggregation
            self.fc1 = nn.Linear(size[0], size[1])
            self.fc2 = nn.Linear(size[1], size[2])
            self.attn = Self_Attn(size[2])
            self.fc3 = nn.Linear(size[2], n_classes)
            # initialization
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    torch.nn.init.xavier_uniform_(m.weight)
                    m.bias.data.fill_(0.0)
            
            self.k_sample = k_sample # used for reduce patch

        elif self.model == 'cor_mil_binary':
            self.cor_mil = TransMIL(size_arg, 2)

        elif self.model == 'wsl' :
            fc = [nn.Linear(size[0], size[1]), nn.ReLU()]
            if dropout:
                fc.append(nn.Dropout(0.25))
            if gate:
                attention_net = Attn_Net_Gated(L = size[1], D = size[2], dropout = dropout, n_classes = 1, dropout_rate=dropout_rate)
            else:
                attention_net = Attn_Net(L = size[1], D = size[2], dropout = dropout, n_classes = 1)

            fc.append(attention_net)
            self.attention_net = nn.Sequential(*fc)
            self.classifiers = nn.Linear(size[1], n_classes)
            
            instance_classifiers = [nn.Linear(size[1], 2) for i in range(n_classes)]
            
            self.instance_classifiers = nn.ModuleList(instance_classifiers)
            self.k_sample = k_sample
            self.instance_loss_fn = instance_loss_fn
            
            self.n_classes = n_classes
            self.subtyping = subtyping

            initialize_weights(self)

        

    def relocate(self):
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if self.model == 'attention':        
            self.fc1 = self.fc1.to(device)
            self.fc2 = self.fc2.to(device)
            self.attn = self.attn.to(device)
            self.fc3 = self.fc3.to(device)
        elif self.model == 'wsl' :
            self.attention_net = self.attention_net.to(device)
            self.instance_classifiers = self.instance_classifiers.to(device)
            self.classifiers = self.classifiers.to(device)
        elif self.model == 'cor_mil_binary':
            self.cor_mil = self.cor_mil.to(device)


    
    @staticmethod
    def create_positive_targets(length, device):
        return torch.full((length, ), 1, device=device).long()
    @staticmethod
    def create_negative_targets(length, device):
        return torch.full((length, ), 0, device=device).long()
    
    #instance-level evaluation for in-the-class attention branch
    def inst_eval(self, A, h, classifier): 
        device=h.device
        if len(A.shape) == 1:
            A = A.view(1, -1)
        top_p_ids = torch.topk(A, self.k_sample)[1][-1]   # topk -> (values, inds)
        # print('topk ind is {}'.format(top_p_ids))
        # print(A.size())
        # print(top_p_ids)
        top_p = torch.index_select(h, dim=0, index=top_p_ids)
        top_n_ids = torch.topk(-A, self.k_sample, dim=1)[1][-1]
        top_n = torch.index_select(h, dim=0, index=top_n_ids)
        p_targets = self.create_positive_targets(self.k_sample, device)
        n_targets = self.create_negative_targets(self.k_sample, device)

        all_targets = torch.cat([p_targets, n_targets], dim=0)
        all_instances = torch.cat([top_p, top_n], dim=0)
        logits = classifier(all_instances)
        all_preds = torch.topk(logits, 1, dim = 1)[1].squeeze(1)
        # print(logits.data.size(), all_targets.data.size())
        instance_loss = self.instance_loss_fn(logits, all_targets)
        
        return instance_loss, all_preds, all_targets
    
    #instance-level evaluation for out-of-the-class attention branch
    def inst_eval_out(self, A, h, classifier):
        device=h.device
        if len(A.shape) == 1:
            A = A.view(1, -1)
        top_p_ids = torch.topk(A, self.k_sample)[1][-1]
        top_p = torch.index_select(h, dim=0, index=top_p_ids)
        p_targets = self.create_negative_targets(self.k_sample, device)
        logits = classifier(top_p)
        p_preds = torch.topk(logits, 1, dim = 1)[1].squeeze(1)
        instance_loss = self.instance_loss_fn(logits, p_targets)
        return instance_loss, p_preds, p_targets

    def forward(self, h, label=None, instance_eval=False, return_features=False, attention_only=False):
        device = h.device

        if self.model == 'attention' or self.model == 'geneh' or self.model == 'deephis':
            """ h: (patch, feature)
            """  
            if self.model == 'geneh':
                self.agg == 'mean'
                if self.k_sample != -1:
                    total_patch = h.size()[0]
                    row_sequence = np.arange(total_patch)
                    np.random.shuffle(row_sequence)
                    # k = int(np.random.choice(self.k_sample))
                    k = row_sequence[0:self.k_sample]
                    h = h[k, :]
            elif self.model == 'deephis':
                # TODO: modify this part
                _require_kmeans()
                cluster_ids_x, cluster_centers = kmeans(
                    X=h, num_clusters=2, distance='euclidean', device=device
                )
                h = h[cluster_ids_x[0,:],:]
                self.agg == 'att'


            x = F.relu(self.fc1(h))
            x = F.relu(self.fc2(x))

            if self.agg == 'att':
                x = self.attn(x.unsqueeze(0).permute(0, 2, 1))  # X: 1, features, patches
                x2 = x.mean(dim=-1) # 1, features
            elif self.agg == 'mean':
                x2 = x.unsqueeze(0).permute(0, 2, 1).mean(dim=-1)
            elif self.agg == 'max':
                x2, _ = x.unsqueeze(0).permute(0, 2, 1).max(dim=-1)
            logits = self.fc3(x2)             
            Y_hat = torch.topk(logits, 1, dim = 1)[1]  # TODO: what is this???
            Y_prob = F.softmax(logits, dim = 1)

            A_raw = [] 
            results_dict = {}

        elif self.model == 'cor_mil_binary':
            results_dict = self.cor_mil(h)
            logits = results_dict['logits']
            Y_hat = results_dict['Y_hat']
            Y_prob = results_dict['Y_prob']
            A_raw = []
            results_dict = {}

        elif self.model == 'wsl' :

            # print('before att', h, h.size())
            A, h = self.attention_net(h)  # NxK
            # print('after att', h, h.size())

            A = torch.transpose(A, 1, 0)  # KxN            
            A = F.softmax(A, dim=1)  # softmax over N
            A_raw = A

            if instance_eval:
                total_inst_loss = 0.0
                all_preds = []
                all_targets = []
                inst_labels = F.one_hot(label, num_classes=self.n_classes).squeeze() #binarize label
                # print(inst_labels)

                for i in range(len(self.instance_classifiers)):
                    # print(i)
                    inst_label = inst_labels[i].item()
                    classifier = self.instance_classifiers[i]
                    # 似乎用两个分类器没有任何区别？？0 和 1 计算的 inst loss 都是基于同样的Attention matrix
                    if inst_label == 1: #in-the-class:
                        instance_loss, preds, targets = self.inst_eval(A, h, classifier)
                        all_preds.extend(preds.cpu().numpy())
                        all_targets.extend(targets.cpu().numpy())
                    else: #out-of-the-class
                        if self.subtyping:
                            instance_loss, preds, targets = self.inst_eval_out(A, h, classifier)
                            all_preds.extend(preds.cpu().numpy())
                            all_targets.extend(targets.cpu().numpy())
                        else:
                            continue
                    total_inst_loss += instance_loss

                if self.subtyping:
                    total_inst_loss /= len(self.instance_classifiers)
                    
            M = torch.mm(A, h) 
            logits = self.classifiers(M)
            Y_hat = torch.topk(logits, 1, dim = 1)[1]  # TODO: what is this???
            Y_prob = F.softmax(logits, dim = 1)
            if instance_eval:
                results_dict = {'instance_loss': total_inst_loss, 'inst_labels': np.array(all_targets), 
                'inst_preds': np.array(all_preds)}
            else:
                results_dict = {}
            if return_features:
                results_dict.update({'features': M})


        return logits, Y_prob, Y_hat, A_raw, results_dict


""" 4th model: attention aggregation
"""

class GatedAttention(nn.Module):
    """Attention Network with Sigmoid Gating (3 fc layers)
    
    args:
        L: input feature dimension
        D: hidden layer dimension
        dropout: whether to use dropout (p = 0.25)
        n_classes: number of classes 
    """
    def __init__(self, L = 1024, D = 256, dropout = False, n_classes = 1, dropout_rate=0.5):
        super(GatedAttention, self).__init__()
        self.attention_a = [
            nn.Linear(L, D),
            nn.Tanh()]
        
        self.attention_b = [nn.Linear(L, D),
                            nn.Sigmoid()]
        if dropout:
            self.attention_a.append(nn.Dropout(dropout_rate))
            self.attention_b.append(nn.Dropout(dropout_rate))

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)
        
        self.attention_c = nn.Linear(D, n_classes)

        self.gamma = nn.Parameter(torch.ones(1))

    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b) # inner product

        return A, x

class Att_multi_label_module(nn.Module):
    """ aggregate patch-level embeddings for binary classification/ multi-label classification
    
        Args:
            cluster: 'att', 'mil', and 'mtl', where means attetion, multi-instance learning and multi-task learning
            cls_method: 'binary' or 'multi'
            size_arg
            n_classes
    """
    def __init__(self, n_classes=2, size_arg = "dino_ss", dropout = False, k_sample=8,dropout_rate=0.5):
        super(Att_multi_label_module, self).__init__()
        size = size_selection(size_arg)
        # self.size_dict = {"1024_small": [1024, 512, 256], "1024_big": [1024, 512, 384], "1024_min": [1024, 256,128], 
        #                 "dino": [384, 256, 128], "dino_b": [384, 300, 256], "dino_ss": [384, 64, 32], "dino_s": [384, 128, 64]}
        # size = self.size_dict[size_arg]
        fc = [nn.Linear(size[0], size[1]), nn.ReLU()]
        attention_net = Attn_Net_Gated(L = size[1], D = size[2], dropout = dropout, n_classes = 1, dropout_rate=dropout_rate)
        fc.append(attention_net)
        self.attention_net = nn.Sequential(*fc) 

        self.classifiers = nn.Linear(size[1], n_classes)
        initialize_weights(self)

    def compute_l1_loss(self, w):
        return torch.abs(w).sum()

    def forward(self, h, label=None, instance_eval=True):
        A, h = self.attention_net(h)  # nx1 and nxd
        A = torch.transpose(A, 1, 0)
        A = F.softmax(A, dim=1)
        M = torch.mm(A, h)   # 1 x d
        logits = self.classifiers(M)  # 1xd --> 1xN, this is used to calculate CE loss
        Y_hat = torch.sigmoid(logits)
        lasso_loss = self.compute_l1_loss(Y_hat)

        result_dic = {}
        result_dic['lasso_loss'] = lasso_loss
        result_dic['Y_region_logits'] = []
        result_dic['Y_region_prob'] = []


        return logits, Y_hat, result_dic



    def relocate(self):
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.attention_net = self.attention_net.to(device)
        self.classifiers = self.classifiers.to(device)
        # self.instance_classifiers = self.instance_classifiers.to(device)


""" transformer decoder + GCN
"""
class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = torch.nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class GCN(nn.Module):
    def __init__(self, nclass, model_size='small', dropout=0.1):
        """ 
            model_size
        """
        super(GCN, self).__init__()

        dmodel =  [nclass*3, nclass*2, nclass]

        self.gc1 = GraphConvolution(dmodel[0], dmodel[1])
        self.gc2 = GraphConvolution(dmodel[1], nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        x = F.relu(x)
        return x
        #TODO: delete the classifier
        # return F.log_softmax(x, dim=1)

    def relocate(self):
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gc1 = self.gc1.to(device)
        self.gc2 = self.gc2.to(device)

class Att_ML(nn.Module):
    """ aggregate patch-level embeddings for binary classification/ multi-label classification
    
        Args:
            cluster: 'att', 'mil', and 'mtl', where means attetion, multi-instance learning and multi-task learning
            cls_method: 'binary' or 'multi'
            size_arg
            n_classes
    """
    def __init__(self, n_classes=2, size_arg = "dino_ss", dropout = False, dropout_rate=0.5, k_sample=8):
        super(Att_ML, self).__init__()
        self.size_dict = {"1024_small": [1024, 512, 256], "1024_big": [1024, 512, 384], "1024_min": [1024, 256,128], 
                          "dino": [384, 256, 128], "dino_b": [384, 300, 256], "dino_ss": [384, 64, 32], "dino_s": [384, 128, 64],
                          "hipt": [192, 128, 64],
                          }
        size = self.size_dict[size_arg]
        fc = [nn.Linear(size[0], size[1]), nn.ReLU()]
        attention_net = Attn_Net_Gated(L = size[1], D = size[2], dropout = dropout, n_classes = 1, dropout_rate=dropout_rate)
        fc.append(attention_net)
        self.attention_net = nn.Sequential(*fc) 

        # initialize linear layer
        initialize_weights(self)
        # self.classifiers = nn.Linear(size[1], n_classes)

    def compute_l1_loss(self, w):
        return torch.abs(w).sum()

    def forward(self, h, label=None, instance_eval=True):
        A, h = self.attention_net(h)  # nx1 and nxd
        A = torch.transpose(A, 1, 0)
        A = F.softmax(A, dim=1)
        M = torch.mm(A, h)   # 1 x d
        return M

        # logits = self.classifiers(M)  # 1xd --> 1xN, this is used to calculate CE loss
        # Y_hat = torch.sigmoid(logits)
        # lasso_loss = self.compute_l1_loss(Y_hat)

        # return logits, Y_hat, lasso_loss

    def relocate(self):
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.attention_net = self.attention_net.to(device)
        self.classifiers = self.classifiers.to(device)



class Multi_View_Aggregation(nn.Module):
    def __init__(self, n_classes=2, size_arg = "dino_ss", dropout = False, dropout_rate=0.5, k_sample=8):
        super(Multi_View_Aggregation, self).__init__()

        # 1 d convolution
        self.l_view = nn.Conv2d(1,4,kernel_size=(20,1),stride=(4,1))
        self.m_view = nn.Conv2d(1,4,kernel_size=(10,1),stride=(4,1))
        self.s_view = nn.Conv2d(1,4,kernel_size=(5,1),stride=(4,1))

        # group convolution: (1, feature_in, patch/2, patch/2) --group-wise--> (1, feature_out, patch/2, patch/2)
        self.g_l_view = nn.Conv2d(1,4,kernel_size=(20,1),stride=(4,1))
        self.g_m_view = nn.Conv2d(1,4,kernel_size=(10,1),stride=(4,1))
        self.g_s_view = nn.Conv2d(1,4,kernel_size=(5,1),stride=(4,1))

        self.g1 = torch.nn.Parameter(torch.Tensor(1))
        self.g2 = torch.nn.Parameter(torch.Tensor(1))
        self.g3 = torch.nn.Parameter(torch.Tensor(1))

        self.act = nn.ReLU()
        initialize_weights(self)

    def forward(self, x):
    # input x: [patch, feature]

    # restore 2d
        n = x.size()[0]
        N_sqrt = torch.ceil(torch.sqrt(n))
        N = torch.square(N_sqrt)
        M = N - n 
    # padding and reshape 

    #
        h = x.unsqueeze(0).unsqueeze(1)
        h1 = self.l_view(h) # [1,4, patch/4, feature]
        h2 = self.m_view(h)
        h3 = self.s_view(h)
        
        # sum
        out = self.g1 * h1 + self.g2 * h2 + self.g3 * h3

        # concate
        out = torch.cat((h1,h2,h3), 1)  #[1, nxchannel, patch/4, feature]
        out = self.act(out)
        return out

class CNN_Patch_Aggregation(nn.Module):
    """ aggregate patch-level embeddings for binary classification/ multi-label classification
    
        Args:
            cluster: 'att', 'mil', and 'mtl', where means attetion, multi-instance learning and multi-task learning
            cls_method: 'binary' or 'multi'
            size_arg
            n_classes
    """
    def __init__(self, n_classes=2, size_arg = "dino_ss", dropout = False, dropout_rate=0.5, k_sample=8):
        super(CNN_Patch_Aggregation, self).__init__()
        self.size_dict = {"1024_small": [1024, 512, 256], "1024_big": [1024, 512, 384], "1024_min": [1024, 256,128], 
                            "dino": [384, 256, 128], "dino_b": [384, 300, 256], "dino_ss": [384, 64, 32], "dino_s": [384, 128, 64]}
        size = self.size_dict[size_arg]
        
        fc = [nn.Linear(size[0], size[1]), nn.ReLU()]
        fc.extend([nn.Conv2d(size[1], size[2], kernel=(3,1), stride=(2,1)), nn.ReLU()]) 


        # attention_net = Attn_Net_Gated(L = size[1], D = size[2], dropout = dropout, n_classes = 1, dropout_rate=dropout_rate)
        # fc.append(attention_net)
        self.cnn_patch = nn.Sequential(*fc) 


        # initialize linear layer
        initialize_weights(self)
        # self.classifiers = nn.Linear(size[1], n_classes)

    def forward(self, h, label=None, instance_eval=True):
        A, h = self.attention_net(h)  # nx1 and nxd
        A = torch.transpose(A, 1, 0)
        A = F.softmax(A, dim=1)
        M = torch.mm(A, h)   # 1 x d
        return M

"""
    For HIPT representation:
        1) simple 3-layer MLP: (P, emb) -> (P, emb/2) -> (P, emb/4)
        2) self-attention with positional embedding: (P+1, embedding) --> (cls token, embedding)
        3) self-attention with dimensional reduction: (P, emb) -> (P, emb/2)
        4) self-attention withtout dimensional reduction, i.e., x = gamma*self-att + x : ()
        5) transformer encoder layer: (P, emb) -> (P, emb)
"""

class MLP_FC(nn.Module):
    def __init__(self, size_arg, dropout, dropout_rate):
        super(MLP_FC, self).__init__()
        size = size_selection(size_arg)
        # size = self.size_dict[size_arg]
        fc = []
        for i in (range(len(size)-1)):
            layer = [nn.Linear(size[i], size[i+1]), nn.ReLU()]
            if dropout == True:
                layer.append(nn.Dropout(p=dropout_rate))
            fc.extend(layer)
        self.mlp_net = nn.Sequential(*fc) 

        # initialize linear layer
        initialize_weights(self)

    def forward(self, x):
        out = self.mlp_net(x)
        return out

from torch.nn.modules.transformer import TransformerEncoderLayer

class transformer_FC(nn.Module):
    """ Self attention Layer
    """
    def __init__(self, num_layer, size_arg, dropout_rate):
        super(transformer_FC, self).__init__()
        self.size_setting = {"1024_small": [1024, 512, 256], "1024_big": [1024, 512, 384], "1024_min": [1024, 256,128], 
                        "dino": [384, 256, 128], "dino_b": [384, 300, 256], "dino_ss": [384, 64, 32], "dino_s": [384, 128, 64],
                        "hipt": [192, 128, 64], "hipt_att": [192,192,192]
                        }
        dmodel = self.size_setting[size_arg]
        in_dim = dmodel[0]    
        att = []
        for _ in range(num_layer):
            att.append(TransformerEncoderLayer(d_model=in_dim,nhead=8,dim_feedforward=in_dim*4,dropout=dropout_rate,activation="relu"))
        self.att = nn.Sequential(*att)

    def forward(self, x):
        if len(x.size()) == 2:
            x = x.unsqueeze(0)
        out = self.att(x)
        return out


from .ml_decoder import GroupFC, TransformerDecoderLayerOptimal, TransformerDecoderWithAttention
from .util_layers import MultiHeadedAttention, DecoderLayer, Decoder, PositionwiseFeedForward

class MLDecoder(nn.Module):
    """ the label decoder for Geneformer
            x:(patch, initial_num_features) -> (patch, d_decoder_embed)
            y:(num_classes, num_label_embed)
        Args: 
            num_label_embed
            d_decoder_embed
            initial_num_features: 
        Input: K, Q, V

        Output: logits
    """
    def __init__(self, num_classes, num_label_embed, d_decoder_embed=768,
                 initial_num_features=2048, zsl=0, num_layers_decoder=1, cls_token=False, decoder_dropout=0.5, decoder_nhead=8):
        super(MLDecoder, self).__init__()

        self.cls_token = cls_token
        # switching to 768 initial embeddings
        d_decoder_embed = 768 if d_decoder_embed < 0 else d_decoder_embed
        
        if initial_num_features != d_decoder_embed:
            embed_standard = nn.Linear(initial_num_features, d_decoder_embed)
            self.standard = True
        else:
            embed_standard = []
            self.standard = False

        # decoder
        decoder_dropout = decoder_dropout
        decoder_nhead = decoder_nhead
        num_layers_decoder = num_layers_decoder

        #TODO: usually the 'dim_feedforward' should be higher than the 'd_decoder_embed'
        # dim_feedforward = d_decoder_embed * 2  # the dimension of feed forward layer in the transformer decoder
        dim_feedforward = d_decoder_embed
        #TODO: use customized cross-attention layer
        # import copy
        # c = copy.deepcopy
        # attn = MultiHeadedAttention(nhead=8, d_model=d_decoder_embed)
        # ff = PositionwiseFeedForward(d_model=d_decoder_embed, d_ff=dim_feedforward,dropout=0.1)
        # self.decoder = Decoder(DecoderLayer(
        #                         d_model=d_decoder_embed, src_attn=c(attn), feed_forward=c(ff), dropout=0.1), 
        #                         N=num_layers_decoder)

        ######## old version ######
        layer_decode = TransformerDecoderLayerOptimal(d_model=d_decoder_embed,
                                                      dim_feedforward=dim_feedforward, dropout=decoder_dropout, nhead=decoder_nhead)

        self.decoder = TransformerDecoderWithAttention(layer_decode, num_layers=num_layers_decoder)
        ###########################

        self.decoder.embed_standart = embed_standard # initial feature -> d_decoder_embed

        self.zsl = zsl

        if self.zsl:
            if d_decoder_embed != 300:
                self.wordvec_proj = nn.Linear(300, d_decoder_embed)
            else:
                self.wordvec_proj = nn.Identity()
            self.decoder.duplicate_pooling = torch.nn.Parameter(torch.Tensor(d_decoder_embed, 1))
            self.decoder.duplicate_pooling_bias = torch.nn.Parameter(torch.Tensor(1))
            self.decoder.duplicate_factor = 1
        
        else:
            # group fully-connected
            self.decoder.num_classes = num_classes
            self.decoder.duplicate_factor = int(num_classes / num_label_embed + 0.999)
            self.decoder.duplicate_pooling = torch.nn.Parameter(
                torch.Tensor(num_label_embed, d_decoder_embed, self.decoder.duplicate_factor))
            self.decoder.duplicate_pooling_bias = torch.nn.Parameter(torch.Tensor(num_classes))
        
        torch.nn.init.xavier_normal_(self.decoder.duplicate_pooling)
        torch.nn.init.constant_(self.decoder.duplicate_pooling_bias, 0)
        
        self.decoder.group_fc = GroupFC(num_label_embed)
        
        self.train_wordvecs = None
        self.test_wordvecs = None

    def forward(self, x, y, attention_only=False):
        data_shape = len(x.shape)
        # x feature
        embedding_spatial = x
        if data_shape == 4:  # [bs,2048, 7,7]
            embedding_spatial = x.flatten(2).transpose(1, 2)
            bs = embedding_spatial[0]

        elif data_shape == 3:
            # [batch, patch, feature]            
            bs = embedding_spatial.shape[0]

        elif data_shape ==2:  # [patch, embedding]
            bs = 1

        if self.standard == True:
            embedding_spatial_standard = self.decoder.embed_standart(embedding_spatial)
            embedding_spatial_standard = torch.nn.functional.relu(embedding_spatial_standard, inplace=True)
        else:
            embedding_spatial_standard = embedding_spatial

        # y feature
        query_embed = y # [label, embedding]

        # tgt = query_embed.unsqueeze(1).repeat(1, bs, 1)
        # tgt = query_embed.unsqueeze(1).expand(-1, bs, -1)  # no allocation of memory with expand
        # h = self.decoder(tgt, embedding_spatial_standard.transpose(0, 1))  # [num_label_embed, batch, 768]
        # h = h.transpose(0, 1)

        """TODO: old torch version < 1.10 : 
            data need the dimension of batch size for MHA 
            input data shape -> (seq, batch, d_feature)
        """
        if data_shape == 3 or data_shape == 4:
            x_memory = embedding_spatial_standard.transpose(0, 1)
        else:
            # [patch, feature] -> [patch, batch, feature], which batch = 1
            x_memory = embedding_spatial_standard.unsqueeze(1)  # expand dim on axis 1

        tgt = query_embed.unsqueeze(1).repeat(1, bs, 1) # [clsNum, feature] -> [clsNum, batch, feature]

        # h = self.decoder(tgt, x_memory)  # [num_label_embed, batch, feature]
        # get attention 
        h, a_score = self.decoder(tgt, x_memory)  # [num_label_embed, batch, feature]
        h = h.transpose(0, 1) # [clsNum, batch, feature] -> [batch, clsNum, feature]

        if self.cls_token == True:
            cls_token = h[:,0,:]
            h = h[:,1:,:]
        else:
            cls_token = []

        out_extrap = torch.zeros(h.shape[0], h.shape[1], self.decoder.duplicate_factor, device=h.device, dtype=h.dtype)
        # out_extrap: [batch, clsNum, 1]
        self.decoder.group_fc(h, self.decoder.duplicate_pooling, out_extrap)        
        # duplicate_pooling: [clsNum, feature, group ratio]

        if not self.zsl:
            h_out = out_extrap.flatten(1)[:, :self.decoder.num_classes]
        else:
            h_out = out_extrap.flatten(1)
        
        h_out += self.decoder.duplicate_pooling_bias
        logits = h_out

        return logits, cls_token, a_score


class MCAT_Decoder(nn.Module):
    """ the MCAT decoder: concate the label embedding (y) with cross-attentioned embedding (h) before label-wise projection 

            x:(patch, initial_num_features) -> (patch, d_decoder_embed)
            y:(num_classes, num_label_embed)
        Args: 
            num_label_embed
            d_decoder_embed
            initial_num_features: 
        Input: K, Q, V

        Output: logits
    """
    def __init__(self, num_classes, num_label_embed, d_decoder_embed=768,
                 initial_num_features=2048, zsl=0, num_layers_decoder=1, cls_token=False):
        super(MCAT_Decoder, self).__init__()

        self.cls_token = cls_token
        # switching to 768 initial embeddings
        d_decoder_embed = 768 if d_decoder_embed < 0 else d_decoder_embed
        
        if initial_num_features != d_decoder_embed:
            embed_standard = nn.Linear(initial_num_features, d_decoder_embed)
            self.standard = True
        else:
            embed_standard = []
            self.standard = False

        # decoder
        decoder_dropout = 0.1
        num_layers_decoder = num_layers_decoder

        #TODO: usually the 'dim_feedforward' should be higher than the 'd_decoder_embed'
        # dim_feedforward = d_decoder_embed * 2  # the dimension of feed forward layer in the transformer decoder
        dim_feedforward = d_decoder_embed
        #TODO: use customized cross-attention layer
        # import copy
        # c = copy.deepcopy
        # attn = MultiHeadedAttention(nhead=8, d_model=d_decoder_embed)
        # ff = PositionwiseFeedForward(d_model=d_decoder_embed, d_ff=dim_feedforward,dropout=0.1)
        # self.decoder = Decoder(DecoderLayer(
        #                         d_model=d_decoder_embed, src_attn=c(attn), feed_forward=c(ff), dropout=0.1), 
        #                         N=num_layers_decoder)

        ######## old version ######
        layer_decode = TransformerDecoderLayerOptimal(d_model=d_decoder_embed,
                                                      dim_feedforward=dim_feedforward, dropout=decoder_dropout)

        self.decoder = TransformerDecoderWithAttention(layer_decode, num_layers=num_layers_decoder)
        ###########################

        self.decoder.embed_standart = embed_standard # initial feature -> d_decoder_embed

        self.zsl = zsl

        if self.zsl:
            if d_decoder_embed != 300:
                self.wordvec_proj = nn.Linear(300, d_decoder_embed)
            else:
                self.wordvec_proj = nn.Identity()
            self.decoder.duplicate_pooling = torch.nn.Parameter(torch.Tensor(d_decoder_embed, 1))
            self.decoder.duplicate_pooling_bias = torch.nn.Parameter(torch.Tensor(1))
            self.decoder.duplicate_factor = 1
        
        else:
            # group fully-connected
            self.decoder.num_classes = num_classes
            self.decoder.duplicate_factor = int(num_classes / num_label_embed + 0.999)
            """ due to the concatenation
            """
            self.decoder.duplicate_pooling = torch.nn.Parameter(
                torch.Tensor(num_label_embed, d_decoder_embed*2, self.decoder.duplicate_factor))
            self.decoder.duplicate_pooling_bias = torch.nn.Parameter(torch.Tensor(num_classes))
        
        torch.nn.init.xavier_normal_(self.decoder.duplicate_pooling)
        torch.nn.init.constant_(self.decoder.duplicate_pooling_bias, 0)
        
        # label-wise projection
        self.decoder.group_fc = GroupFC(num_label_embed)
        
        self.train_wordvecs = None
        self.test_wordvecs = None

    def forward(self, x, y, attention_only=False):
        data_shape = len(x.shape)
        # x feature
        embedding_spatial = x
        if data_shape == 4:  # [bs,2048, 7,7]
            embedding_spatial = x.flatten(2).transpose(1, 2)
            bs = embedding_spatial[0]

        elif data_shape == 3:
            # [batch, patch, feature]            
            bs = embedding_spatial.shape[0]

        elif data_shape ==2:  # [patch, embedding]
            bs = 1

        if self.standard == True:
            embedding_spatial_standard = self.decoder.embed_standart(embedding_spatial)
            embedding_spatial_standard = torch.nn.functional.relu(embedding_spatial_standard, inplace=True)
        else:
            embedding_spatial_standard = embedding_spatial

        # y feature
        query_embed = y # [label, embedding]

        # tgt = query_embed.unsqueeze(1).repeat(1, bs, 1)
        # tgt = query_embed.unsqueeze(1).expand(-1, bs, -1)  # no allocation of memory with expand
        # h = self.decoder(tgt, embedding_spatial_standard.transpose(0, 1))  # [num_label_embed, batch, 768]
        # h = h.transpose(0, 1)

        """TODO: old torch version < 1.10 : 
            data need the dimension of batch size for MHA 
            input data shape -> (seq, batch, d_feature)
        """
        if data_shape == 3 or data_shape == 4:
            x_memory = embedding_spatial_standard.transpose(0, 1)
        else:
            # [patch, feature] -> [patch, batch, feature], which batch = 1
            x_memory = embedding_spatial_standard.unsqueeze(1)  # expand dim on axis 1

        tgt = query_embed.unsqueeze(1).repeat(1, bs, 1) # [clsNum, feature] -> [clsNum, batch, feature]

        # h = self.decoder(tgt, x_memory)  # [num_label_embed, batch, feature]
        # get attention 
        h, a_score = self.decoder(tgt, x_memory)  # [num_label_embed, batch, feature]
        h = h.transpose(0, 1) # [clsNum, batch, feature] -> [batch, clsNum, feature]

        """ key variant: concatenate y after h
        """
        h = torch.cat([h, tgt.transpose(0, 1)],dim = -1) # [] + [] -> [batch, clsNum, 2 * feature]

        if self.cls_token == True:
            cls_token = h[:,0,:]
            h = h[:,1:,:]
        else:
            cls_token = []

        out_extrap = torch.zeros(h.shape[0], h.shape[1], self.decoder.duplicate_factor, device=h.device, dtype=h.dtype)
        # out_extrap: [batch, clsNum, 1]
        self.decoder.group_fc(h, self.decoder.duplicate_pooling, out_extrap)        
        # duplicate_pooling: [clsNum, feature, group ratio]

        if not self.zsl:
            h_out = out_extrap.flatten(1)[:, :self.decoder.num_classes]
        else:
            h_out = out_extrap.flatten(1)
        
        h_out += self.decoder.duplicate_pooling_bias
        logits = h_out

        return logits, cls_token, a_score


class Cross_Attn(nn.Module):
    """ modify for our task

        x:(patch, initial_num_features) -> (patch, d_decoder_embed)
        y:(num_classes, num_label_embed)
    
    Args: 
        num_label_embed
        d_decoder_embed
        initial_num_features: 
    
    Inputs: K~(K,d), Q~(P,d), V~(K,d)

    Output: (P, d) = (Q * K^T)*V
    """
    def __init__(self, num_classes, num_label_embed, d_decoder_embed=768,
                 initial_num_features=2048, zsl=0, num_layers_decoder=1, cls_token=False):
        super(Cross_Attn, self).__init__()
        embed_standard = []
        self.standard = False

        decoder_dropout = 0.1

        # dim setting
        dim_feedforward = d_decoder_embed


        # cross        
        layer_decode = TransformerDecoderLayerOptimal(d_model=d_decoder_embed,
                                                      dim_feedforward=dim_feedforward, dropout=decoder_dropout)
        self.decoder = TransformerDecoderWithAttention(layer_decode, num_layers=num_layers_decoder)


    def forward(self, q, kv, attention_only=False):
        x = q
        data_shape = len(x.shape)
        # x feature
        embedding_spatial = x
        if data_shape == 4:  # [bs,2048, 7,7]
            embedding_spatial = x.flatten(2).transpose(1, 2)
            bs = embedding_spatial[0]

        elif data_shape == 3:
            # [batch, patch, feature]            
            bs = embedding_spatial.shape[0]

        elif data_shape ==2:  # [patch, embedding]
            bs = 1

        embedding_spatial_standard = embedding_spatial

        if data_shape == 3 or data_shape == 4:
            query = embedding_spatial_standard.transpose(0, 1)
        else:
            # [patch, feature] -> [patch, batch, feature], which batch = 1
            query = embedding_spatial_standard.unsqueeze(1)  #

        tgt = kv.unsqueeze(1).repeat(1, bs, 1)  # [clsNum, batch, feature]

        h, a_score = self.decoder(query, tgt) #

        h = h.transpose(0, 1).squeeze(0) #  [batch, Patch, feature]

        # do not need cls token for query
        cls_token = []

        return h, cls_token, a_score

# TODO: sin-cos positional embedding
def position_encoding_init(n_position, d_pos_vec):
    ''' Init the sinusoid position encoding table '''

    # keep dim 0 for padding token position encoding zero vector
    position_enc = np.array([
        [pos / np.power(10000, 2*i/d_pos_vec) for i in range(d_pos_vec)]
        if pos != 0 else np.zeros(d_pos_vec) for pos in range(n_position)])

    position_enc[1:, 0::2] = np.sin(position_enc[1:, 0::2]) # dim 2i
    position_enc[1:, 1::2] = np.cos(position_enc[1:, 1::2]) # dim 2i+1
    return torch.from_numpy(position_enc).type(torch.FloatTensor)

# TODO: the modified transformer-based MIL
try:
    from .TransMIL import TransLayer, PPEG
except ImportError:
    TransLayer = None
    PPEG = None


def _require_transmil():
    if TransLayer is None or PPEG is None:
        raise ImportError(
            "nystrom_attention is required for TransMIL and SA_MIL visual extractors. "
            "Install nystrom_attention or choose agg_method='att', 'dense', 'self_att', "
            "'abmil', or 'hipt'."
        )

class TransMIL(nn.Module):
    def __init__(self, model_size, n_classes):
        super(TransMIL, self).__init__()
        _require_transmil()
        size = size_selection(model_size)

        self.pos_layer = PPEG(dim=size[1])
        self._fc1 = nn.Sequential(nn.Linear(size[0], size[1]), nn.ReLU())
        self.cls_token = nn.Parameter(torch.randn(1, 1, size[1]))
        self.n_classes = n_classes
        self.layer1 = TransLayer(dim=size[1])
        self.layer2 = TransLayer(dim=size[1])
        self.norm = nn.LayerNorm(size[1])
        self._fc2 = nn.Linear(size[1], self.n_classes)

        # PE
        d_word_vec = size[1]
        self.position_enc = torch.nn.Parameter(torch.Tensor(1, d_word_vec))
        # self.position_enc = nn.Embedding(
                    # n_position, d_word_vec, padding_idx=Constants.PAD)

        # self.position_enc.weight.data = position_encoding_init(n_position, d_word_vec)

    def forward(self, h):
        if len(h.size()) == 2:
            h = h.unsqueeze(0)
        h = self._fc1(h) #[B, n, 512]
        
        #---->pad
        H = h.shape[1]
        _H, _W = int(np.ceil(np.sqrt(H))), int(np.ceil(np.sqrt(H)))
        add_length = _H * _W - H
        h = torch.cat([h, h[:,:add_length,:]],dim = 1) #[B, N, 512]

        #---->cls_token
        B = h.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1).to(h.device)
        h = torch.cat((cls_tokens, h), dim=1)

        #---->Translayer x1
        h = self.layer1(h) #[B, N, 512]

        #---->PPEG
        h = self.pos_layer(h, _H, _W) #[B, N, 512]
        
        #---->Translayer x2
        h = self.layer2(h) #[B, N, 512]

        #---->cls_token
        h = self.norm(h)

        # TODO: old version
        h_feat, h_cls = h[:,1:], h[:,0]
        # remove padding patch
        # h_feat, h_cls = h[:,1:-add_length], h[:,0]

        #---->predict
        logits = self._fc2(h_cls) #[B, n_classes]
        Y_hat = torch.argmax(logits, dim=1)
        Y_prob = F.softmax(logits, dim = 1)
        
        results_dict = {'logits': logits, 'Y_prob': Y_prob, 'Y_hat': Y_hat, 'h_feat':h_feat, 'h_cls':h_cls}

        return results_dict

#
class SA_MIL(nn.Module):
    def __init__(self, model_size):
        super(SA_MIL, self).__init__()
        _require_transmil()
        size = size_selection(model_size)
        self._fc1 = nn.Sequential(nn.Linear(size[0], size[1]), nn.ReLU())
        self.layer1 = TransLayer(dim=size[1])
        self.layer2 = TransLayer(dim=size[1])
        self.norm = nn.LayerNorm(size[1])

    def forward(self, h):
        if len(h.size()) == 2:
            h = h.unsqueeze(0)
        h = self._fc1(h) #[B, n, 512]
        #---->Nys self-att layer x1
        h = self.layer1(h) #[B, N, 512]
        #---->Nys self-att layer x2
        h = self.layer2(h) #[B, N, 512]
        h = self.norm(h)

        h_feat = h 
        results_dict = {'h_feat':h_feat}

        return results_dict

# label decoder
class query2label(nn.Module):
    """ x_emb --------------> 
                               } --> transformer decoder/ dot product --> multi-label prediction
        label_emb --> GCN --> 
    """
    def __init__(self, num_classes, num_of_groups=-1, 
                model_size='1024_small',zsl=0, adj=None, dropout=True, 
                dropout_rate=0.5, k_sample=8, num_layers_decoder=1, gcn='forward', agg_method='att',num_layers_encoder=2, 
                region_cls=True, num_region=1,instance_loss_fn=None):
        super(query2label, self).__init__() 
        self.size_setting = {"1024_small": [1024, 512, 256], "1024_big": [1024, 512, 384], "1024_min": [1024, 256,128], 
                             "dino": [384, 256, 128], "dino_b": [384, 300, 256], "dino_ss": [384, 64, 32], "dino_s": [384, 128, 64],
                             "hipt": [192, 128, 64], "hipt_att": [192,192,192]
                             }

        size = self.size_setting[model_size]
        initial_d_features = size[1]
        d_decoder_embed = size[2]
        d_label_embed = d_decoder_embed * 3

        # patch aggregation
        if agg_method == 'att':
            self.att = Att_ML(size_arg=model_size, dropout=dropout, dropout_rate=dropout_rate)
        elif agg_method == 'tran':
            self.att = transformer_FC(num_layer=num_layers_encoder, size_arg=model_size, dropout_rate=dropout_rate)
        elif agg_method == 'cor_mil':
            self.att = TransMIL(model_size, n_classes=num_region)
        self.agg_method = agg_method

        # region classification
        self.region_cls = region_cls
        if self.region_cls == True and num_region != 1 and agg_method != 'cor_mil':
            model_dict = {"model":'wsl',  
            'n_classes': num_region, "size_arg": "hipt", 
            'k_sample': k_sample, 
            "dropout": dropout,
            'dropout_rate':dropout_rate}

            self.region_cls_module = CLAM_SB(**model_dict, instance_loss_fn=instance_loss_fn)
            self.alpha = torch.nn.Parameter(data=torch.ones(1, dtype=torch.float32) ,requires_grad=True)

        # label embedding
        """ d_decoder_embed: the dimension of decoder's input embedding: 128
            num_label_embed: the length of label embedding
        """
        num_label_embed = 100 if num_of_groups < 0 else num_of_groups
        if num_label_embed > num_classes:
            num_label_embed = num_classes

        # create non-learnable label queries
        if not zsl:            
            query_embed = nn.Embedding(num_label_embed, d_label_embed)
            query_embed.requires_grad_(False)
        else:
            query_embed = None
        self.query_embed = query_embed
        self.zsl = zsl

        if str(type(adj)) == "<class 'numpy.ndarray'>":
            self.adj = adj.astype(np.float32).to_tensor(dtype=torch.float32)
        else:
            self.adj = adj

        # graph aggregation
        self.gcn = GCN(nclass=d_decoder_embed,model_size=model_size,dropout=0.1)
        
        # TODO: add transformer-based graph network 
        # self.gcn = Graph_Tran(in_dim=d_label_embed, dropout_rate=0.1, num_layer=2, adj=adj)

        
        self.forward_net = [nn.Linear(d_label_embed, d_decoder_embed), nn.ReLU()]
        if dropout:
            self.forward_net.append(nn.Dropout(dropout_rate)) 
        self.forward_net = nn.Sequential(*self.forward_net)
        
        self.gamma = nn.Parameter(torch.ones(1))
        self.gcn_connect = gcn

        # multi-label classification
        self.ml_decoder = MLDecoder(num_classes=num_classes, num_label_embed=num_label_embed, d_decoder_embed=d_decoder_embed,
                 initial_num_features=initial_d_features, zsl=zsl,num_layers_decoder=num_layers_decoder)
                 # decoder embedding is the dimension of cross attention 
                 # initial_num_features is the embeeding dimension from x 
        
        initialize_weights(self)
    def compute_l1_loss(self, w):
        return torch.abs(w).sum()

    def forward(self, h, label=None, instance_eval=True, attention_only=False):
        # patch aggregation
        if self.agg_method == 'cor_mil':
            result = self.att(h)
            Y_region_logits = result['logits']
            Y_region_prob = result['Y_prob']
            Y_region_hat = result['Y_hat']
            x = result['h_feat']
            results_dict = {}
        
        else:
            x = self.att(h)

            # region classification
            if self.region_cls == True:
                Y_region_logits, Y_region_prob, Y_region_hat, A, results_dict = self.region_cls_module(x.squeeze(), label, instance_eval)
                #x = A.transpose(0,1) * x.squeeze()            
                x = x.squeeze() + self.alpha * A.transpose(0,1) * x.squeeze()  # (P, 1) .* (P, D)

            else:
                Y_region_logits = []
                Y_region_prob = []
                A = []
                results_dict ={}

        adj = self.adj
        # label embedding 
        if self.zsl:
            query_embed = torch.nn.functional.relu(self.wordvec_proj(self.query_embed))
        else:
            query_embed = self.query_embed.weight
        
        # graph
        if self.gcn_connect == 'forward':
            y = self.gcn(query_embed, adj)
        elif self.gcn_connect == 'bypass':
            y = self.forward_net(query_embed)
            b = self.gcn(query_embed, adj)
            y = y + self.gamma * b
        elif self.gcn_connect == 'abandon':
            y = self.forward_net(query_embed)

        # decoder: TODO:without sigmoid, should be added in the loss
        logits, _, a_score = self.ml_decoder(x, y)
        probs = torch.sigmoid(logits)
        lasso_loss = self.compute_l1_loss(probs)

        if attention_only == True:
            return a_score
        # 
        results_dict['a_score'] = a_score
        results_dict['lasso_loss'] = lasso_loss
        results_dict['Y_region_logits'] = Y_region_logits
        results_dict['Y_region_prob'] = Y_region_prob
        return logits, probs, results_dict
    
    def relocate(self):
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gcn = self.gcn.to(device)
        self.att = self.att.to(device)
        self.query_embed = self.query_embed.to(device)
        self.ml_decoder = self.ml_decoder.to(device)



""" graph transformer
"""
class Graph_Tran(nn.Module):
    """
        Graph Transformer

        Input:
            [N, d]

        Output:
            [N+1, d]  if virtual node
            [N, d] else
    """
    def __init__(self, in_dim, dropout_rate, num_layer, adj, if_cls_token=True):
        super(Graph_Tran, self).__init__()    
        # positional embedding 

        # multi-head self-attention
        mha = []
        for i in range(num_layer):
            #TODO: decrease the forward model
            mha.append(TransformerEncoderLayer(d_model=in_dim,nhead=8,dim_feedforward=in_dim,dropout=dropout_rate,activation="relu"))
            # mha.append(TransformerEncoderLayer(d_model=in_dim,nhead=8,dim_feedforward=in_dim*4,dropout=dropout_rate,activation="relu"))
        self.mha = nn.Sequential(*mha) 

        self.cls_token = nn.Parameter(torch.randn(1, in_dim))
        pos_enc_dim = round(adj.size()[0]/2)
        self.pe = self.positional_encoding(adj=adj, pos_enc_dim=pos_enc_dim, if_cls_token=if_cls_token)
        self.if_cls_token = if_cls_token

        self.pe_proj = nn.Linear(pos_enc_dim, in_dim, bias=True)

    def positional_encoding(self, adj, pos_enc_dim, if_cls_token=True):
        """
            Graph positional encoding v/ Laplacian eigenvectors
        """
        from scipy import sparse as sp
        # Laplacian
        # A = g.adjacency_matrix_scipy(return_edge_ids=False).astype(float)
        # N = sp.diags(dgl.backend.asnumpy(g.in_degrees()).clip(1) ** -0.5, dtype=float)
        # L = sp.eye(g.number_of_nodes()) - N * A * N
        if if_cls_token == True:
           # add virtual node 
           row = torch.ones(adj.size()[1]).unsqueeze(0)
           adj = torch.cat([row, adj], dim=0) 
           col = torch.ones(adj.size()[0]).unsqueeze(1)
           adj = torch.cat([col, adj], dim=1) 

        number_of_nodes = adj.size()[0]
        A = adj
        D, A = A.sum(axis=0) - torch.ones(number_of_nodes), A.numpy()
        N = sp.diags(D.numpy().clip(1) ** -0.5, dtype=float)
        L = sp.eye(number_of_nodes) - N * A * N
        

        # Eigenvectors with numpy
        # EigVal, EigVec = np.linalg.eig(L.toarray()) # np.linalg.eig: calculate eigen vector
        EigVal, EigVec = np.linalg.eig(L)
        idx = EigVal.argsort() # increasing order
        EigVal, EigVec = EigVal[idx], np.real(EigVec[:,idx])
        Pos_En = torch.from_numpy(EigVec[:,1:pos_enc_dim+1]).float()

        # g.ndata['pos_enc'] = torch.from_numpy(EigVec[:,1:pos_enc_dim+1]).float() 

        # # Eigenvectors with scipy
        # EigVal, EigVec = sp.linalg.eigs(L, k=pos_enc_dim+1, which='SR')
        # EigVec = EigVec[:, EigVal.argsort()] # increasing order
        # g.ndata['pos_enc'] = torch.from_numpy(np.abs(EigVec[:,1:pos_enc_dim+1])).float() 
        
        return Pos_En

    def forward(self, h):
        """ (num_token, emb) -> (num_token, emb')
        """
        device = h.device
        self.pe = self.pe.to(device)
        pe = self.pe_proj(self.pe)

        if self.if_cls_token:           
            h = torch.cat((self.cls_token, h), dim=0) # [N, d] -> [N+1, d]

        h = h + pe
        if len(h.size()) == 2:
            h = h.unsqueeze(0)
        out = self.mha(h)
        out = out.squeeze(0)
        return out

class transformer_decoder(nn.Module):
    """ x_emb ----------------------------> 
                                            } --> transformer decoder/ dot product --> multi-label prediction
        label_emb --> graph transformer --> 
    """
    def __init__(self, num_classes, num_of_groups=-1, 
                model_size='1024_small',zsl=0, adj=None, dropout=True, 
                dropout_rate=0.5, k_sample=8, num_layers_decoder=1, gcn='forward', agg_method='att',num_layers_encoder=2, 
                region_cls=True, num_region=1,instance_loss_fn=None,cls_token=True):
        super(transformer_decoder, self).__init__() 
        self.size_setting = {"1024_small": [1024, 512, 256], "1024_big": [1024, 512, 384], "1024_min": [1024, 256,128], 
                             "dino": [384, 256, 128], "dino_b": [384, 300, 256], "dino_ss": [384, 64, 32], "dino_s": [384, 128, 64],
                             "hipt": [192, 128, 64], "hipt_att": [192,192,192]
                             }

        size = self.size_setting[model_size]

        initial_d_features = size[1]   # x: 384 -> 256 
        d_decoder_embed = size[1]    #  256 
        d_label_embed = d_decoder_embed # 256

        # patch aggregation
        if agg_method == 'att':
            self.att = Att_ML(size_arg=model_size, dropout=dropout, dropout_rate=dropout_rate)
        elif agg_method == 'tran':
            self.att = transformer_FC(num_layer=num_layers_encoder, size_arg=model_size, dropout_rate=dropout_rate)
        elif agg_method == 'cor_mil':
            self.att = TransMIL(model_size, n_classes=num_region)
        self.agg_method = agg_method

        # region classification
        self.region_cls = region_cls
        if self.region_cls == True and num_region != 1 and agg_method != 'cor_mil':
            model_dict = {"model":'wsl',  
            'n_classes': num_region, "size_arg": "hipt", 
            'k_sample': k_sample, 
            "dropout": dropout,
            'dropout_rate':dropout_rate}

            self.region_cls_module = CLAM_SB(**model_dict, instance_loss_fn=instance_loss_fn)
            self.alpha = torch.nn.Parameter(data=torch.ones(1, dtype=torch.float32) ,requires_grad=True)

        # label embedding
        """ d_decoder_embed: the dimension of decoder's input embedding: 128
            num_label_embed: the length of label embedding
        """
        num_label_embed = 100 if num_of_groups < 0 else num_of_groups
        if num_label_embed > num_classes:
            num_label_embed = num_classes

        # create non-learnable label queries
        if not zsl:            
            query_embed = nn.Embedding(num_label_embed, d_label_embed)  # embeddin的weight已经是对每个label产生不同的索引
            query_embed.requires_grad_(False)
        else:
            query_embed = None
        

        self.query_embed = query_embed
        self.zsl = zsl

        if str(type(adj)) == "<class 'numpy.ndarray'>":
            self.adj = adj.astype(np.float32).to_tensor(dtype=torch.float32)
        else:
            self.adj = adj
        
        # TODO: add transformer-based graph network 
        self.gcn = Graph_Tran(in_dim=d_label_embed, dropout_rate=0.1, num_layer=2, adj=adj, if_cls_token=cls_token)        
        # self.gcn = Graph_Tran(in_dim=d_label_embed, dropout_rate=0.1, num_layer=3, adj=adj, if_cls_token=cls_token)        
        self.cls_token =cls_token
        if self.cls_token:
            self.fc = nn.Linear(d_label_embed, num_classes)

        self.forward_net = [nn.Linear(d_label_embed, d_decoder_embed), nn.ReLU()]
        if dropout:
            self.forward_net.append(nn.Dropout(dropout_rate)) 
        self.forward_net = nn.Sequential(*self.forward_net)
        
        self.gamma = nn.Parameter(torch.ones(1))
        self.gcn_connect = gcn

        # multi-label classification
        self.ml_decoder = MLDecoder(num_classes=num_classes, num_label_embed=num_label_embed, d_decoder_embed=d_decoder_embed,
                 initial_num_features=initial_d_features, zsl=zsl,num_layers_decoder=num_layers_decoder, cls_token=cls_token)
                 # decoder embedding is the dimension of cross attention 
                 # initial_num_features is the embeeding dimension from x 

        initialize_weights(self)

    def compute_l1_loss(self, w):
        return torch.abs(w).sum()

    def forward(self, h, label=None, instance_eval=True):
        # patch aggregation
        if self.agg_method == 'cor_mil':
            result = self.att(h)
            Y_region_logits = result['logits']
            Y_region_prob = result['Y_prob']
            Y_region_hat = result['Y_hat']
            x = result['h_feat']
            results_dict = {}
        
        else:
            x = self.att(h)

            # region classification
            if self.region_cls == True:
                Y_region_logits, Y_region_prob, Y_region_hat, A, results_dict = self.region_cls_module(x.squeeze(), label, instance_eval)
                #x = A.transpose(0,1) * x.squeeze()            
                x = x.squeeze() + self.alpha * A.transpose(0,1) * x.squeeze()  # (P, 1) .* (P, D)

            else:
                Y_region_logits = []
                Y_region_prob = []
                A = []
                results_dict ={}

        adj = self.adj
        # label embedding 
        if self.zsl:
            query_embed = torch.nn.functional.relu(self.wordvec_proj(self.query_embed))
        else:
            query_embed = self.query_embed.weight
        
        # graph
        if self.gcn_connect == 'forward':
            y = self.gcn(query_embed)
            
        # elif self.gcn_connect == 'bypass':
        #     y = self.forward_net(query_embed)
        #     b = self.gcn(query_embed, adj)
        #     y = y + self.gamma * b
        # elif self.gcn_connect == 'abandon':
        #     y = self.forward_net(query_embed)

        # decoder: TODO:without sigmoid, should be added in the loss
        logits, virtual_node, a_score = self.ml_decoder(x, y)
        if self.cls_token:
            logits = self.fc(virtual_node)

        probs = torch.sigmoid(logits)
        lasso_loss = self.compute_l1_loss(probs)

        results_dict['a_score'] = a_score
        results_dict['lasso_loss'] = lasso_loss
        results_dict['Y_region_logits'] = Y_region_logits
        results_dict['Y_region_prob'] = Y_region_prob
        return logits, probs, results_dict
    
    def relocate(self):
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gcn = self.gcn.to(device)
        self.att = self.att.to(device)
        self.query_embed = self.query_embed.to(device)
        self.ml_decoder = self.ml_decoder.to(device)


from .graphormer import GraphormerLayer, Graphormer, MultiHeadedAttention
class graphormer_decoder(nn.Module):
    """ x_emb -- patch embedd ---> 
                                    } --> transformer decoder/ dot product --> multi-label prediction
        label_emb --> graphormer --> 
    """
    def __init__(self, num_classes, num_of_groups=-1, 
                model_size='1024_small',zsl=0, adj=None, dropout=True, 
                dropout_rate=0.5, k_sample=8, num_layers_decoder=1, num_layers_encoder=2, 
                gcn='forward', agg_method='att', cls_token=True,
                region_cls=True, num_region=1, instance_loss_fn=None, 
                spatial_mask=None, pe=None,num_graph_layer=2, 
                word_embedd=None, learnable_embedd=False):
        super(graphormer_decoder, self).__init__() 

        size = size_selection(model_size)

        initial_d_features = size[1]   # x: 384 -> 256 
        d_decoder_embed = size[1]    #  256 
        d_label_embed = d_decoder_embed # 256


        # patch aggregation
        if agg_method == 'att':
            self.att = Att_ML(size_arg=model_size, dropout=dropout, dropout_rate=dropout_rate)
        elif agg_method == 'tran':
            self.att = transformer_FC(num_layer=num_layers_encoder, size_arg=model_size, dropout_rate=dropout_rate)
        elif agg_method == 'cor_mil':
            self.att = TransMIL(model_size, n_classes=num_region)
        self.agg_method = agg_method


        # region classification
        self.region_cls = region_cls
        if self.region_cls == True and num_region != 1 and agg_method != 'cor_mil':
            model_dict = {"model":'wsl',  
            'n_classes': num_region, "size_arg": "hipt", 
            'k_sample': k_sample, 
            "dropout": dropout,
            'dropout_rate':dropout_rate}

            self.region_cls_module = CLAM_SB(**model_dict, instance_loss_fn=instance_loss_fn)
            self.alpha = torch.nn.Parameter(data=torch.ones(1, dtype=torch.float32) ,requires_grad=True)

        # label embedding
        """ d_decoder_embed: the dimension of decoder's input embedding: 128
            num_label_embed: the length of label embedding
        """
        num_label_embed = 100 if num_of_groups < 0 else num_of_groups
        if num_label_embed > num_classes:
            num_label_embed = num_classes

        # create non-learnable label queries
        self.if_word_embedd = False
        if not zsl:            
            if word_embedd is None:            
                query_embed = nn.Embedding(num_label_embed, d_label_embed)  # embeddin的weight已经是对每个label产生不同的索引
                query_embed.requires_grad_(False)
            else:
                query_embed = word_embedd
                self.word_proj = nn.Linear(in_features= word_embedd.size()[-1], out_features=d_label_embed, bias=True)
                self.word_proj.requires_grad_(learnable_embedd)
                self.if_word_embedd = True 
            # matrix factorization??
            # if word_embedded is not None:
               # query_embedd = factorization(word_embedd, d_label_embedd)  # (labels, 768) -> (labels, label embedd) * (label embedd, 768)
        else:
            query_embed = None
        

        self.query_embed = query_embed
        self.zsl = zsl

        if str(type(adj)) == "<class 'numpy.ndarray'>":
            self.adj = adj.astype(np.float32).to_tensor(dtype=torch.float32)
        else:
            self.adj = adj
        
        # TODO: add graphormer and  GraphormerLayer
        
        graphlayer = GraphormerLayer(d_label_embed, nhead=8, dim_feedforward=d_label_embed*4, dropout=dropout_rate, activation="relu",
                 layer_norm_eps=1e-5)      
        self.gcn = Graphormer(graphlayer,d_label_embed, num_layer=num_graph_layer, adj=adj, spatial_mask=spatial_mask, pe=pe, if_cls_token=cls_token)        

        # cls token
        self.cls_token = cls_token
        if self.cls_token:
            self.fc = nn.Linear(d_label_embed, num_classes)

        self.d_decoder_embed = d_decoder_embed
        self.num_classes = num_classes

        # graph connect
        # self.forward_net = [nn.Linear(d_label_embed, d_decoder_embed), nn.ReLU()]
        # if dropout:
        #     self.forward_net.append(nn.Dropout(dropout_rate)) 
        # self.forward_net = nn.Sequential(*self.forward_net)
        # self.norm = nn.LayerNorm(d_label_embed, eps=1e-5)     

        # self.gamma = nn.Parameter(torch.ones(1))
        self.gcn_connect = gcn

        # self.self_attn = MultiHeadedAttention(d_model=d_label_embed, nhead=8, dropout=dropout_rate)
        self.self_attn = MultiHeadedAttention(d_model=d_label_embed, nhead=8, dropout=0.1)

        # multi-label classification
        self.ml_decoder = MLDecoder(num_classes=num_classes, num_label_embed=num_label_embed, d_decoder_embed=d_decoder_embed,
                 initial_num_features=initial_d_features, zsl=zsl,num_layers_decoder=num_layers_decoder, cls_token=cls_token)
                 # decoder embedding is the dimension of cross attention 
                 # initial_num_features is the embeeding dimension from x 

        initialize_weights(self)

    def compute_l1_loss(self, w):
        return torch.abs(w).sum()

    def forward(self, h, label=None, instance_eval=True):
        device = h.device
        self.query_embed = self.query_embed.to(device)
        # patch aggregation
        if self.agg_method == 'cor_mil':
            result = self.att(h)
            Y_region_logits = result['logits']
            Y_region_prob = result['Y_prob']
            Y_region_hat = result['Y_hat']
            x = result['h_feat']
            results_dict = {}
        
        else:
            x = self.att(h)

            # region classification
            if self.region_cls == True:
                Y_region_logits, Y_region_prob, Y_region_hat, A, results_dict = self.region_cls_module(x.squeeze(), label, instance_eval)
                #x = A.transpose(0,1) * x.squeeze()            
                x = x.squeeze() + self.alpha * A.transpose(0,1) * x.squeeze()  # (P, 1) .* (P, D)

            else:
                Y_region_logits = []
                Y_region_prob = []
                A = []
                results_dict ={}

        adj = self.adj
        # label embedding 
        if self.zsl:
            query_embed = torch.nn.functional.relu(self.wordvec_proj(self.query_embed))
        else:
            if self.if_word_embedd: 
                query_embed = self.word_proj(self.query_embed)
            else:
                query_embed = self.query_embed.weight


        results_dict['Y_init'] = self.query_embed
        results_dict['Y_emb'] = query_embed
        # graph
        if self.gcn_connect == 'forward':           
            y = self.gcn(query_embed)
            
        # elif self.gcn_connect == 'bypass':
        #     y = self.forward_net(query_embed)
        #     b = self.gcn(query_embed, adj)
        #     y = y + self.gamma * b
        elif self.gcn_connect == 'abandon':
            y = self.forward_net(query_embed)
        
        
        #
        # m = torch.normal(mean=0., std=1.0,size=(self.num_classes, self.d_decoder_embed)).to(device)
        # nn.init.xavier_normal_(m)
        # one = torch.matmul(torch.eye(self.num_classes).to(device), m)
        
        # one_emb_p = self.gcn(one)      
        
        # results_dict['one_emb_p'] = one_emb_p
        # results_dict['Y_emb_p'] = y

        # y = y.unsqueeze(1) # -> (token, batch, embed)
        # y, g_attn = self.self_attn(y,y,y) # -> (token, embed)
        # y = self.norm(y.squeeze(1))

        # decoder: TODO:without sigmoid, should be added in the loss
        logits, virtual_node, a_score = self.ml_decoder(x, y)
        if self.cls_token:
            logits = self.fc(virtual_node)

        probs = torch.sigmoid(logits)
        lasso_loss = self.compute_l1_loss(probs)

        g_attn = torch.normal(mean=0., std=1.0,size=(self.num_classes, self.d_decoder_embed)).to(device)
        results_dict['Y_attn'] = g_attn
        results_dict['a_score'] = a_score
        results_dict['lasso_loss'] = lasso_loss
        results_dict['Y_region_logits'] = Y_region_logits
        results_dict['Y_region_prob'] = Y_region_prob
        return logits, probs, results_dict
    
    def relocate(self):
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gcn = self.gcn.to(device)
        self.att = self.att.to(device)
        self.query_embed = self.query_embed.to(device)
        self.ml_decoder = self.ml_decoder.to(device)
        # self.forward_net = self.forward_net.to(device)
        self.word_proj = self.word_proj.to(device)

from .graphormer import GraphormerLayer, Graphormer
class geneformer_v2(nn.Module):
    def __init__(self, num_classes, num_of_groups=-1, 
            model_size='1024_small',zsl=0, adj=None, dropout=True, 
            dropout_rate=0.5, k_sample=8, num_layers_decoder=1, num_layers_encoder=2, 
            gcn='forward', graph_module='graphormer', 
            agg_method='att', cls_token=True,
            region_cls=True, num_region=1, instance_loss_fn=None, 
            spatial_mask=None, pe=None,num_graph_layer=2, 
            word_embedd=None,learnable_embedd=False,proj=False):
        super(geneformer_v2, self).__init__() 

        size = size_selection(model_size)

        # TODO: modify for ver.2
        initial_d_features = size[0]   # x: 384 -> 256 
        d_decoder_embed = size[0]    #  256 
        d_label_embed = size[0] # 256

        # project
        if proj == True:
            self.proj_x = [nn.Linear(size[0], size[0]), nn.Tanh()]
            self.proj_x = nn.Sequential(*self.proj_x)
        self.proj = proj

        # patch agg
        self.patch_agg(agg_method, model_size, dropout, dropout_rate, num_layers_encoder, num_region)

        # no region classification
        self.region_cls = False

        # label embedding
        self.label_embedding(d_label_embed, num_of_groups, 
                            num_classes, zsl, 
                            word_embedd, learnable_embedd)

        # graph module
        self.graph_module(d_decoder_embed, d_label_embed, 
                    graph_module, num_graph_layer, model_size, 
                    adj, spatial_mask, pe, cls_token)
        
        # cls token
        self.cls_token = cls_token
        # if self.cls_token:
            # self.fc = nn.Linear(d_label_embed, num_classes)

        # cross-attention decoder
        self.ml_decoder = Cross_Attn(num_classes=num_classes, num_label_embed=self.num_label_embed, d_decoder_embed=d_decoder_embed,
                    initial_num_features=initial_d_features, zsl=zsl,num_layers_decoder=num_layers_decoder, cls_token=cls_token)

        # classifier
        self.fc = nn.Linear(size[1], num_classes)

    def label_embedding(self, d_label_embed, num_of_groups, 
                        num_classes, zsl, 
                        word_embedd, learnable_embedd):        
        num_label_embed = 100 if num_of_groups < 0 else num_of_groups
        if num_label_embed > num_classes:
            num_label_embed = num_classes
        # create non-learnable label queries
        self.if_word_embedd = False
        if not zsl:            
            if word_embedd is None:            
                query_embed = nn.Embedding(num_label_embed, d_label_embed)  # embeddin的weight已经是对每个label产生不同的索引
                query_embed.requires_grad_(False)
            else:
                query_embed = word_embedd
                self.word_proj = nn.Linear(in_features= word_embedd.size()[-1], out_features=d_label_embed, bias=True)
                self.word_proj.requires_grad_(learnable_embedd)
                self.if_word_embedd = True 
        else:
            query_embed = None
        self.query_embed = query_embed
        self.zsl = zsl
        self.num_label_embed = num_label_embed

    def patch_agg(self, agg_method, model_size, dropout, dropout_rate, num_layers_encoder, num_region):
        # patch aggregation
        if agg_method == 'att':
            self.att = Att_ML(size_arg=model_size, dropout=dropout, dropout_rate=dropout_rate)
        elif agg_method == 'tran':
            self.att = transformer_FC(num_layer=num_layers_encoder, size_arg=model_size, dropout_rate=dropout_rate)
        elif agg_method == 'cor_mil':
            self.att = TransMIL(model_size, n_classes=num_region)
        self.agg_method = agg_method


    def graph_module(self, d_decoder_embed, d_label_embed, 
                    graph_module, num_graph_layer, model_size, 
                    adj, spatial_mask, pe, cls_token):

        if str(type(adj)) == "<class 'numpy.ndarray'>":
            self.adj = adj.astype(np.float32).to_tensor(dtype=torch.float32)
        else:
            self.adj = adj
        
        # TODO: add graphormer and  GraphormerLayer
        if graph_module == 'graphormer' or graph_module =='geneformer_v2':
            graphlayer = GraphormerLayer(d_label_embed, nhead=8, dim_feedforward=d_label_embed, dropout=0.1, activation="relu",
                    layer_norm_eps=1e-5)      
            self.gcn = Graphormer(graphlayer,d_label_embed, num_layer=num_graph_layer, adj=adj, spatial_mask=spatial_mask, pe=pe, if_cls_token=cls_token)         

        elif graph_module == 'gcn':
            # TODO:
            self.gcn = GCN(nclass=d_decoder_embed,model_size=model_size,dropout=0.1)
        
        elif graph_module == 'transformer':
            self.gcn = Graph_Tran(in_dim=d_label_embed, dropout_rate=0.1, num_layer=num_graph_layer, adj=adj, if_cls_token=cls_token)
        
    def compute_l1_loss(self, w):
        return torch.abs(w).sum()
    def forward(self, x, label=None, instance_eval=True):
        """ x -> proj ->  
                           | decoder | -> patch aggregation -> classifier ->
            y -> graph ->
        """
        # data project
        if self.proj:
            x = self.proj_x(x)


        # multi-task        
        Y_region_logits = []
        Y_region_prob = []

        # label embed
        if self.zsl:
            query_embed = torch.nn.functional.relu(self.wordvec_proj(self.query_embed))
        else:
            if self.if_word_embedd: 
                query_embed = self.word_proj(self.query_embed)
            else:
                query_embed = self.query_embed.weight

        y = self.gcn(query_embed)

        # decoder
        h, virtual_node, a_score = self.ml_decoder(x, y)

        # patch aggregation
        if self.agg_method == 'cor_mil':
            result = self.att(h)
            x = result['h_feat']  # bs, patch, d
            cls = result['h_cls']
            results_dict = {}

            if self.cls_token:
                logits = self.fc(cls)
            else:
                logits = self.fc(x.mean(dim=1))

        probs = torch.sigmoid(logits)
        lasso_loss = self.compute_l1_loss(probs)

        results_dict['a_score'] = a_score
        results_dict['lasso_loss'] = lasso_loss
        results_dict['Y_region_logits'] = Y_region_logits
        results_dict['Y_region_prob'] = Y_region_prob

        return logits, probs, results_dict

    def relocate(self):
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gcn = self.gcn.to(device)
        self.att = self.att.to(device)
        self.query_embed = self.query_embed.to(device)
        self.ml_decoder = self.ml_decoder.to(device)
