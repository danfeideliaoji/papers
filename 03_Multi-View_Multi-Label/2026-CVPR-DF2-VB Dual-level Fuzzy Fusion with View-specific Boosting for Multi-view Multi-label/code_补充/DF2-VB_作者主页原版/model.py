import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn.init as Init

class Tester(nn.Module):
    def __init__(self, num_views, dims, dim_hidden, dim_comm, num_classes, device, n_rule=7):
        super(Tester, self).__init__()
        self.num_views = num_views
        self.dim_hidden = dim_hidden
        self.dim_comm = dim_comm
        self.dim_comp = dim_comm
        self.num_classes = num_classes
        self.device = device

        self.n_rule = n_rule
        self.fls = Fuzzifier(self.dim_hidden, n_rules=self.n_rule).to(self.device)
        self.EncoderLists = nn.ModuleList([EncoderList(dims[i], self.dim_hidden) for i in range(self.num_views)])
        self.DecoderLists = nn.ModuleList([DecoderList(self.dim_hidden, dims[i]) for i in range(self.num_views)])
        self.ClassifierLists = ClassifierList(self.dim_hidden, self.num_classes)
        self.CommonalityExtractorLists =\
            nn.ModuleList([CommonalityExtractorList(self.dim_hidden, self.dim_comm) for _ in range(self.num_views)])

        self.ComplementarityExtractorLists = \
            nn.ModuleList([ComplementarityExtractorList(self.dim_hidden, self.dim_comp) for _ in range(self.num_views)])

        self.ComplementarityClassifierLists = \
            nn.ModuleList([ClassifierList(self.dim_comp + self.dim_comm, self.num_classes) for _ in range(self.num_views)])

        self.FusedComplementarityClassifierLists = \
            nn.ModuleList([ClassifierList(self.dim_comp + self.dim_comm, self.num_classes) for _ in range(self.num_views)])

        self.FusionNet = nn.ModuleList([CommonalityComplementarityFusionList(self.dim_comm, self.dim_comp) for _ in range(self.num_views)])
        self.BN_comm = nn.BatchNorm1d(self.dim_comm)
        self.BN_comp = nn.BatchNorm1d(self.dim_comp)
        self.BN_fuse = nn.BatchNorm1d(self.dim_comp)

    def forward(self, X, instance=True, prototype=True):
        # get evidence
        hiddens = dict()
        restore_X = dict()
        for v in range(self.num_views):
            hiddens[v] = self.EncoderLists[v](X[v])
            restore_X[v] = self.DecoderLists[v](hiddens[v])

        return hiddens, restore_X

    def Extra_Commonality(self, hiddens):
        commons = dict()
        for v in range(self.num_views):
            commons[v] = self.CommonalityExtractorLists[v](hiddens[v])
        return commons

    def Extra_Complementarity(self, hiddens):
        comples = dict()
        for v in range(self.num_views):
            comples[v] = self.ComplementarityExtractorLists[v](hiddens[v])
        return comples

    def Fuse_Commonality(self, commons, fz_fusion=False):

        fusion_result = 0

        if fz_fusion is False:
            for v in range(self.num_views):
                fusion_result += commons[v]
            return self.BN_comm(F.normalize(fusion_result / self.num_views, dim=-1))
        else:
            for v in range(self.num_views):
                fz_degree, x = self.fls.fuzzify(commons[v], comb='max')
                fusion_result += x
            return self.BN_comm(F.normalize(fusion_result, dim=-1))

    def ReconstrctionLoss(self, X, restore_X):
        recon_loss = 0
        for v in range(self.num_views):
            recon1 = F.mse_loss(X[v], restore_X[v])
            recon_loss += recon1
        return recon_loss / self.num_views

    def Fuse_Commonality_Complementarity(self, fused_commons, comples, mlp=False):
        fusion_comles = dict()
        view_num = len(comples)
        if mlp is True:
            for v in range(view_num):
                fusion_comles[v] = self.FusionNet[v](fused_commons, comples[v])
        else:  # cat
            for v in range(view_num):
                fusion_comles[v] = torch.cat((fused_commons, comples[v]), dim=-1)
        return fusion_comles

    def DecomposeLoss(self, comms, comps, fused_comm, vector=True, tau=2):

        f = lambda x: torch.exp(x / tau)
        comms_pooling = dict()
        comps_pooling = dict()

        if vector is True:  # 利用向量进行对比损失计算
            for v in range(self.num_views):
                comms_pooling[v] = torch.mean(comms[v], dim=0, keepdim=True)
                comps_pooling[v] = torch.mean(comps[v], dim=0, keepdim=True)
            fused_comm_pooling = torch.mean(fused_comm, dim=0, keepdim=True)

            # contrast in pooling vectors
            pos_sim = 0
            neg_sim = 0
            for v in range(self.num_views):
                pos_sim += f(sim(comms_pooling[v], fused_comm_pooling))
                neg_sim += f(sim(comps_pooling[v], fused_comm_pooling))
                for u in range(v, self.num_views):
                    pos_sim += f(sim(comms_pooling[v], comms_pooling[u]))
                    neg_sim += f(sim(comms_pooling[v], comps_pooling[u]))
                    neg_sim += f(sim(comps_pooling[v], comps_pooling[u]))
            return -torch.log(pos_sim / (pos_sim + neg_sim))

        else:  # matrix
            for v in range(self.num_views):
                # print(comms[v].shape)
                comms_pooling[v] = torch.mean(comms[v], dim=0, keepdim=True)
                # print(comms_pooling[v].shape)
                comps_pooling[v] = torch.mean(comps[v], dim=0, keepdim=True)
            fused_comm_pooling = torch.mean(fused_comm, dim=0, keepdim=True)

            # contrast in pooling vectors
            pos_sim = 0
            neg_sim = 0
            for v in range(self.num_views):
                pos_sim += f(sim(comms[v], fused_comm))
                neg_sim += f(sim(comps[v], fused_comm))
                neg_sim += f(sim(comms[v], comps[v]))
                for u in range(v, self.num_views):
                    pos_sim += f(sim(comms[v], comms[u]))
                    neg_sim += f(sim(comms[v], comps[u]))
                    neg_sim += f(sim(comps[v], comps[u]))

            return torch.mean(-torch.log(pos_sim.diag() / torch.sum((pos_sim + neg_sim), dim=-1)))

    # Calculate the logits for each complementarity feature
    def Complementarity_Classify(self, compl, comb=1):
        view_num = len(compl)

        if comb == 1:  # boosting
            pred_list = dict()
            for v in range(view_num):
                pred_list[v] = self.ComplementarityClassifierLists[v](compl[v])
            return pred_list
        elif comb == 2:  # average prediction with multiple classifiers
            cls_list = dict()
            fusion_cls = 0
            for v in range(view_num):
                cls_list[v] = self.FusedComplementarityClassifierLists[v](compl[v])
                fusion_cls += cls_list[v]
            return fusion_cls / view_num
        else:  # average prediction with a shared classifier
            return self.FusedComplementarityClassifierLists(compl[0])

    # for emotion
    def AdaboostComplementarity(self, logits, labels, thrs, iteration=2, trained_weight_cls=None, num_sample=None):
        eps = 1e-6
        # eps = 0
        view_num = len(logits)

        logits_dict = dict()
        num_classes = self.num_classes
        B = labels.shape[0]
        batch_size = logits[0].shape[0]

        if num_sample is None:
            weight_samples = torch.ones(batch_size, num_classes).to(labels.device) / B
        else:
            weight_samples = torch.ones(batch_size, num_classes).to(labels.device) / num_sample

        weight_cls_list = []

        if num_sample is None:
            fusion_cls = torch.zeros(B, num_classes).to(labels.device)
        else:
            fusion_cls = torch.zeros(num_sample, num_classes).to(labels.device)

        if trained_weight_cls is None:
            for i in range(iteration):
                weight_cls_list = []
                for v in range(view_num):
                    logits_dict[v] = torch.sigmoid(logits[v]) > thrs
                    wrong_pred = logits_dict[v] != labels
                    err_cls = torch.sum(wrong_pred * weight_samples, dim=0)
                    err_cls = err_cls / torch.sum(weight_samples, dim=0)

                    weight_cls = 0.5 * torch.log((1 - err_cls + eps) / (err_cls + eps))  # shape [V, C]
                    weight_cls = weight_cls * (weight_cls > 0) + eps
                    weight_cls_list.append(weight_cls)

                    weight_sum = torch.sum(weight_samples, dim=0)
                    ratio = weight_samples / weight_sum
                    weight_samples = ratio * torch.exp(-weight_cls * labels * logits_dict[v])

            weight_tensor = torch.stack(weight_cls_list, dim=0)
            weight_tensor = torch.softmax(weight_tensor, dim=0)
            for v in range(view_num):
                fusion_cls += logits[v] * weight_tensor[v]  # broadcast over batch

            return fusion_cls, weight_tensor

        else:
            weight_tensor = trained_weight_cls
            for v in range(view_num):
                fusion_cls += logits[v] * weight_tensor[v]  # broadcast over batch
            return fusion_cls.detach()

def sim(z1, z2):
    z1 = F.normalize(z1)
    z2 = F.normalize(z2)
    return torch.mm(z1, z2.t())


class CommonalityComplementarityFusionList(nn.Module):
    def __init__(self, dim_comm, dim_comp):
        super(CommonalityComplementarityFusionList, self).__init__()
        dim_list = [dim_comm + dim_comp, int(dim_comm * 1.5)]
        # dim_list = [dim_comm + dim_comp]
        self.num_layers = len(dim_list)
        self.net = nn.ModuleList()
        for i in range(self.num_layers - 1):
            self.net.append(nn.Linear(dim_list[i], dim_list[i + 1]))
            self.net.append(nn.ReLU())
        self.net.append(nn.Linear(dim_list[self.num_layers - 1], dim_comm))

        self.BN = nn.BatchNorm1d(dim_comm)

    def forward(self, commons, comples):

        fusion = torch.cat((commons, comples), dim=-1)

        # print(fusion.shape)
        h = self.net[0](fusion)
        for i in range(1, len(self.net)):
            h = self.net[i](h)

        h = F.normalize(h, p=2.0)
        final_fusion = self.BN(h)
        return final_fusion


class EncoderList(nn.Module):
    def __init__(self, dims, dim_hidden):
        super(EncoderList, self).__init__()
        dim_list = [dims.item(), int(1.4 * dim_hidden), int(1.2 * dim_hidden)]
        self.BN0 = nn.BatchNorm1d(dim_hidden)
        self.BN = nn.BatchNorm1d(dim_hidden)
        self.num_layers = len(dim_list)
        self.net = nn.ModuleList()
        for i in range(self.num_layers - 1):
            self.net.append(nn.Linear(dim_list[i], dim_list[i + 1]))
            self.net.append(nn.ReLU())
            self.BN0 = nn.BatchNorm1d(dim_hidden)
            self.net.append(nn.Dropout(0.05))  # 0.05
        self.net.append(nn.Linear(dim_list[self.num_layers - 1], dim_hidden))

    def forward(self, x):
        h = self.net[0](x)
        for i in range(1, len(self.net)):
            h = self.net[i](h)
        h = F.normalize(h, p=2.0)
        h = self.BN(h)
        return h


class DecoderList(nn.Module):
    def __init__(self, dim_hidden, dims):
        super(DecoderList, self).__init__()
        dim_list = [dim_hidden, int(0.6 * dims.item()), int(0.8 * dims.item())]
        self.num_layers = len(dim_list)
        self.net = nn.ModuleList()
        for i in range(self.num_layers - 1):
            self.net.append(nn.Linear(dim_list[i], dim_list[i + 1]))
            self.net.append(nn.ReLU())
        self.net.append(nn.Linear(dim_list[self.num_layers - 1], dims.item()))

    def forward(self, x):
        h = self.net[0](x)
        for i in range(1, len(self.net)):
            h = self.net[i](h)
        return h


class ClassifierList(nn.Module):
    def __init__(self, dims, num_classes):
        super(ClassifierList, self).__init__()
        # self.num_layers = len(dims)
        self.net = nn.ModuleList()

        ratio = 1
        dim_list = [dims, int(ratio * dims)]
        self.num_layers = len(dim_list)
        for i in range(self.num_layers - 1):
            self.net.append(nn.Linear(dim_list[i], dim_list[i + 1]))
            self.net.append(nn.ReLU())
        self.net.append(nn.Linear(int(ratio*dims), num_classes))

    def forward(self, x):
        # source code
        h = self.net[0](x)
        for i in range(1, len(self.net)):
            h = self.net[i](h)
        return h

    def classify(self, x):
        view_num = len(x)

        pred = 0
        h = dict()

        for i in range(view_num):
            h[i] = self.net[0](x[i])
            for j in range(1, len(self.net)):
                h[i] = self.net[j](h[i])
            pred += h[i]
        pred /= view_num

        return pred

    def fusion_classify(self, x):

        pred = self.net(x)
        return pred


class CommonalityExtractorList(nn.Module):
    def __init__(self, dim_hidden, dim_com):
        super(CommonalityExtractorList, self).__init__()
        dim_list = [dim_hidden]
        self.num_layers = len(dim_list)
        self.net = nn.ModuleList()
        self.BN = nn.BatchNorm1d(dim_com)
        for i in range(self.num_layers - 1):
            self.net.append(nn.Linear(dim_list[i], dim_list[i + 1]))
            self.net.append(nn.ReLU())
        self.net.append(nn.Linear(dim_list[self.num_layers - 1], dim_com))

    def forward(self, x):
        h = self.net[0](x)
        for i in range(1, len(self.net)):
            h = self.net[i](h)
        h = self.BN(h)
        return h


class ComplementarityExtractorList(nn.Module):
    def __init__(self, dim_hidden, dim_com):
        super(ComplementarityExtractorList, self).__init__()
        dim_list = [dim_hidden]
        self.num_layers = len(dim_list)
        self.net = nn.ModuleList()
        for i in range(self.num_layers - 1):
            self.net.append(nn.Linear(dim_list[i], dim_list[i + 1]))
            self.net.append(nn.ReLU())
            # self.net.append(nn.PReLU())
            # self.net.append(nn.Dropout(0.1))
        self.net.append(nn.Linear(dim_list[self.num_layers - 1], dim_com))

    def forward(self, x):
        h = self.net[0](x)
        for i in range(1, len(self.net)):
            h = self.net[i](h)
        h = F.normalize(h, p=2.0)
        return h


def compute_corr(x1, x2):
    # Subtract the mean
    x1_mean = torch.mean(x1, 0, True)
    x1 = x1 - x1_mean
    x2_mean = torch.mean(x2, 0, True)
    x2 = x2 - x2_mean

    # Compute the cross correlation
    sigma1 = torch.sqrt(torch.mean(x1.pow(2)))
    sigma2 = torch.sqrt(torch.mean(x2.pow(2)))
    corr = torch.abs(torch.mean(x1*x2))/(sigma1*sigma2)

    return corr


class Fuzzifier(nn.Module):
    def __init__(self, in_dim, n_rules, fz=True):
        super(Fuzzifier, self).__init__()
        self.in_dim = in_dim
        self.n_rules = n_rules
        self.fz = fz  # whether to do fuzzifying operation
        self.eps = 1e-6

        self.build_model()

    def build_model(self):
        if self.fz == True:  # create antecedent parameters

            self.Cs = nn.Parameter(torch.FloatTensor(size=(self.in_dim, self.n_rules)), requires_grad=True)
            Init.normal_(self.Cs, mean=0, std=1)
            self.Vs = nn.Parameter(torch.FloatTensor(size=self.Cs.size()), requires_grad=True)
            Init.uniform_(self.Vs, 0, 1)

    # This process has calculated the fuzzy boundary
    def fuzzify(self, features, comb='max'):

        fz_degree = -(features.unsqueeze(dim=2) - self.Cs) ** 2 / ((2 * self.Vs ** 2) + self.eps)

        if self.fz == True:
            fz_degree = torch.exp(fz_degree)
            if comb == 'max':
                weighted_fz_degree = torch.max(fz_degree, dim=2)[0]
            elif comb == 'min':
                weighted_fz_degree = torch.min(fz_degree, dim=2)[0]
            elif comb == 'weight':
                norm_fz_weight = F.normalize(self.fz_weight, p=1, dim=1)
                weighted_fz_degree = torch.sum(torch.mul(fz_degree, norm_fz_weight), dim=2)
            else:  # mean
                weighted_fz_degree = torch.mean(fz_degree, dim=2)[0]
        weighted_fz_degree = torch.nan_to_num(weighted_fz_degree, nan=1e-6) + 1e-6

        fz_features = torch.mul(features, weighted_fz_degree)

        return weighted_fz_degree, fz_features