import math 
import torch
import torch.nn as nn
import torch.nn.functional as F

nINF = -100

class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=True, reduction='none'):
        super(AsymmetricLoss, self).__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps
        self.reduction = reduction

    def forward(self, x, y):
        """"
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """

        # Calculating Probabilities
        x_sigmoid = torch.sigmoid(x)
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        # Basic CE calculation
        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))
        loss = los_pos + los_neg

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            pt0 = xs_pos * y
            pt1 = xs_neg * (1 - y)  # pt = p if t > 0 else 1-p
            pt = pt0 + pt1
            one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1 - y)
            one_sided_w = torch.pow(1 - pt, one_sided_gamma)
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            loss *= one_sided_w

        if self.reduction == 'sum':
            loss = -loss.sum()
        elif self.reduction == 'none':
            loss = -loss

        return loss


 
## asl for unlabeled data   
class ASL_UL(nn.Module):

    def __init__(self, margin=0.0, reduce=None, size_average=None):
        super(ASL_UL, self).__init__()

        self.margin = margin
        self.reduce = reduce
        self.size_average = size_average
        
    def forward(self, input, labels):

        input, labels = input.float(), labels.float()

        preds = torch.sigmoid(input)
        # preds = input
        loss_mtx = torch.zeros_like(input)
        observed_labels = labels
        gamma1 = 0
        gamma2 = 4
        
        m= 0.05
        negtive_margin = torch.clamp(preds[observed_labels == -1] - m, min=0)
        loss_mtx[observed_labels == 1] = torch.pow((1 - preds[observed_labels == 1]), gamma1) * neg_log(preds[observed_labels == 1])
        loss_mtx[observed_labels == 0] = 0  #torch.pow(negtive_margin , gamma2) * neg_log(1.0 - negtive_margin)
        loss_mtx[observed_labels == -1] = torch.pow(negtive_margin , gamma2) * neg_log(1.0 - negtive_margin)

        mask = observed_labels != 0
        if torch.sum(mask) == 0:
            temp = torch.tensor(0.0).cuda()
        else:
            temp = loss_mtx[mask]
        return temp

## asl for unlabeled data   
class ASL_UL_weight(nn.Module):

    def __init__(self, margin=0.0, reduce=None, size_average=None):
        super(ASL_UL_weight, self).__init__()

        self.margin = margin

        self.reduce = reduce
        self.size_average = size_average


    def forward(self, input, labels, weights=None):

        input, labels = input.float(), labels.float()

        preds = torch.sigmoid(input)
        # preds = input
        loss_mtx = torch.zeros_like(input)
        observed_labels = labels
        gamma1 = 0
        gamma2 = 4
        m= 0.05
        
        negtive_margin = torch.clamp(preds[observed_labels == -1] - m, min=0)
        loss_mtx[observed_labels == 1] = torch.pow((1 - preds[observed_labels == 1]), gamma1) * neg_log(preds[observed_labels == 1])
        loss_mtx[observed_labels == 0] = 0  
        loss_mtx[observed_labels == -1] = torch.pow(negtive_margin , gamma2) * neg_log(1.0 - negtive_margin)
        loss_mtx = loss_mtx * weights

        mask = observed_labels != 0
        if torch.sum(mask) == 0:
            temp = torch.tensor(0.0).cuda()
        else:
            temp = loss_mtx[mask]
        return temp
 
## asl for unlabeled data   
class ASL_UL_softlabel(nn.Module):

    def __init__(self, margin=0.0, reduce=None, size_average=None):
        super(ASL_UL_softlabel, self).__init__()

        self.margin = margin

        self.reduce = reduce
        self.size_average = size_average


    def forward(self, input, weights=None):

        input = input.float()

        preds = torch.sigmoid(input)
        # preds = input
        loss_mtx_pos = torch.zeros_like(input)
        loss_mtx_neg = torch.zeros_like(input)
        gamma1 = 0
        gamma2 = 4
        m= 0.05
        
        negtive_margin = torch.clamp(preds - m, min=0)
        loss_mtx_pos = torch.pow((1 - preds), gamma1) * neg_log(preds)
        loss_mtx_neg = torch.pow(negtive_margin , gamma2) * neg_log(1.0 - negtive_margin)
        
        loss_mtx = loss_mtx_pos * weights + loss_mtx_neg * (1 - weights)

        return loss_mtx
 


class TwoWayLoss_class_mine(nn.Module):
    def __init__(self, Tp=1., Tn=1.):
        super(TwoWayLoss_class_mine, self).__init__()
        self.Tp = Tp
        self.Tn = Tn

    def forward(self, x, y):
        class_mask = (y > 0).any(dim=0)
        sample_mask = (y > 0).any(dim=1)


        # Calculate hard positive/negative logits
        pmask = y.masked_fill(y <= 0, nINF).masked_fill(y > 0, float(0.0))
        plogit_class = torch.logsumexp(-x/self.Tp + pmask, dim=0).mul(self.Tp)[class_mask]
        plogit_sample = torch.logsumexp(-x/self.Tp + pmask, dim=1).mul(self.Tp)[sample_mask]
    
        nmask = y.masked_fill(y != 0, nINF).masked_fill(y == 0, float(0.0))
        nlogit_class = torch.logsumexp(x/self.Tn + nmask, dim=0).mul(self.Tn)[class_mask]
        nlogit_sample = torch.logsumexp(x/self.Tn + nmask, dim=1).mul(self.Tn)[sample_mask]


        class_loss = torch.nn.functional.softplus(nlogit_class).sum() \
        + torch.nn.functional.softplus(plogit_class).sum()
        # sample_loss = torch.nn.functional.softplus(nlogit_sample + plogit_sample).sum()
        # return sample_loss
        return class_loss



class AsymmetricLossOptimized(nn.Module):
    ''' Notice - optimized version, minimizes memory allocation and gpu uploading,
    favors inplace operations'''

    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=False):
        super(AsymmetricLossOptimized, self).__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps

        # prevent memory allocation and gpu uploading every iteration, and encourages inplace operations
        self.targets = self.anti_targets = self.xs_pos = self.xs_neg = self.asymmetric_w = self.loss = None

    def forward(self, x, y):
        """"
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """

        self.targets = y
        self.anti_targets = 1 - y

        # Calculating Probabilities
        self.xs_pos = torch.sigmoid(x)
        self.xs_neg = 1.0 - self.xs_pos

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            self.xs_neg.add_(self.clip).clamp_(max=1)

        # Basic CE calculation
        self.loss = self.targets * torch.log(self.xs_pos.clamp(min=self.eps))
        self.loss.add_(self.anti_targets * torch.log(self.xs_neg.clamp(min=self.eps)))

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            self.xs_pos = self.xs_pos * self.targets
            self.xs_neg = self.xs_neg * self.anti_targets
            self.asymmetric_w = torch.pow(1 - self.xs_pos - self.xs_neg,
                                          self.gamma_pos * self.targets + self.gamma_neg * self.anti_targets)
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            self.loss *= self.asymmetric_w

        return -self.loss.sum()



class ContrastiveLoss(nn.Module):
    """
    Document: https://github.com/adambielski/siamese-triplet/blob/master/losses.py
    """

    def __init__(self, batchSize, reduce=None, size_average=None):
        super(ContrastiveLoss, self).__init__()

        self.batchSize = batchSize
        self.concatIndex = self.getConcatIndex(batchSize)

        self.reduce = reduce
        self.size_average = size_average

        self.cos = torch.nn.CosineSimilarity(dim=2, eps=1e-9)

    def forward(self, input, target):
        """
        Shape of input: (BatchSize, classNum, featureDim)
        Shape of target: (BatchSize, classNum), Value range of target: (-1, 0, 1)
        """

        target_ = target.detach().clone()
        target_[target_ != 1] = 0
        pos2posTarget = target_[self.concatIndex[0]] * target_[self.concatIndex[1]]

        pos2negTarget = 1 - pos2posTarget
        pos2negTarget[(target[self.concatIndex[0]] == 0) | (target[self.concatIndex[1]] == 0)] = 0
        pos2negTarget[(target[self.concatIndex[0]] == -1) & (target[self.concatIndex[1]] == -1)] = 0

        target_ = -1 * target.detach().clone()
        target_[target_ != 1] = 0
        neg2negTarget = target_[self.concatIndex[0]] * target_[self.concatIndex[1]]

        distance = self.cos(input[self.concatIndex[0]], input[self.concatIndex[1]])

        if self.reduce:
            pos2pos_loss = (1 - distance)[pos2posTarget == 1]
            pos2neg_loss = (1 + distance)[pos2negTarget == 1]
            neg2neg_loss = (1 + distance)[neg2negTarget == 1]

            if pos2pos_loss.size(0) != 0:
                if neg2neg_loss.size(0) != 0:
                    neg2neg_loss = torch.cat((torch.index_select(neg2neg_loss, 0, torch.randperm(neg2neg_loss.size(0))[:2 * pos2pos_loss.size(0)].cuda()),
                                              torch.sort(neg2neg_loss, descending=True)[0][:pos2pos_loss.size(0)]), 0)
                if pos2neg_loss.size(0) != 0:
                    if pos2neg_loss.size(0) != 0:    
                        pos2neg_loss = torch.cat((torch.index_select(pos2neg_loss, 0, torch.randperm(pos2neg_loss.size(0))[:2 * pos2pos_loss.size(0)].cuda()),
                                                  torch.sort(pos2neg_loss, descending=True)[0][:pos2pos_loss.size(0)]), 0)

            loss = torch.cat((pos2pos_loss, pos2neg_loss, neg2neg_loss), 0)

            if self.size_average:
                return torch.mean(loss) if loss.size(0) != 0 else torch.mean(torch.zeros_like(loss).cuda())
            return torch.sum(loss) if loss.size(0) != 0 else torch.sum(torch.zeros_like(loss).cuda())
 
        return distance

    def getConcatIndex(self, classNum):
        res = [[], []]
        for index in range(classNum - 1):
            res[0] += [index for i in range(classNum - index - 1)]
            res[1] += [i for i in range(index + 1, classNum)]
        return res
    


LOG_EPSILON = 1e-5

def neg_log(x):
    return - torch.log(x + LOG_EPSILON)
