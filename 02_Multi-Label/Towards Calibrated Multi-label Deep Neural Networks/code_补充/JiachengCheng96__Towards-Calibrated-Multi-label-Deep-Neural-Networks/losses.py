import torch
import torch.nn as nn

class SPALoss(nn.Module):
    def __init__(self, eps=1e-8, k0=3, b0=0.75, zeta0=5):
        super(SPALoss, self).__init__()

        self.eps = eps
        self.k0 = k0
        self.b0 = b0
        self.zeta0 = zeta0

    def forward(self, x, y):

        num_classes = y.shape[1]
        batch_size = y.shape[0]


        xs_pos = torch.sigmoid(x).clamp(min=self.eps, max=1-self.eps)
        xs_neg = torch.sigmoid(- self.k0 * (x-self.b0)).clamp(min=self.eps, max=1-self.eps)

        # los_pos = -torch.log(xs_pos[y>0.5]).sum()
        # los_neg = -torch.log((xs_neg)[(y<0.5)]).sum() / self.C

        los_pos = (y * (-torch.log(xs_pos))).sum()
        los_neg = ((1-y) * (-torch.log(xs_neg) / self.zeta0)).sum()


        # l_spa = (los_pos + los_neg ) / (batch_size)
        # l_spa = self.lmd0 * l_spa

        l_spa = (los_pos + los_neg)

        return l_spa, los_pos, los_neg


class SPAInvLink(nn.Module):
    def __init__(self, eps=1e-8, k0=3, b0=0.75, zeta0=5):
        super().__init__()

        self.k0 = k0
        self.b0 = b0
        self.zeta0 = zeta0

        self.eps = eps

    def forward(self, x):

        temp1 = torch.exp(x)
        temp2 = torch.exp(- self.k0 * (x - self.b0))

        out = (self.k0 * temp1 + 1) / (self.zeta0 + self.zeta0 *temp2 + self.k0 + self.k0 * temp1)
        
        return torch.clip(out, min=self.eps, max=1-self.eps)