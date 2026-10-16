import torch
import torch.nn as nn
import torch.nn.functional as F


class unsupervised_contrastive_loss(nn.Module):
    def __init__(self, temperature=0.5, use_ema=False):


        super(unsupervised_contrastive_loss, self).__init__()
        self.temperature = temperature
        self.use_ema = use_ema

    def forward(self, feat_q, feat_k, feat_k_ema=None):

        if self.use_ema:
            assert feat_k_ema is not None, "EMA feature must be provided if use_ema=True"
            feat_k = feat_k_ema

        B, C, D = feat_q.shape
        q = feat_q.view(B * C, D)
        k = feat_k.view(B * C, D)
        logits = torch.matmul(q, k.T) / self.temperature
        labels = torch.arange(B * C).to(q.device)

        # InfoNCE loss
        loss = F.cross_entropy(logits, labels, reduction='sum') / (B * C)
        return loss
    


class unsupervised_contrastive_loss_flatten(nn.Module):
    def __init__(self, temperature=0.5, use_ema=False):

        super(unsupervised_contrastive_loss_flatten, self).__init__()
        self.temperature = temperature
        self.use_ema = use_ema

    def forward(self, feat_q, feat_k, feat_k_ema=None):
      
        if self.use_ema:
            assert feat_k_ema is not None, "EMA feature must be provided if use_ema=True"
            feat_k = feat_k_ema

        B, C = feat_q.shape

        q = feat_q
        k = feat_k
        logits = torch.matmul(q, k.T) / self.temperature

        labels = torch.arange(B).to(q.device)

        loss = F.cross_entropy(logits, labels, reduction='sum') / B
        return loss