"""Loss functions used by BPGT training scripts."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.utils import ComparativeMultiLabelLoss, multilabel_categorical_crossentropy


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None, reduction="mean"):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction

    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(
            logits,
            targets,
            weight=self.weight,
            reduction="none",
        )
        pt = torch.exp(-ce_loss)
        loss = ((1 - pt) ** self.gamma) * ce_loss
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


class class_balanced_loss(nn.Module):
    """Class-balanced CE/focal loss.

    This preserves the old constructor API:
    ``class_balanced_loss(samples_per_cls, no_of_classes, loss_type, beta, gamma)``.
    """

    def __init__(self, samples_per_cls, no_of_classes, loss_type="focal", beta=0.9999, gamma=2.0):
        super().__init__()
        effective_num = 1.0 - torch.pow(torch.tensor(beta), torch.tensor(samples_per_cls, dtype=torch.float32))
        weights = (1.0 - beta) / torch.clamp(effective_num, min=1e-12)
        weights = weights / weights.sum() * no_of_classes
        self.register_buffer("weights", weights.float())
        self.loss_type = loss_type
        self.gamma = gamma

    def forward(self, logits, targets):
        weights = self.weights.to(logits.device)
        if self.loss_type == "focal":
            return FocalLoss(gamma=self.gamma, weight=weights)(logits, targets)
        return F.cross_entropy(logits, targets, weight=weights)


__all__ = [
    "ComparativeMultiLabelLoss",
    "FocalLoss",
    "class_balanced_loss",
    "multilabel_categorical_crossentropy",
]
