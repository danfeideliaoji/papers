import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

def average_precision(output, target):
    epsilon = 1e-8

    # sort examples
    indices = output.argsort()[::-1]
    # Computes prec@i
    total_count_ = np.cumsum(np.ones((len(output), 1)))

    target_ = target[indices]
    ind = target_ == 1
    pos_count_ = np.cumsum(ind)
    total = pos_count_[-1]
    pos_count_[np.logical_not(ind)] = 0
    pp = pos_count_ / total_count_
    precision_at_i_ = np.sum(pp)
    precision_at_i = precision_at_i_ / (total + epsilon)

    return precision_at_i

def mAP(targs, preds):
    """Returns the model's average precision for each class
    Return:
        ap (FloatTensor): 1xK tensor, with avg precision for each class k
    """

    if np.size(preds) == 0:
        return 0
    ap = np.zeros((preds.shape[1]))
    # compute average precision for each class
    for k in range(preds.shape[1]):
        # sort scores
        scores = preds[:, k]
        targets = targs[:, k]
        # compute average precision
        ap[k] = average_precision(scores, targets)
    return 100 * ap.mean()


class BinaryECE(nn.Module):
    """
    Calculates the Expected Calibration Error of a model.
    (This isn't necessary for temperature scaling, just a cool metric).
    The input to this loss is the logits of a model, NOT the softmax scores.
    This divides the confidence outputs into equally-sized interval bins.
    In each bin, we compute the confidence gap:
    bin_gap = | avg_confidence_in_bin - accuracy_in_bin |
    We then return a weighted average of the gaps, based on the number
    of samples in each bin
    See: Naeini, Mahdi Pakdaman, Gregory F. Cooper, and Milos Hauskrecht.
    "Obtaining Well Calibrated Probabilities Using Bayesian Binning." AAAI.
    2015.
    """
    def __init__(self, n_bins=20, strategy='quantile'):
        """
        n_bins (int): number of confidence interval bins
        """
        super(BinaryECE, self).__init__()

        self.strategy = strategy
        assert strategy in ['uniform', 'quantile']

        if self.strategy == 'uniform':
            bin_boundaries = torch.linspace(0, 1, n_bins + 1)
            self.bin_lowers = bin_boundaries[:-1]
            self.bin_uppers = bin_boundaries[1:]

        self.n_bins = n_bins

    def get_classwise(self, y_pred, y_true):
        assert len(y_pred.shape) == 2
        N, C = y_pred.shape

        eces = []
        aces = []
        mces = []

        for i in range(C):
            results, _, _, _ = self.forward(y_pred[:,i], y_true[:,i])
            ece, ace, mce = results[0], results[1], results[2]

            eces.append(ece)
            aces.append(ace)
            mces.append(mce)

        class_ece = torch.stack(eces).mean()
        class_ace = torch.stack(aces).mean()
        class_mce = torch.stack(mces).mean()

        return [class_ece, class_ace, class_mce]


    def forward(self, y_pred, y_true):
        # softmax_scores, labels: torch tensors

        y_true = y_true.flatten()
        y_pred = y_pred.flatten()

        y_pred = y_pred.to(torch.float64)

        if y_pred.min() < 0 or y_pred.max() > 1:
            raise ValueError("y_prob has values outside [0, 1].")



        if self.strategy == 'quantile':
            quantiles = torch.linspace(0, 1, self.n_bins + 1)
            # bin_boundaries = torch.linspace(0, 1, self.n_bins + 1)
            bin_boundaries = torch.quantile(y_pred.flatten(), quantiles.to(y_pred.dtype))
            self.bin_lowers = bin_boundaries[:-1]
            self.bin_uppers = bin_boundaries[1:]

        # confidences, predictions = torch.max(softmax_scores, 1)
        # accuracies = predictions.eq(labels)

        weights = []
        prob_true, prob_pred = [], []

        ece = 0
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = y_pred.gt(bin_lower.item()) * y_pred.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()    


            if prop_in_bin.item() > 0:

                tp_in_bin = y_true[in_bin].float().mean()
                avg_confidence_in_bin = y_pred[in_bin].mean()

                prob_true.append(tp_in_bin.item())
                prob_pred.append(avg_confidence_in_bin.item())
                weights.append(prop_in_bin.item())

        # print(1)
        prob_true = torch.tensor(prob_true)
        prob_pred = torch.tensor(prob_pred)
        weights = torch.tensor(weights)
        
        residual  = (prob_true - prob_pred).abs()

        ece = (weights * residual).mean()
        ace = (residual).mean()
        mce = (residual).max()


        return [ece, ace, mce], prob_true, prob_pred, weights

def eval_calibration(output_2d, target_2d, verbose=False, nbins=15):

    eceloss_uniform = BinaryECE(n_bins=nbins, strategy='uniform')
    results, _, _, _ = eceloss_uniform(output_2d, target_2d)


    ece, ace, mce = results[0], results[1], results[2]

    class_results = eceloss_uniform.get_classwise(output_2d, target_2d)
    class_ece, class_ace, class_mce = class_results[0], class_results[1], class_results[2]

    if not verbose: 
        print(f'ECE: {ece.item():.3f}, ACE: {ace.item():.3f}, MCE: {mce.item():.3f}')
        print(f'Classwise ECE: {class_ece.item():.3f}, Classwise ACE: {class_ace.item():.3f}, Classwise MCE: {class_mce.item():.3f}')

