import torch
import numpy as np
import torch.nn as nn
from torch import Tensor
from sklearn.metrics import hamming_loss, coverage_error, label_ranking_loss, label_ranking_average_precision_score, f1_score
# from torchmetrics.classification import MultilabelF1Score

FMT = '%.4f'


def all_metrics(outputs, pre_labels, true_labels):
    metrics_name = ['HM', 'AP', 'OE', 'RL', 'subset', 'coverage', 'macro', 'micro']
    hamming_loss = HammingLoss(pre_labels, true_labels)
    avg_precision = AveragePrecision(outputs, true_labels)
    ranking_loss = RankingLoss(outputs, true_labels)
    coverage = Coverage(outputs, true_labels)
    subset = Subset_accuracy(pre_labels, true_labels)
    macrof1 = MacroF1(pre_labels, true_labels)
    microf1 = MicroF1(pre_labels, true_labels)
    one_error = OneError(outputs, true_labels)

    metrics_res = [hamming_loss, avg_precision, one_error, ranking_loss, subset, coverage, macrof1, microf1]
    # 列表的元素是元组
    return list(zip(metrics_name, metrics_res)), metrics_res


def Coverage(output, target):
    co = coverage_error(target, output) - 1
    return float(FMT % co)
    cnt = 0
    for i in range(output.shape[0]):
        ans = 0
        rank = torch.zeros(output.shape[1])
        num = 0
        for j in range(output.shape[1]):
            for k in range(output.shape[1]):
                if output[i][k] >= output[i][j]:
                    rank[j] += 1
            if target[i][j] == 1:
                ans = max(ans, rank[j])
                num += 1
        cnt += ans - 1
    cnt /= output.shape[0]
    #cnt /= output.shape[1] #归一化
    print('cnt1:{} cnt:{}'.format(cnt1, cnt))
    return cnt


def Subset_accuracy(predict, target):
    sa = ((predict == target).sum(1) == target.shape[1]).sum() / predict.shape[0]
    return float(FMT % sa)
    cnt = 0
    for i in range(predict.shape[0]):
        flag = True
        for j in range(predict.shape[1]):
            if predict[i][j] != target[i][j]:
                flag = False
                break
        if flag:
            cnt += 1
    print('cnt1:{} cnt:{}'.format(cnt1, cnt))
    return cnt / predict.shape[0]


# AP通过输出的logits计算
def AveragePrecision(outputs, true_labels):
    ap = label_ranking_average_precision_score(true_labels, outputs)
    return float(FMT % ap)


# HM loss通过预测的标签计算
def HammingLoss(pre_labels, true_labels):
    hl = hamming_loss(true_labels, pre_labels)
    return float(FMT % hl)


def OneError(outputs, true_labels):
    indices = np.argmax(outputs, axis=1)
    cnt1 = 0
    for i in range(outputs.shape[0]):
        cnt1 += (true_labels[i][indices[i]] == 0)
    # print('cnt1:', cnt1.item())
    oe = cnt1.item() / outputs.shape[0]
    return float(FMT % oe)


def RankingLoss(outputs, true_labels):
    rl = label_ranking_loss(true_labels, outputs)
    return float(FMT % rl)


def MacroF1(pre_labels, true_labels):
    maf = f1_score(true_labels, pre_labels, average='macro')
    return float(FMT % maf)


def MicroF1(pre_labels, true_labels):
    mif = f1_score(true_labels, pre_labels, average='micro')
    return float(FMT % mif)

