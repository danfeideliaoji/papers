"""多标签分类评估指标。

模型输出为每个样本、每个类别的实数分数 ``[N, C]``。本模块先按置信度
阈值将分数转成二值预测，再统一计算汉明损失、平均精度、One-Error、排序
损失、子集准确率、Coverage、Macro-F1 和 Micro-F1。
"""

import torch
import numpy as np
from sklearn.metrics import hamming_loss, coverage_error, label_ranking_loss, label_ranking_average_precision_score, f1_score

FMT = '%.4f'

def test_metrics(model, test_features, labels, params, mode):
    """在不记录梯度的情况下运行模型并返回全部评估指标。

    Args:
        model (HGMVMLC_Model): 待评估模型。
        test_features (list[torch.Tensor]): 多视图特征列表，第 ``v`` 项形状为 ``[N, D_v]``。
        labels (torch.Tensor): 二值多标签矩阵，形状为 ``[N, C]``。
        params (dict[str, object]): 包含 ``confidence_bar`` 等字段的数据集配置。
        mode (str): ``'train'`` 时复用训练超图调用 ``model``；其他值调用
            ``model.evaluate``，使用不依赖真实标签的验证/测试超图。

    Returns:
        ``[(指标名, 指标值), ...]``，固定包含八项指标。
    """
    if mode == 'train':
        # torch.no_grad() 临时关闭梯度记录：评估时节省显存，也不会修改模型参数。
        with torch.no_grad():
            model.eval()
            output,_, _ = model(test_features)
            # torch.sigmoid 把任意实数 logits 压到 (0,1)，可解释为各标签的预测概率。
            pre_labels = np.array((torch.sigmoid(output) > params['confidence_bar']).cpu(), dtype=np.int64)
            outputs = output.cpu().numpy()
            true_labels = labels.cpu().numpy()
            return all_metrics(outputs, pre_labels, true_labels)
    else:
        # no_grad 只关闭求导，不会自动切换模型的 train/eval 状态，所以上面仍需 model.eval()。
        with torch.no_grad():
            model.eval()
            output,_ ,_ = model.evaluate(test_features)
            pre_labels = np.array((torch.sigmoid(output) > params['confidence_bar']).cpu(), dtype=np.int64)
            outputs = output.cpu().numpy()
            true_labels = labels.cpu().numpy()
            return all_metrics(outputs, pre_labels, true_labels)

def all_metrics(outputs, pre_labels, true_labels):
    """汇总八项多标签指标，并维持训练脚本依赖的固定顺序。

    Args:
        outputs (numpy.ndarray): 连续预测分数 ``[N, C]``。
        pre_labels (numpy.ndarray): 阈值化后的二值预测 ``[N, C]``。
        true_labels (numpy.ndarray): 二值真实标签 ``[N, C]``。
    """
    metrics_name = ['HM', 'AP', 'OE', 'RL', 'subset','coverage', 'macro', 'micro']
    hamming_loss = HammingLoss(pre_labels, true_labels)
    avg_precision = AveragePrecision(outputs, true_labels)
    ranking_loss = RankingLoss(outputs, true_labels)
    coverage = Coverage(outputs, true_labels)
    subset = Subset_accuracy(pre_labels, true_labels)
    macrof1 = MacroF1(pre_labels, true_labels)
    microf1 = MicroF1(pre_labels, true_labels)
    one_error = OneError(outputs, true_labels)
    metrics_res = [hamming_loss, avg_precision, one_error, ranking_loss, subset, coverage, macrof1, microf1]
    # 返回名称和值组成的二元组，便于训练脚本按名称展示、按位置保存。
    return list(zip(metrics_name, metrics_res))

def Coverage(output, target):
    """计算归一化前的 Coverage Error（减 1 后与标签排名从 0 开始一致）。

    Args:
        output (numpy.ndarray): 连续预测分数 ``[N,C]``。
        target (numpy.ndarray): 二值真实标签 ``[N,C]``。
    """
    co = coverage_error(target, output) - 1
    return float(co)

def Subset_accuracy(predict, target):
    """计算子集准确率：一个样本的全部标签都正确时才计为正确。

    Args:
        predict (numpy.ndarray): 二值预测 ``[N,C]``。
        target (numpy.ndarray): 二值真实标签 ``[N,C]``。
    """
    sa =  ((predict == target).sum(1) == target.shape[1]).sum() / predict.shape[0]
    return float(sa)

def AveragePrecision(outputs, true_labels):
    """计算标签排序平均精度（LRAP，数值越大越好）。

    Args:
        outputs (numpy.ndarray): 连续预测分数 ``[N,C]``。
        true_labels (numpy.ndarray): 二值真实标签 ``[N,C]``。
    """
    # 保留原始排名与并列关系，让 sklearn 的计数/除法使用 float64。
    ap = label_ranking_average_precision_score(true_labels, np.asarray(outputs, dtype=np.float64))
    return float(ap)

def HammingLoss(pre_labels, true_labels):
    """计算逐标签汉明损失（数值越小越好）。

    Args:
        pre_labels (numpy.ndarray): 二值预测 ``[N,C]``。
        true_labels (numpy.ndarray): 二值真实标签 ``[N,C]``。
    """
    hl = hamming_loss(true_labels, pre_labels)
    return float(hl)

def OneError(outputs, true_labels):
    """计算 One-Error：最高分标签不属于真实标签集合的样本比例。

    Args:
        outputs (numpy.ndarray): 连续预测分数 ``[N,C]``。
        true_labels (numpy.ndarray): 二值真实标签 ``[N,C]``。
    """
    indices = np.argmax(outputs, axis=1)
    cnt1 = 0
    for i in range(outputs.shape[0]):
        cnt1 += (true_labels[i][indices[i]] == 0)
    oe = cnt1.item() / outputs.shape[0]
    return float(oe)

def RankingLoss(outputs, true_labels):
    """计算标签排序损失（错误排序的正负标签对比例，越小越好）。

    Args:
        outputs (numpy.ndarray): 连续预测分数 ``[N,C]``。
        true_labels (numpy.ndarray): 二值真实标签 ``[N,C]``。
    """
    rl = label_ranking_loss(true_labels, outputs)
    return float(rl)

def MacroF1(pre_labels, true_labels):
    """按类别计算 F1 后求平均；没有真阳性的类别对总和贡献为 0。

    Args:
        pre_labels (numpy.ndarray): 二值预测 ``[N,C]``。
        true_labels (numpy.ndarray): 二值真实标签 ``[N,C]``。
    """
    cnt = 0
    mul = (pre_labels * true_labels).sum(0) * 2
    su = pre_labels.sum(0) + true_labels.sum(0)
    for i in range(pre_labels.shape[1]):
        if mul[i] > 0:
            cnt += mul[i] / su[i]
    maf = cnt / pre_labels.shape[1]
    return float(maf)

def MicroF1(pre_labels, true_labels):
    """汇总所有样本和类别的 TP/FP/FN 后计算 Micro-F1。

    Args:
        pre_labels (numpy.ndarray): 二值预测 ``[N,C]``。
        true_labels (numpy.ndarray): 二值真实标签 ``[N,C]``。
    """
    mif = f1_score(true_labels, pre_labels, average='micro', zero_division=0)
    return float(mif)
