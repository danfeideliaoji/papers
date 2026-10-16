import torch
import numpy as np
from sklearn.metrics import hamming_loss, coverage_error, label_ranking_loss, label_ranking_average_precision_score, f1_score

FMT = '%.4f'

def test_metrics(model, test_features, labels, params, mode):
    if mode == 'train':
        with torch.no_grad():
            model.eval()
            output,_, _ = model(test_features)
            pre_labels = np.array((torch.sigmoid(output) > params['confidence_bar']).cpu(), dtype=np.int64)
            outputs = output.cpu().numpy()
            true_labels = labels.cpu().numpy()
            return all_metrics(outputs, pre_labels, true_labels)
    else:
        with torch.no_grad():
            model.eval()
            output,_ ,_ = model.evaluate(test_features)
            pre_labels = np.array((torch.sigmoid(output) > params['confidence_bar']).cpu(), dtype=np.int64)
            outputs = output.cpu().numpy()
            true_labels = labels.cpu().numpy()
            return all_metrics(outputs, pre_labels, true_labels)

def all_metrics(outputs, pre_labels, true_labels):
    metrics_name = ['HM', 'AP', 'OE', 'RL', 'subset','coverage', 'macro', 'micro']
    hamming_loss = HammingLoss(pre_labels, true_labels)
    avg_precision = AveragePrecision(outputs, true_labels)
    ranking_loss = RankingLoss(outputs, true_labels)
    coverage = Coverage(outputs, true_labels)
    subset = Subset_accuracy(pre_labels, true_labels)
    macrof1 = MacroF1(pre_labels, true_labels)
    microf1 = MicroF1(pre_labels, true_labels)
    one_error = OneError(outputs, true_labels)
    # tensor
    # outputs = torch.tensor(outputs, dtype=torch.float32, device='cpu')
    # pre_labels = torch.tensor(pre_labels, dtype=torch.float32, device='cpu')
    # true_labels = torch.tensor(true_labels, dtype=torch.float32, device='cpu')
    metrics_res = [hamming_loss, avg_precision, one_error, ranking_loss, subset, coverage, macrof1, microf1]
    # tuple
    return list(zip(metrics_name, metrics_res))

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
    #cnt /= output.shape[1] #norm
    print('cnt1:{} cnt:{}'.format(cnt1, cnt))
    return cnt

def Subset_accuracy(predict, target):
    sa =  ((predict == target).sum(1) == target.shape[1]).sum() / predict.shape[0]
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

def AveragePrecision(outputs, true_labels):
    ap = label_ranking_average_precision_score(true_labels, outputs)
    return float(FMT % ap)

def HammingLoss(pre_labels, true_labels):
    hl = hamming_loss(true_labels, pre_labels)
    return float(FMT % hl)

def OneError(outputs, true_labels):
    indices = np.argmax(outputs, axis=1)
    # print('logits:', logits, logits.shape)
    # print('indices:', indices, indices.shape)
    # print('label_i', labels)
    # print('labels:', labels[indices])
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
    cnt = 0
    mul = (pre_labels * true_labels).sum(0) * 2
    su = pre_labels.sum(0) + true_labels.sum(0)
    for i in range(pre_labels.shape[1]):
        if mul[i] > 0:
            cnt += mul[i] / su[i]
    maf = cnt / pre_labels.shape[1]
    return float(FMT % maf)

def MicroF1(pre_labels, true_labels):
    mif = f1_score(true_labels, pre_labels, average='micro')
    return float(FMT % mif)
