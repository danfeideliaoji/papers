# 历史评估工具，当前训练入口不调用本文件。
# 当前公式对齐评估统一使用 strict_training.predict/metrics；旧结果从自己的源码快照重放。

import torch
import numpy as np
from sklearn.metrics import label_ranking_loss, label_ranking_average_precision_score, coverage_error,hamming_loss,f1_score,accuracy_score,recall_score,average_precision_score,precision_score
import copy

# def weighted_cross_entropy(prediction, gt, weight, weight_decay=0.001,model=None):
#
# 	loss=weight*gt*torch.log(prediction)+(1-weight)*(1-gt)*(torch.log(1-prediction))
# 	# loss=gt*torch.log(prediction)+(1-gt)*(torch.log(1-prediction))
# 	return torch.mean(-loss)

# 计算与训练端一致的加权 BCE 和参数范数正则，评估损失不用于反向传播。
def weighted_cross_entropy(prediction, gt, weight, weight_decay=0.005,model=None):
    ce_loss = weight * gt * torch.log(prediction + 1e-10) + (1 - weight) * (1 - gt) * torch.log(1 - prediction + 1e-10)
    ce_loss = -torch.mean(ce_loss)
   	# L2
    l2_reg = 0.0
    for param in model.parameters():
        l2_reg += torch.norm(param)
    loss = ce_loss + weight_decay * l2_reg
    return loss
# 备用的两视图标签原型计算；当前评估使用外部传入的标签初始特征。
def compute_embedding(views, target):
    target = target[0]
    num_classes = target.shape[1]#num_classes=6
    feature_dim = views[0].shape[1]#feature_dim=128
    com = torch.zeros(num_classes, feature_dim)
    num = torch.zeros(num_classes)
    for i in range(target.shape[0]):
        for j in range(num_classes):
            if target[i][j] == 1:
                com[j] += views[0][i] + views[1][i]
                num[j] += 1
    for i in range(num_classes):
        if num[i] > 0:
            com[i] /= num[i]
    return com
# 辅助输出正负标签识别率；二值化阈值固定为 0.5。
def cal_metric(gt,prediction):
	prediction=torch.tensor(prediction)
	gt=torch.tensor(gt)
	count_pos = float(torch.sum(gt))
	count_all=float((gt.shape[0]*gt.shape[1]))
	count_neg=count_all-count_pos
	prediction[prediction>=0.5]=1
	prediction[prediction<0.5]=0
	gt_1_pos, pre_1_pos, gt_2_pos, pre_2_pos = [], [], [], []
	for i, gt_label in enumerate(gt):
		if torch.sum(gt_label) == 1:
			gt_1_pos.append(gt_label)
			pre_1_pos.append(prediction[i])
		if torch.sum(gt_label) == 2:
			gt_2_pos.append(gt_label)
			pre_2_pos.append(prediction[i])
	gt_1_pos, pre_1_pos, gt_2_pos, pre_2_pos= \
		torch.stack(gt_1_pos), torch.stack(pre_1_pos), torch.stack(gt_2_pos), torch.stack(pre_2_pos), \

	tp_1 = torch.sum((pre_1_pos==gt_1_pos)[gt_1_pos==1]).float()
	tn_1 = torch.sum((pre_1_pos == gt_1_pos)[gt_1_pos == 0]).float()
	fn_1 = torch.sum((pre_1_pos != gt_1_pos)[gt_1_pos == 1]).float()
	fp_1 = torch.sum((pre_1_pos != gt_1_pos)[gt_1_pos == 0]).float()
	count_pos_1 = float(torch.sum(gt_1_pos))
	count_all_1 = float((gt_1_pos.shape[0] * gt_1_pos.shape[1]))
	count_neg_1 = count_all_1 - count_pos_1
	print("count_pos_1:", count_pos_1, "count_neg_1", count_neg_1, "tp_1:",tp_1/count_pos_1,"\ttn_1:",tn_1/count_neg_1,"\tfn_1:",fn_1/count_pos_1,"\tfp_1:",fp_1/count_neg_1)

	tp_2 = torch.sum((pre_2_pos == gt_2_pos)[gt_2_pos == 1]).float()
	tn_2 = torch.sum((pre_2_pos == gt_2_pos)[gt_2_pos == 0]).float()
	fn_2 = torch.sum((pre_2_pos != gt_2_pos)[gt_2_pos == 1]).float()
	fp_2 = torch.sum((pre_2_pos != gt_2_pos)[gt_2_pos == 0]).float()
	count_pos_2 = float(torch.sum(gt_2_pos))
	count_all_2 = float((gt_2_pos.shape[0] * gt_2_pos.shape[1]))
	count_neg_2 = count_all_2 - count_pos_2
	print("count_pos_2:", count_pos_2, "count_neg_2", count_neg_2, "tp_2:", tp_2/count_pos_2, "\ttn_2:", tn_2/count_neg_2, "\tfn_2:", fn_2/count_pos_2, "\tfp_2:", fp_2/count_neg_2)


	tp_=torch.sum((prediction==gt)[gt==1]).float()
	tn_=torch.sum((prediction==gt)[gt==0]).float()
	fn_=torch.sum((prediction!=gt)[gt==1]).float()
	fp_=torch.sum((prediction!=gt)[gt==0]).float()

	rate_tp=tp_/count_pos
	rate_tn=tn_/count_neg
	print("count_pos:", count_pos, "count_neg", count_neg, "tp:",rate_tp,"\ttn:",rate_tn,"\tfn:",fn_/count_pos,"\tfp:",fp_/count_neg)

	return rate_tp,rate_tn

# 按标签计算 VOC 风格 AP 的诊断值，与用于模型选择的 LRAP 含义不同。
def iccv2019_mAP(gt, prediction):
	gt = np.array(gt)
	prediction = np.array(prediction)
	num_target = np.sum(gt, axis=1, keepdims=True)
	threshold = 1 / (num_target + 1e-6)
	sample_num = len(gt)
	class_num = gt.shape[1]
	tp = np.zeros(sample_num)
	fp = np.zeros(sample_num)
	aps = []
	for class_id in range(class_num):
		confidence = prediction[:, class_id]
		sorted_ind = np.argsort(-confidence)
		sorted_label = [gt[x][class_id] for x in sorted_ind]
		for i in range(sample_num):
			tp[i] = (sorted_label[i] > 0)#0
			fp[i] = (sorted_label[i] <= 0)#0
		true_num = sum(tp)
		# True Positive (tp) & False Positive (fp)
		fp = np.cumsum(fp)
		tp = np.cumsum(tp)
		rec = tp / true_num
		prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
		ap = voc_ap(rec, prec)
		aps += [ap]

	np.set_printoptions(precision=3, suppress=True)
	mAP = np.mean(aps)
	return mAP
# 对精确率包络与召回率变化点积分，得到单类别 AP。
def voc_ap(rec, prec):
	mrec = np.concatenate(([0.], rec, [1.]))
	mpre = np.concatenate(([0.], prec, [0.]))
	for i in range(mpre.size - 1, 0, -1):
		mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
	i = np.where(mrec[1:] != mrec[:-1])[0]
	ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
	return ap

# 给定分数矩阵，检查最高分标签是否为真实正标签。
def One_Error(logits, labels):
    _, indices = torch.max(logits, dim=1)
    #print('logits:', logits, logits.shape)
    #print('indices:', indices, indices.shape)
    #print('label_i', labels)
    #print('labels:', labels[indices])
    cnt1 = 0
    for i in range(logits.shape[0]):
        cnt1 += (labels[i][indices[i]] == 0)
    #print('cnt1:', cnt1.item())
    return cnt1.item() / logits.shape[0]
    #return cnt1
    #correct = torch.sum(indices == labels)
    cnt = 0
    for i in range(logits.shape[0]):
        mx = 0
        loc = 0
        for j in range(logits.shape[1]):
            if logits[i][j] > mx:
                mx = logits[i][j]
                loc = j
        if labels[i][loc] == 0:
            cnt += 1
        #print('i:{} loc:{} indices:{}'.format(i, loc, indices[i]))
    cnt /= logits.shape[0]
    print('cnt1:{} cnt:{}'.format(cnt1, cnt))
    return cnt

# 计算每个样本全部标签都正确的比例。
def subset_accuracy(predict, target):
    return ((predict == target).sum(1) == target.shape[1]).sum() / predict.shape[0]
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


# 整理测试指标，排序类指标读连续分数，F1/Hamming 读 threshold 二值预测。
# 注意当前评估端把 y_pred 传给 One_Error，可能因阈值化并列影响最高标签选择。
def calculate_indexes(y_true, y_score,args):
	print("test function:")
	print(f"y_true:{y_true.shape}")
	print(f"y_true:{y_true}")
	print(f"y_score:{y_score.shape}")
	print(f"y_score:{y_score}")
	# , , ,,
	# ,accuracy_score,recall_score,average_precision_score,precision_score
	y_true = y_true.cpu().detach()
	y_score = y_score.cpu().detach()
	y_pred = copy.deepcopy(y_score)

	y_pred[y_pred >= args.threshold] = 1
	y_pred[y_pred < args.threshold] = 0
	print(f"y_pred_0-1:{y_pred}")

	ham_loss = hamming_loss(y_true, y_pred)

	Label_Ranking_loss = label_ranking_loss(y_true, y_score)

	# 保留原测试实现：此处使用二值预测，不等同于训练端对连续分数计算 One_Error。
	one_error = One_Error(y_pred, y_true)

	Coverage = coverage_error(y_true, y_score) -1

	Label_Ranking_Average_PrecisionScore = label_ranking_average_precision_score(y_true, y_score)

	Macro_F1 = f1_score(y_true, y_pred, average='macro')

	Micro_F1 = f1_score(y_true, y_pred, average='micro')

	Subset_Accuracy = subset_accuracy(y_pred, y_true)

	accuracyScore = accuracy_score(y_true, y_pred)
	Recall_Score = recall_score(y_true, y_pred,average=None)
	Precision_Score = precision_score(y_true, y_pred,average=None)
	Average_Precision_Score = average_precision_score(y_true, y_score)

	indexes_paper = {}
	indexes_paper["haming_loss"] = ham_loss
	indexes_paper["ranking_loss"] = Label_Ranking_loss
	indexes_paper["one_error"] = one_error
	indexes_paper["coverage"] = Coverage
	# 字典键沿用作者命名，实际值是样本级标签排序平均精度 LRAP。
	indexes_paper["average_precision"] = Label_Ranking_Average_PrecisionScore
	indexes_paper["micro_F1"] = Micro_F1
	indexes_paper["subset_accuracy"] = Subset_Accuracy
	indexes_paper["macro_F1"] = Macro_F1

	indexes_mine = {}
	indexes_mine["accuracyScore"] = accuracyScore
	indexes_mine["Recall_Score"] = Recall_Score
	indexes_mine["Precision_Score"] = Precision_Score
	indexes_mine["Average_Precision_Score"] = Average_Precision_Score
	return indexes_paper, indexes_mine

# 备用的标签逆频率权重函数，当前主评估没有使用它。
def compute_class_weights(gt):
    smooth_factor = 1e-6
    class_counts = torch.mean(gt, dim=0)
    class_frequencies = class_counts / gt.size(0)
    class_weights = 1.0 / class_frequencies + smooth_factor
    class_weights_normalized = class_weights / torch.sum(class_weights)
    return class_weights_normalized

# 完整评估一个测试折；输入 gt 是初始标签特征，gt_batch 是本批真实标签。
# 标签真值用于损失和指标，构图中的 generate_labels 只使用其形状、复制全部标签节点。
def testa(valloader,model,category_num,target,Y_testa,gt,args,indices_views_test,reshaped_test_view):
	model.eval()
	results = {'predict': [], 'label': []}
	Y_test = copy.deepcopy(Y_testa)

	batch_num = 0
	# #######################
	# class_weights = compute_class_weights(Y_testa[0])
	# # print(f"class_weights:{class_weights}")
	# class_weights = 1 - class_weights
	# #######################
	running_loss = []
	for idx, (inputs, gt_batch) in enumerate(valloader):
		# if gt_batch.shape[0]<args.k:
		# 	# n = args.k - gt_batch.shape[0]
		# 	Y_test[0] = Y_test[0][:-gt_batch.shape[0],:]
		# 	continue
		# 真实标签转到模型设备，防止 CUDA 预测与 CPU 标签计算损失时报设备不一致。
		device = next(model.parameters()).device
		gt_batch_temp = torch.transpose(copy.deepcopy(gt_batch), 0, 1).to(device)
		gt_batch = gt_batch.to(device)
		# gt = compute_embedding(inputs, gt)
		cross_edges, counts = model(inputs, gt, gt_batch,args,indices_views_test,reshaped_test_view,batch_num)
		batch_num = batch_num+1
		# gt[gt == -1] = 0
		count_cur, loss = 0, 0
		prediction_score = list()
		for j, count in enumerate(counts):
			prediction = cross_edges[count_cur: count_cur + (count[0] * count[1])]  # (6,64)
			prediction = prediction.view(-1, category_num)
			# prediction =  torch.max(prediction, dim=0)[0]
			# 每样本跨图边先还原为 [V,L]，再对 V 个视图取均值得到 [L] 标签概率。
			prediction = torch.mean(prediction, dim=0)
			prediction_score.append(prediction.unsqueeze(0))
			count_cur += (count[0] * count[1])
		# print('-----------------------prediction_score:', prediction_score)
		prediction = torch.cat(prediction_score, dim=0)
		# weight_matrix = class_weights.unsqueeze(0).expand_as(prediction).cuda()
		# prediction = prediction * weight_matrix
		# #######################
		# 保存各批预测与标签，最终一次计算整个测试折的指标。
		results['predict'].extend([x.tolist() for x in prediction])
		results['label'].extend([y.tolist() for y in gt_batch_temp[0]])
		# true_label = Y_test[0].to("cuda:0")
		loss = weighted_cross_entropy(prediction=prediction, gt=gt_batch_temp[0], weight=0.8, model=model)  # 0.80
		running_loss.append(loss.item())
	running_loss_return = np.mean(np.array(running_loss)).item()
	rate_ptrate_tp, rate_ptrate_pn = cal_metric(results['label'], results['predict'])
	ICCV_test_acc = iccv2019_mAP(results['label'], results['predict'])
	indexes_paper,indexes_mine = calculate_indexes(
		Y_test[0], torch.tensor(results['predict']),args)
	indexes = {}
	indexes['paper'] = indexes_paper
	indexes['mine'] = indexes_mine
	print(
		f"test set indicators：haming_loss:{indexes_paper['haming_loss']}, Ranking_loss:{indexes_paper['ranking_loss']}, one_error:{indexes_paper['one_error']}, Coverage:{indexes_paper['coverage']}, Average_Precision:{indexes_paper['average_precision']}, Micro_F1:{indexes_paper['micro_F1']}, Subset_Accuracy:{indexes_paper['subset_accuracy']}, Macro_F1:{indexes_paper['macro_F1']}")
	print("*" * 66)
	return rate_ptrate_tp, rate_ptrate_pn, ICCV_test_acc,indexes,running_loss_return
