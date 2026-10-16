import torch
from sklearn.neighbors import NearestNeighbors
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
import torch.multiprocessing as mp
from GNBlock_model import _Model
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import os
import copy
import random
from build_graph import Graph_Generator
from encoder import Encoder
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim
from torch.nn import init
import math
import copy
from test import testa
from sklearn.metrics import label_ranking_loss, label_ranking_average_precision_score, coverage_error,hamming_loss,f1_score,accuracy_score,recall_score,average_precision_score,precision_score

from sklearn.model_selection import train_test_split

def data_clean_zscore(view, target):
    print("data process")
    eps = 1e-5
    for i in range(len(view)):
        valid_col = []
        # print('less:', (view[i] < 0).sum())
        for j in range(view[i].shape[1]):
            if not (view[i][:, j] == 0).sum() == view[i][:, j].shape[0]:
                valid_col.append(j)
            view[i][:, j] = (view[i][:, j] - view[i][:, j].mean()) / (torch.std((view[i][:, j])) + eps)
    return view, target

def print_std(list, name):
    x = torch.tensor(list)
    mean = x.mean()
    std = torch.std(x)
    print('{}:{}±{}'.format(name, mean, std))
    return mean,std

def get_shu(data, indce):
    res = torch.zeros(data.shape[0], data.shape[1])
    for i in range(data.shape[0]):
        res[i] = copy.deepcopy(data[indce[i]])
    return res


def shuffle(x, y, seed=2020):
    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)

    num_ins = x[0].shape[0]
    indces = list(range(num_ins))
    random.shuffle(indces)
    # print(f"indces:{indces}")
    # view
    a = [get_shu(x[i], indces) for i in range(len(x))]
    # target
    b = get_shu(y, indces)

    return a, b


def shuffle_columns(tensor, seed=2021):
    """
    Shuffle the columns of a given tensor.

    Args:
    - tensor: input tensor to be shuffled
    - seed: random seed for reproducibility (default=None)

    Returns:
    - shuffled tensor
    """
    if seed is not None:
        torch.manual_seed(seed)

    # Get the number of columns in the tensor
    num_cols = tensor.size(1)

    # Generate a permutation of column indices
    perm = torch.randperm(num_cols)

    # Shuffle the tensor according to the permutation
    shuffled_tensor = tensor[:, perm]

    return shuffled_tensor
def get_k_fold_data(k, i, X, y):

    assert k > 1
    fold_size = X.shape[0] // k

    X_train, y_train = None, None
    for j in range(k):
        idx = slice(j * fold_size, (j + 1) * fold_size)
        X_part, y_part = X[idx, :], y[idx]
        if j == i:
            X_test, y_test = X_part, y_part
        elif X_train is None:
            X_train, y_train = X_part, y_part
        else:
            X_train = torch.cat((X_train, X_part), dim=0)
            y_train = torch.cat((y_train, y_part), dim=0)
    return X_train, y_train, X_test, y_test

def plot_training_results(train_accuracy_list, test_accuracy_list, train_loss_list, test_loss_list,fold_num,ap_mean,args,result_total):

    """
    Plot training and testing accuracy curves, and training loss curves for each fold.

    Args:
    - train_accuracy_list: A list of training accuracy for each fold.
    - test_accuracy_list: A list of testing accuracy for each fold.
    - train_loss_list: A list of training loss for each fold.
    - fold_num: Number of folds.

    Returns:
    - None
    """
    pic = 'record'
    if not os.path.exists(pic):
        os.makedirs(pic)
    name = args.dataset[:args.dataset.find('.')]
    # ML-BVAE IMvMLC LrMMC LSPC SIMM FIMAN
    emotion_AP_result = {"ML-BVAE":0.572,"IMvMLC":0.782,"LrMMC":0.763, "LSPC":0.773,"SIMM":0.634 ,"FIMAN":0.806}
    plant_AP_result = {"ML-BVAE":0.505,"IMvMLC":0.544,"LrMMC":0.464, "LSPC":0.376,"SIMM":0.369 ,"FIMAN":0.492}
    scene_AP_result = {"ML-BVAE":0.717,"IMvMLC":0.844,"LrMMC":0.852, "LSPC":0.647,"SIMM":0.608 ,"FIMAN":0.827}
    Yeast_AP_result = {"ML-BVAE":0.712,"IMvMLC":0.738,"LrMMC":0.610, "LSPC":0.554,"SIMM":0.712 ,"FIMAN":0.740}
    human_AP_result = {"ML-BVAE":0.536,"IMvMLC":0.600,"LrMMC":0.480, "LSPC":0.304,"SIMM":0.495 ,"FIMAN":0.583}
    Pascal_AP_result = {"ML-BVAE":0.659,"IMvMLC":0.695,"LrMMC":0.422, "LSPC":0.116,"SIMM":0.685 ,"FIMAN":0.721}
    Corel5k_AP_result = {"ML-BVAE":0.286,"IMvMLC":0.333,"LrMMC":0.215, "LSPC":0.075,"SIMM":0.292 ,"FIMAN":0.430}
    Iaprtc12_AP_result = {"ML-BVAE":0.255,"IMvMLC":0.280,"LrMMC":0.219, "LSPC":0.021,"SIMM":0.326 ,"FIMAN":0.348}
    Espgame_AP_result = {"ML-BVAE":0.257,"IMvMLC":0.273,"LrMMC":0.170, "LSPC":0.020,"SIMM":0.308 ,"FIMAN":0.284}
    Mirflickr_AP_result = {"ML-BVAE":0.058,"IMvMLC":0.149,"LrMMC":0.053, "LSPC":0.272,"SIMM":0.119 ,"FIMAN":0}

    if name == "emotions":
        result_dict = emotion_AP_result
    elif name == "plant":
        result_dict = plant_AP_result
    elif name == "scene":
        result_dict = scene_AP_result
    elif name == "Yeast":
        result_dict = Yeast_AP_result
    elif name == "human":
        result_dict = human_AP_result
    elif name == "split-Pascal":
        result_dict = Pascal_AP_result
    elif name == "Corel5k":
        result_dict = Corel5k_AP_result
    elif name == "split-Iaprtc12":
        result_dict = Iaprtc12_AP_result
    elif name == "split-Espgame":
        result_dict = Espgame_AP_result
    elif name == "split-Mirflickr":
        result_dict = Mirflickr_AP_result
    else:
        raise ValueError("Invalid dataset name")

    plt.figure(figsize=(14, 10))
    index = range(len(result_dict))
    plt.bar(index, result_dict.values(), color='skyblue', label='Contrast Methods')
    plt.bar(len(result_dict), ap_mean, color='#fb6a4a', label='VAMS')
    plt.xlabel('Method')
    plt.ylabel('Average Precision (AP)')
    plt.title(f'Average Precision for {name} dataset')
    plt.xticks(list(index) + [len(result_dict)], list(result_dict.keys()) + ['VAMS'])
    for i, v in enumerate(result_dict.values()):
        plt.text(i, v + 0.01, f"{v:.3f}", ha='center')
    plt.text(len(result_dict), ap_mean + 0.01, f"{ap_mean:.3f}", ha='center')
    plt.legend()

    path = pic + '/' + name + '_comparative_result.png'
    plt.savefig(path, dpi=300)
    plt.show()

    fig, axs = plt.subplots(2, figsize=(20, 18))
    for fold in range(fold_num):
        axs[0].plot(train_accuracy_list[fold], label=f'Train Fold {fold+1}')
        axs[0].plot(test_accuracy_list[fold], linestyle='dashed', label=f'Test Fold {fold+1}')

    axs[0].set_title('Training and Testing Average Precision')
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Average Precision (AP)')
    axs[0].legend()

    for fold in range(fold_num):
        axs[1].plot(train_loss_list[fold], label=f'Train Fold {fold+1}')
        axs[1].plot(test_loss_list[fold],linestyle='dashed', label=f'Test Fold {fold+1}')
    axs[1].set_title('Training and Testing Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('Loss')
    axs[1].legend()
    plt.tight_layout()
    path = pic  + '/'+ name+'_display_result.png'
    plt.savefig(path, dpi=300)

    fold_num = 0
    result_test_hl = result_total['result_test_hl'][fold_num]
    result_test_rl = result_total['result_test_rl'][fold_num]
    result_test_oe = result_total['result_test_oe'][fold_num]
    result_test_cov = result_total['result_test_cov'][fold_num]
    result_test_ap = result_total['result_test_ap'][fold_num]
    result_test_micro = result_total['result_test_micro'][fold_num]
    result_test_suba = result_total['result_test_suba'][fold_num]
    result_test_macro = result_total['result_test_macro'][fold_num]
    data = [result_test_hl,result_test_rl,result_test_oe,result_test_cov,result_test_ap,result_test_micro,result_test_suba,result_test_macro]
    df = pd.DataFrame(data)
    df.to_excel('output'+'_'+name+'.xlsx', index=False, header=False)


    result = {'Hamming Loss':result_test_hl,'Ranking Loss':result_test_rl ,'One Error':result_test_oe ,'Coverage':result_test_cov ,'Average Precision':result_test_ap ,'Micro-F1':result_test_micro ,'Subset Accuracy':result_test_suba,'Macro-F1':result_test_macro}

    plt.figure(figsize=(20,18))
    for key, value in result.items():
        plt.plot(value, label=key)
    plt.title('Training Performance')
    plt.xlabel('Epoch')
    plt.ylabel('Performance')
    plt.legend()
    plt.tight_layout()
    path = pic  + '/'+ name+'_total_result.png'
    plt.savefig(path, dpi=300)
    plt.show()

def draw(x1, x2, path):
    leng = len(x1)
    x_list = np.zeros(leng)
    train_l = np.zeros(leng)
    test_l = np.zeros(leng)
    for i in range(leng):
        x_list[i] = i
        train_l[i] = x1[i]
        test_l[i] = x2[i]
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.plot(x_list, train_l, color='r', linewidth=1, alpha=0.6, label='train')
    plt.plot(x_list, test_l, color='b', linewidth=1, alpha=0.6, label='test')
    plt.legend()
    plt.savefig(path, dpi=300)


def one_hot(categories,args):
    '''
    :param categories:[2047,6]
    :param args:
    :return:
    '''
    one_hot_encoded = np.eye(categories.shape[1])
    extended_encoding = []
    for vector in one_hot_encoded:
        extended_vector = np.repeat(vector, args.nhidden // categories.shape[1])#np.tile
        if len(extended_vector) < args.nhidden:
            extended_vector = np.pad(extended_vector, (0, args.nhidden - len(extended_vector)), mode='constant',
                                     constant_values=0)
        extended_encoding.append(extended_vector)
    extended_encoding = np.array(extended_encoding)
    extended_encoding = torch.tensor(extended_encoding).float()
    return extended_encoding
def one_hot2(categories,args):
    with torch.no_grad():
        one_hot_encoded = np.eye(categories.shape[1])
        extended_encoding = []
        in_feat = categories.shape[1]
        if in_feat <=512:
            encoder = nn.Sequential(
                nn.Linear(in_feat, 256),
                nn.LeakyReLU(0.2),
                # nn.Dropout(p=self.dropout),
                nn.Linear(256, args.target_dim),
            )
        elif in_feat <= 1024:
            encoder = nn.Sequential(
                nn.Linear(in_feat, 512),
                nn.LeakyReLU(0.2),
                nn.Linear(512, 256),
                nn.LeakyReLU(0.2),
                nn.Linear(256, args.target_dim)
            )
        else:
            encoder = nn.Sequential(
                nn.Linear(in_feat, 2048),
                nn.LeakyReLU(0.2),
                nn.Linear(2048, 1024),
                nn.LeakyReLU(0.2),
                nn.Linear(1024, 512),
                nn.LeakyReLU(0.2),
                nn.Linear(512, 256),
                nn.LeakyReLU(0.2),
                nn.Linear(256, args.target_dim)
            )
        for vector in one_hot_encoded:
            vector = torch.tensor(vector).float()
            extended_vector = encoder(vector)
            extended_encoding.append(extended_vector)
        extended_encoding = torch.stack(extended_encoding)
    return extended_encoding

def compute_embedding(views, target):
    num_classes = target.shape[1]#num_classes=6
    feature_dim = views[0].shape[1]#feature_dim=128

    com = torch.zeros(num_classes,feature_dim)
    num = torch.zeros(num_classes)
    # print(f"com:{com.shape}")
    for i in range(target.shape[0]):
        for j in range(num_classes):
            if target[i][j] == 1:
                for k in range(len(views)):
                    com[j] += views[k][i]
                num[j] += 1

    for i in range(num_classes):
        if num[i] > 0:
            com[i] /= num[i]
    return com

# def weighted_cross_entropy(prediction, gt, weight, weight_decay=0.001,model=None):
#
# 	loss=weight*gt*torch.log(prediction)+(1-weight)*(1-gt)*(torch.log(1-prediction))
# 	# loss=gt*torch.log(prediction)+(1-gt)*(torch.log(1-prediction))
# 	return torch.mean(-loss)

def weighted_cross_entropy(prediction, gt, weight, weight_decay=0.005,model=None):

    ce_loss = weight * gt * torch.log(prediction + 1e-10) + (1 - weight) * (1 - gt) * torch.log(1 - prediction + 1e-10)
    ce_loss = -torch.mean(ce_loss)

    l2_reg = 0.0
    for param in model.parameters():
        l2_reg += torch.norm(param)
    loss = ce_loss + weight_decay * l2_reg

    return loss

def cal_metric(gt,prediction):
	prediction=torch.tensor(prediction)
	gt=torch.tensor(gt)

	count_pos = float(torch.sum(gt))
	count_all=float((gt.shape[0]*gt.shape[1]))
	count_neg=count_all-count_pos

	prediction[prediction>=0.5]=1
	prediction[prediction<0.5]=0
	gt_1_pos, pre_1_pos, gt_2_pos, pre_2_pos, gt_3_pos, pre_3_pos, gt_4_pos, pre_4_pos = [], [], [], [], [], [], [], []

	for i, gt_label in enumerate(gt):
		if torch.sum(gt_label) == 1:
			gt_1_pos.append(gt_label)
			pre_1_pos.append(prediction[i])
		if torch.sum(gt_label) == 2:
			gt_2_pos.append(gt_label)
			pre_2_pos.append(prediction[i])
		if torch.sum(gt_label) == 3:
			gt_3_pos.append(gt_label)
			pre_3_pos.append(prediction[i])
		if torch.sum(gt_label) == 4:
			gt_4_pos.append(gt_label)
			pre_4_pos.append(prediction[i])

	gt_1_pos, pre_1_pos, gt_2_pos, pre_2_pos = \
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
            tp[i] = (sorted_label[i] > 0)
            fp[i] = (sorted_label[i] <= 0)
        true_num = sum(tp)
        fp = np.cumsum(fp)
        tp = np.cumsum(tp)
        rec = tp / true_num
        prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
        ap = voc_ap(rec, prec)
        aps += [ap]

    np.set_printoptions(precision=3, suppress=True)
    mAP = np.mean(aps)
    return mAP

def voc_ap(rec, prec):
	mrec = np.concatenate(([0.], rec, [1.]))
	mpre = np.concatenate(([0.], prec, [0.]))
	for i in range(mpre.size - 1, 0, -1):
		mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
	i = np.where(mrec[1:] != mrec[:-1])[0]
	ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
	return ap

def calculate_indexes(y_true, y_score,args):
    print(f"The Ground true & prediected of training set")
    print(f"y_true：{y_true}")
    print(f"y_score：{y_score}")
    # , , ,,
    # ,accuracy_score,recall_score,average_precision_score,precision_score
    y_true = y_true.cpu().detach()
    y_score= y_score.cpu().detach()
    y_pred = copy.deepcopy(y_score)

    y_pred[y_pred >= args.threshold] = 1
    y_pred[y_pred < args.threshold] = 0
    print(f"y_pred：{y_pred}")

    haming_loss = hamming_loss(y_true, y_pred)

    Label_Ranking_loss = label_ranking_loss(y_true, y_score)

    one_error = One_Error(y_score, y_true)

    Coverage = coverage_error(y_true, y_score)- 1

    Label_Ranking_Average_PrecisionScore = label_ranking_average_precision_score(y_true,y_score)

    # Macro_F1
    Macro_F1 = f1_score(y_true, y_pred, average='macro')
    # Micro_F1
    Micro_F1 = f1_score(y_true, y_pred, average='micro')

    # Subset_Accuracy
    Subset_Accuracy = subset_accuracy(y_pred,y_true)

    accuracyScore = accuracy_score(y_true, y_pred)
    Recall_Score = recall_score(y_true,y_pred,average='macro')
    Precision_Score = precision_score(y_true,y_pred,average='macro')

    Average_Precision_Score = average_precision_score(y_true, y_score)

    indexes_paper = {}
    indexes_paper["haming_loss"] = haming_loss
    indexes_paper["ranking_loss"] = Label_Ranking_loss
    indexes_paper["one_error"] = one_error
    indexes_paper["coverage"] = Coverage
    indexes_paper["average_precision"] = Label_Ranking_Average_PrecisionScore
    indexes_paper["micro_F1"] = Micro_F1
    indexes_paper["subset_accuracy"] = Subset_Accuracy
    indexes_paper["macro_F1"] = Macro_F1

    indexes_mine = {}
    indexes_mine["accuracyScore"] = accuracyScore
    indexes_mine["Recall_Score"] = Recall_Score
    indexes_mine["Precision_Score"] = Precision_Score
    indexes_mine["Average_Precision_Score"] = Average_Precision_Score


    return indexes_paper,indexes_mine

def compute_class_weights(gt):
    smooth_factor = 1e-6
    class_counts = torch.mean(gt, dim=0)
    class_frequencies = class_counts / gt.size(0) + smooth_factor
    class_weights = 1.0 / class_frequencies + smooth_factor
    class_weights_normalized = class_weights / torch.sum(class_weights)
    return class_weights_normalized

def batch(train_view, Y_traina, test_view, Y_test, dim_view, model,args,target,target_embed,num):

    save_path = "./Model_Best"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    model = model
    for layer in model.modules():
        if isinstance(layer, torch.nn.Linear):
            init.kaiming_uniform_(layer.weight, a=math.sqrt(5))
            if layer.bias is not None:
                fan_in, _ = init._calculate_fan_in_and_fan_out(layer.weight)
                bound = 1 / math.sqrt(fan_in)
                init.uniform_(layer.bias, -bound, bound)

    best_score = 0
    fold_Haming_loss, fold_Ranking_loss, fold_One_error, fold_Coverage, fold_Average_precision, fold_Micro_F1, fold_Subset_accuracy, fold_Macro_F1 = 0, 0, 0, 0, 0, 0, 0, 0
    if args.resume is not None:
        print(f"args.resume:{args.resume}")  # D:/Machine_Learning_Dataset/Model_Saved/scene_model_best.pth
        split_path = args.resume.split('/')
        name = split_path[-1]
        fold_number = int(name.split('_')[0])
        datasetnmae = name.split('_')[2]
        if os.path.isfile(args.resume) and args.dataset.split('.')[0] == datasetnmae and fold_number == num:
            print(f"loading {args.dataset.split('.')[0]} dataset {fold_number}th-fold train result")
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            # #################
            best_score = checkpoint['best_score']
            fold_Average_precision = checkpoint['best_score']
            fold_Haming_loss = checkpoint['fold_Haming_loss']
            fold_Ranking_loss = checkpoint['fold_Ranking_loss']
            fold_One_error = checkpoint['fold_One_error']
            fold_Coverage = checkpoint['fold_Coverage']
            fold_Micro_F1 = checkpoint['fold_Micro_F1']
            fold_Subset_accuracy = checkpoint['fold_Subset_accuracy']
            fold_Macro_F1 = checkpoint['fold_Macro_F1']
            # #################
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint best_score '{}' (epoch {})"
                  .format(best_score, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    criterion = torch.nn.BCELoss()

    optimizer = torch.optim.SGD(params=model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.005)#SGD

    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.933)#gamma=0.941,0.933

    if torch.cuda.is_available():
        print("CUDA available")
        model = model.cuda()
        print("Number of GPUs available:", torch.cuda.device_count())
        model = torch.nn.DataParallel(model)
        criterion.cuda()
    else:
        print("CUDA unavailable")
    gt_embed = target_embed  # [6,128]
    stacked_train_view = torch.stack(train_view, dim=0)#troch.size(2,1924,128)
    reshaped_train_view = torch.transpose(stacked_train_view, 0, 1)#[1924,2,128]
    #[[[],[]],[[],[]]]
    stacked_Y_train = torch.stack(Y_traina)#troch.size(2,1947,6)
    reshaped_Y_train = torch.transpose(stacked_Y_train, 0, 1)#troch.size(1924,2,6)
    trainset =  TensorDataset(reshaped_train_view, reshaped_Y_train)

    stacked_test_view = torch.stack(test_view, dim=0)
    stacked_Y_test = torch.stack(Y_test)
    reshaped_test_view = torch.transpose(stacked_test_view, 0, 1)
    reshaped_Y_test = torch.transpose(stacked_Y_test, 0, 1)
    valset = TensorDataset(reshaped_test_view,reshaped_Y_test)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=False)
    valloader = torch.utils.data.DataLoader(valset, batch_size=args.batch_size, shuffle=False)

    print("taning begin-100 epoches")
    category_num = args.category_num
    best_prec = best_score

    Haming_loss = []
    Ranking_loss=[]
    One_error = []
    Coverage = []
    Average_precision = []
    Micro_F1 = []
    Subset_accuracy = []
    Macro_F1 = []
    Recall_Score,Accuracy_Score, Precision_Score, Average_Precision_Score = [],[],[],[]
    ####################
    Haming_loss_T = []
    Ranking_loss_T = []
    One_error_T = []
    Coverage_T =[]
    Average_precision_T = []
    Micro_F1_T =[]
    Subset_accuracy_T =[]
    Macro_F1_T =[]
    Recall_Score_T, Accuracy_Score_T, Precision_Score_T, Average_Precision_Score_T = [], [], [], []


    fold_Accuracy_Score ,fold_Recall_Score ,fold_Precision_Score ,fold_Average_Precision_Score = 0,0,0,0


    indices_views = []
    knn = NearestNeighbors(n_neighbors=args.k + 1,metric='cosine')#,metric='cosine'

    for view_sample_feature in stacked_train_view:# [2,1947,128]
        knn.fit(view_sample_feature.cpu().detach())  # [1947,128]
        _, indices = knn.kneighbors(view_sample_feature.cpu().detach())
        indices_views.append(torch.tensor(indices[:, 1:]))
    indices_views = torch.stack(indices_views)  # [2,1947, 3]

    indices_views_test = []
    knn = NearestNeighbors(n_neighbors=args.k + 1,metric='cosine')#,metric='cosine'

    for view_sample_feature in stacked_test_view:  # [2,481,128]
        knn.fit(view_sample_feature.cpu().detach())  # [481,128]
        _, indices = knn.kneighbors(view_sample_feature.cpu().detach())
        # indices_views_test.append(torch.tensor(indices))
        indices_views_test.append(torch.tensor(indices[:, 1:]))
    indices_views_test = torch.stack(indices_views_test)  # [2,481, 3]

    indices_views_temp = []
    indices_views_test_temp = []
    reshaped_train_view_temp = []
    reshaped_test_view_temp = []
    for i in range(args.GPUS):
        indices_views_temp.append(indices_views)
        indices_views_test_temp.append(indices_views_test)
        reshaped_train_view_temp.append(reshaped_train_view)
        reshaped_test_view_temp.append(reshaped_test_view)
    indices_views_temp = torch.stack(indices_views_temp)
    indices_views_test_temp = torch.stack(indices_views_test_temp)
    reshaped_train_view_temp = torch.stack(reshaped_train_view_temp)
    reshaped_test_view_temp = torch.stack(reshaped_test_view_temp)

    running_loss_return = []
    running_loss_T = []
    for epoch in range(args.end_epochs):
        print(f"{epoch+1} th training")
        running_loss = []
        results = {'predict': [], 'label': []}
        Y_train = copy.deepcopy(Y_traina)

        batch_num = 0
        for idx, (inputs, gt) in enumerate(trainloader):
            optimizer.zero_grad()
            gt_temp = torch.transpose(copy.deepcopy(gt), 0, 1).cuda()
            # gt = torch.transpose(gt, 0, 1)#torch.size(2,32,6)
            gt = gt.cuda()
            cross_edges, counts, = model(inputs,gt_embed,gt,args,indices_views_temp,reshaped_train_view_temp,batch_num)
            batch_num = batch_num+1
            # gt_embedding[ == -1] = 0
            count_cur, loss = 0, 0
            prediction_score = list()
            for j, count in enumerate(counts):
                prediction = cross_edges[count_cur: count_cur + (count[0] * count[1])]  # (6,64)
                prediction = prediction.view(-1, category_num)#6
                # prediction =  torch.max(prediction, dim=0)[0]
                prediction = torch.mean(prediction, dim=0)
                prediction_score.append(prediction.unsqueeze(0))
                count_cur += (count[0] * count[1])
            prediction = torch.cat(prediction_score, dim=0)
            # #######################
            # weight_matrix = class_weights.unsqueeze(0).expand_as(prediction).cuda()
            # prediction = prediction * weight_matrix
            # #######################
            results['predict'].extend([x.tolist() for x in prediction])
            results['label'].extend([y.tolist() for y in gt_temp[0]])

            # weight = 0.080
            loss = weighted_cross_entropy(prediction=prediction,gt=gt_temp[0], weight = 0.8,model=model)#0.80
            loss.backward()
            optimizer.step()

            running_loss.append(loss.item())

        running_loss_return.append(np.mean(np.array(running_loss)).item())
        print(f"The Average loss of this epoch：{running_loss_return[epoch]}")
        print("*" * 66)

        rate_tp, rate_tn = cal_metric(results['label'], results['predict'])
        ICCV_train_acc = iccv2019_mAP(results['label'], results['predict'])

        print("Epoch:", epoch + 1, "\ttrain_tp_rate:", round(rate_tp.item(), 4), "\ttrain_tn_rate:",
              round(rate_tn.item(), 4),
              'ICCV_train_acc:', round(ICCV_train_acc, 4))
        print("*" * 66)

        indexes_paper,indexes_mine= calculate_indexes(
            Y_train[0], torch.tensor(results['predict']),args)
        print(
            f"training set：indicators of {epoch + 1} -th of this epoch:8 items in the paper: haming_loss:{indexes_paper['haming_loss']}, Ranking_loss:{indexes_paper['ranking_loss']}, one_error:{indexes_paper['one_error']}, Coverage:{indexes_paper['coverage']}, Average_Precision:{indexes_paper['average_precision']}, Micro_F1:{indexes_paper['micro_F1']}, Subset_Accuracy:{indexes_paper['subset_accuracy']}, Macro_F1:{indexes_paper['subset_accuracy']}")

        this_time_train = indexes_paper['average_precision']
        Haming_loss.append(indexes_paper['haming_loss'])
        Ranking_loss.append(indexes_paper['ranking_loss'])
        One_error.append(indexes_paper['one_error'])
        Coverage.append(indexes_paper['coverage'])
        Average_precision.append(indexes_paper['average_precision'])
        Micro_F1.append(indexes_paper['micro_F1'])
        Subset_accuracy.append(indexes_paper['subset_accuracy'])
        Macro_F1.append(indexes_paper['macro_F1'])

        Accuracy_Score.append(indexes_mine["accuracyScore"])
        Recall_Score.append(indexes_mine["Recall_Score"])
        Precision_Score.append(indexes_mine["Precision_Score"])
        Average_Precision_Score.append(indexes_mine["Average_Precision_Score"])

        print("*" * 66)

        model.eval()
        torch.set_grad_enabled(False)
        test_tp, test_tn, ICCV_test_acc ,indexes,running_loss_test_return= testa(valloader,model,category_num,target,Y_test,gt_embed,args,indices_views_test_temp,reshaped_test_view_temp)
        running_loss_T.append(running_loss_test_return)
        print("Epoch:", epoch + 1, "\ttest_tp_rate:", round(test_tp.item(), 4), "\ttest_tn_rate:",
              round(test_tn.item(), 4),
              'ICCV_test_acc:', round(ICCV_test_acc, 4))

        Haming_loss_T.append(indexes['paper']["haming_loss"])
        Ranking_loss_T.append(indexes['paper']["ranking_loss"])
        One_error_T.append(indexes['paper']["one_error"])
        Coverage_T.append(indexes['paper']["coverage"])
        Average_precision_T.append(indexes['paper']["average_precision"])
        Micro_F1_T.append(indexes['paper']["micro_F1"])
        Subset_accuracy_T.append(indexes['paper']["subset_accuracy"])
        Macro_F1_T.append(indexes['paper']["macro_F1"])

        Accuracy_Score_T.append(indexes['mine']["accuracyScore"])
        Recall_Score_T.append(indexes['mine']["Recall_Score"])
        Precision_Score_T.append(indexes['mine']["Precision_Score"])
        Average_Precision_Score_T.append(indexes['mine']["Average_Precision_Score"])
        print("*" * 66)


        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        print(f"current lr:：{current_lr}")

        print("Start checking if it is the best model")
        this_time = indexes['paper']["average_precision"]

        is_best = indexes['paper']["average_precision"] > best_prec
        last_best = min(indexes['paper']["average_precision"], best_prec)
        best_prec = max(indexes['paper']["average_precision"], best_prec)
        print(f"The results of this eopch in training set: {this_time_train:0.4f}\n")
        print(f"The results of this eopch in testing set: {this_time:0.4f}\n")
        if is_best:
            print(f"last best: {last_best:0.4f}\n")
        print(f"best_prec: {best_prec:0.4f}\n")
        print("*" * 66)
        if is_best:
            print('--------------------------model_best----------------')
            model_best = '_model_best.pth'
            model_best_path = save_path + str(num) + '_fold_' + args.dataset.split('.')[0] + model_best
            torch.save({
                'epoch': epoch,
                'state_dict': model.module.state_dict() if torch.cuda.is_available() else model.state_dict(),
                'best_score': best_prec,
                'fold_Haming_loss':indexes['paper']["haming_loss"],
                'fold_Ranking_loss':indexes['paper']["ranking_loss"],
                'fold_One_error':indexes['paper']["one_error"],
                'fold_Coverage':indexes['paper']["coverage"],
                'fold_Micro_F1':indexes['paper']["micro_F1"],
                'fold_Subset_accuracy':indexes['paper']["subset_accuracy"],
                'fold_Macro_F1':indexes['paper']["macro_F1"]
            }, model_best_path)
            fold_Haming_loss = indexes['paper']["haming_loss"]
            fold_Ranking_loss = indexes['paper']["ranking_loss"]
            fold_One_error = indexes['paper']["one_error"]
            fold_Coverage = indexes['paper']["coverage"]
            fold_Average_precision = indexes['paper']["average_precision"]
            fold_Micro_F1 = indexes['paper']["micro_F1"]
            fold_Subset_accuracy = indexes['paper']["subset_accuracy"]
            fold_Macro_F1 = indexes['paper']["macro_F1"]

            # DIV
            fold_Accuracy_Score = indexes['mine']["accuracyScore"]
            fold_Recall_Score = indexes['mine']["Recall_Score"]
            fold_Precision_Score = indexes['mine']["Precision_Score"]
            fold_Average_Precision_Score =indexes['mine']["Average_Precision_Score"]
        # args.resume = model_best_path

        # Training mode
        model.train()
        torch.set_grad_enabled(True)
    print("After 100 training sessions, return eight indicators")

    result_train = {'Haming_loss':Haming_loss,'Ranking_loss':Ranking_loss ,'One_error':One_error ,'Coverage':Coverage ,'Average_precision':Average_precision ,'Micro_F1':Micro_F1 ,'Subset_accuracy':Subset_accuracy ,'Macro_F1':Macro_F1}
    result_test = {'Haming_loss_T':Haming_loss_T,'Ranking_loss_T':Ranking_loss_T ,'One_error_T':One_error_T ,'Coverage_T':Coverage_T ,'Average_precision_T':Average_precision_T ,'Micro_F1_T':Micro_F1_T ,'Subset_accuracy_T':Subset_accuracy_T ,'Macro_F1_T':Macro_F1_T}

    # fold_Recall_Score = sum(Recall_Score) / len(Recall_Score)
    # fold_Haming_loss = sum(Haming_loss) / len(Haming_loss)
    # fold_Ranking_loss = sum(Ranking_loss) / len(Ranking_loss)
    # fold_One_error = sum(One_error) / len(One_error)
    # fold_Coverage = sum(Coverage) / len(Coverage)
    # fold_Average_precision = sum(Average_precision) / len(Average_precision)
    # fold_Micro_F1 = sum(Micro_F1) / len(Micro_F1)
    # fold_Subset_accuracy = sum(Subset_accuracy) / len(Subset_accuracy)
    # fold_Macro_F1 = sum(Macro_F1) / len(Macro_F1)

    fold_Haming_loss_T = sum(Haming_loss_T) / len(Haming_loss_T)
    fold_Ranking_loss_T = sum(Ranking_loss_T) / len(Ranking_loss_T)
    fold_One_error_T = sum(One_error_T) / len(One_error_T)
    fold_Coverage_T = sum(Coverage_T) / len(Coverage_T)
    fold_Average_precision_T = sum(Average_precision_T) / len(Average_precision_T)
    fold_Micro_F1_T = sum(Micro_F1_T) / len(Micro_F1_T)
    fold_Subset_accuracy_T = sum(Subset_accuracy_T) / len(Subset_accuracy_T)
    fold_Macro_F1_T = sum(Macro_F1_T) / len(Macro_F1_T)

    fold_Accuracy_Score_T = sum(Accuracy_Score_T) / len(Accuracy_Score_T)
    fold_Recall_Score_T = sum(Recall_Score_T) / len(Recall_Score_T)
    fold_Precision_Score_T = sum(Precision_Score_T) / len(Precision_Score_T)
    fold_Average_Precision_Score_T = sum(Average_Precision_Score_T) / len(Average_Precision_Score_T)

    print(f"fold_Haming_loss:{fold_Haming_loss}")
    print(f"fold_Ranking_loss:{fold_Ranking_loss}")
    print(f"fold_One_error:{fold_One_error}")
    print(f"fold_Coverage:{fold_Coverage}")
    print(f"fold_Average_precision:{fold_Average_precision}")
    print(f"fold_Micro_F1:{fold_Micro_F1}")
    print(f"fold_Subset_accuracy:{fold_Subset_accuracy}")
    print(f"fold_Macro_F1:{fold_Macro_F1}")
    # print(f"fold_average_accuracy_T:{fold_Accuracy_Score}")
    # print(f"fold_Recall_Score_T:{fold_Recall_Score}")
    # print(f"fold_Precision_Score_T:{fold_Precision_Score}")
    # print(f"fold_Average_Precision_Score_T:{fold_Average_Precision_Score}")

    return fold_Haming_loss,fold_Ranking_loss,fold_One_error,fold_Coverage ,fold_Average_precision,fold_Micro_F1,fold_Subset_accuracy,fold_Macro_F1,result_train,result_test,running_loss_return,running_loss_T

def process_fold(args, data, target, gt_embed ,fold_index):
    torch.set_num_threads(1)
    k = args.k_fold
    args.resume = './Model_Best/' + str(fold_index + 1) + '_fold_' + args.dataset.split('.')[0] + '_model_best.pth'
    train_view = []
    test_view = []

    Y_train = []
    Y_test = []

    model = _Model(dim_node_ins=args.nhidden, dim_node_label=args.target_dim, dim_edge_ins=args.nhidden * 2,
                   dim_edge_label=args.target_dim * 2)
    dim_view = []

    data, target = shuffle(data, target ,args.seed)

    for j in range(len(data)):
        print('i:{} k:{} j:{} len:{}'.format(fold_index, k, j, len(data)))

        train_v, y_train, test_v, y_test = get_k_fold_data(k, fold_index, data[j], target)

        print(
            f"The data of the {j + 1} view obtained from the {fold_index} fold: train_v:{train_v.shape},y_train:{y_train.shape}, test_v:{test_v.shape}, y_test:{y_test.shape}")

        train_view.append(train_v)
        Y_train.append(y_train)
        test_view.append(test_v)
        Y_test.append(y_test)

        dim_view.append(test_v.shape[1])
    print(f"{fold_index + 1}-th fold training")

    ham_b, rank_b, one_b, cov_b, avp_b, mif1_b, sa_b, maf1_b, result_train, result_test,running_loss_return,running_loss_test_return= batch(
        train_view,
        Y_train,
        test_view,
        Y_test,
        dim_view,
        model,
        args, target, gt_embed, fold_index + 1)

    return ham_b, rank_b, one_b, cov_b, avp_b, mif1_b, sa_b, maf1_b, result_train, result_test, running_loss_return,running_loss_test_return

def k_fold(args,data,target):

    args = args
    k = args.k_fold
#    gt_embed = compute_embedding(data, target)
    # gt_embed = one_hot(target, args)
    gt_embed = one_hot2(target, args)

    gt_embed_temp = []
    for i in range(args.GPUS):
        gt_embed_temp.append(gt_embed)
    gt_embed = torch.stack(gt_embed_temp)
    mp.set_start_method('spawn')
    pool = mp.Pool(processes=k)
    results = []

    for i in range(k):
        results.append(pool.apply_async(process_fold, (args, data, target, gt_embed, i)))
    pool.close()
    pool.join()

    train_loss_sum, test_loss_sum = 0, 0
    train_acc_sum, test_acc_sum = 0, 0
    acc_cnt_sum = 0

    train_a = []
    train_l = []
    test_a = []
    test_l = []

    ham_sum, rank_sum, one_sum, mif1_sum, cov_sum, sa_sum, maf1_sum, avp_sum = 0, 0, 0, 0, 0, 0, 0, 0
    ham_l, rank_l, one_l, mif1_l, cov_l = [], [], [], [], []
    sa_l = []
    maf1_l = []
    avp_l = []
    result_train_ap = []
    result_test_ap = []
    result_test_hl = []
    result_test_rl = []
    result_test_oe = []
    result_test_cov = []
    result_test_micro = []
    result_test_suba = []
    result_test_macro = []
    
    result_loss = []
    result_loss_T = []
    for result in results:
        ham_b, rank_b, one_b, cov_b, avp_b, mif1_b, sa_b, maf1_b, result_train, result_test,running_loss_return,running_loss_test_return = result.get()
        ham_sum += ham_b
        rank_sum += rank_b
        one_sum += one_b
        mif1_sum += mif1_b
        cov_sum += cov_b
        sa_sum += sa_b
        maf1_sum += maf1_sum
        avp_sum += avp_sum

        ham_l.append(ham_b)
        rank_l.append(rank_b)
        one_l.append(one_b)
        mif1_l.append(mif1_b)
        cov_l.append(cov_b)
        sa_l.append(sa_b)
        maf1_l.append(maf1_b)
        avp_l.append(avp_b)

        result_train_ap.append(result_train['Average_precision'])
        
        result_test_ap.append(result_test['Average_precision_T'])
        result_test_hl.append(result_test['Haming_loss_T'])
        result_test_rl.append(result_test['Ranking_loss_T'])
        result_test_oe.append(result_test['One_error_T'])
        result_test_cov.append(result_test['Coverage_T'])
        result_test_micro.append(result_test['Micro_F1_T'])
        result_test_suba.append(result_test['Subset_accuracy_T'])
        result_test_macro.append(result_test['Macro_F1_T'])

        result_total = {'result_test_hl':result_test_hl,'result_test_rl':result_test_rl ,'result_test_oe':result_test_oe ,'result_test_cov':result_test_cov ,'result_test_ap':result_test_ap ,'result_test_micro':result_test_micro ,'result_test_suba':result_test_suba,'result_test_macro':result_test_macro}
        
        result_loss.append(running_loss_return)
        result_loss_T.append(running_loss_test_return)

    print(f"K-fold ending, calculate mean and variance")
    ap_mean,_ = print_std(avp_l, 'ap')
    hl_mean,_ = print_std(ham_l, 'hamming loss')
    rl_mean,_ = print_std(rank_l, 'ranking loss')
    oe_mean,_ = print_std(one_l, 'one error')
    cov_mean,_ = print_std(cov_l, 'coverage')
    mif1_mean,_ = print_std(mif1_l, 'micro-F1')
    sa_mean,_ = print_std(sa_l, 'subset accuracy')
    maf1_mean,_ = print_std(maf1_l, 'macro_f1')

    plot_training_results(result_train_ap, result_test_ap, result_loss,result_loss_T,k,ap_mean,args,result_total)

    train_acc_sum /= k
    test_acc_sum /= k
    train_loss_sum /= k
    test_loss_sum /= k
    acc_cnt_sum /= k

    sa_sum /= k
    ham_sum /= k
    rank_sum /= k
    one_sum /= k
    mif1_sum /= k
    cov_sum /= k
    maf1_sum /= k
    avp_sum /= k

    print('train_acc:{} train_loss:{}'.format(train_acc_sum, train_loss_sum))
    print('test_acc:{} test_loss:{}:'.format(test_acc_sum, test_loss_sum))
    print('acc_cnt:{} ham:{} rank:{} one:{} mif1:{} cov:{} avp:{} maf1:{} sa:{}'.format(acc_cnt_sum, ham_sum, rank_sum, one_sum, mif1_sum,
                                                                   cov_sum,avp_sum,maf1_sum,sa_sum))


    return train_acc_sum, train_loss_sum, test_acc_sum, test_loss_sum, acc_cnt_sum, ham_sum, rank_sum, one_sum


def encoder(view,target,args):
    view_dict = {}

    for i in range(len(view)):
        view_name = 'view' + str(i)
        view_feature = view[i].float()
        view_dict.update({view_name: view_feature})


    view_names = list(view_dict.keys())
    view_data = [torch.Tensor(view_dict[view]) for view in view_names]

    input_dims = [view.shape[1] for view in view_data]
    autoencoders = [Encoder(input_dim, args.nhidden, target.shape[1], 2, 0.2) for input_dim in input_dims]

    criterion_encoder = nn.MSELoss()
    # criterion_encoder = nn.L1Loss()
    optimizers_encoder = [optim.Adam(autoencoder.parameters(), lr=0.001) for autoencoder in autoencoders]

    datasets = [TensorDataset(view, view) for view in view_data]


    batch_size = 32
    dataloaders = [DataLoader(dataset, batch_size=batch_size, shuffle=True) for dataset in datasets]

    print("Start training the autoencoder for each view：")
    num_epochs = 10
    for epoch in range(num_epochs):

        for dataloader, autoencoder, optimizer in zip(dataloaders, autoencoders, optimizers_encoder):
            for data in dataloader:
                inputs, targets = data

                optimizer.zero_grad()
                encoded, decoded = autoencoder(inputs)
                loss = criterion_encoder(decoded, targets)
                loss.backward()
                optimizer.step()
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')


    checkpoint = {}
    for i in range(len(view)):
        model_state_dict = autoencoders[i].state_dict(),
        optimizer_state_dict = optimizers_encoder[i].state_dict()
        checkpointi = {'model_state_dict': model_state_dict, 'optimizer_state_dict': optimizer_state_dict}
        checkpoint.update({'encoder_view' + str(i): checkpointi})
    save_path = 'record/encoder.pth'
    directory = os.path.dirname(save_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

    torch.save(checkpoint, save_path)

    with torch.no_grad():
        encoded_representations = [autoencoder.encoder(view) for view, autoencoder in zip(view_data, autoencoders)]
        for i in range(len(encoded_representations)):
            print(f"encoded_representations:{encoded_representations[i]}")

        print(f"encoded_representations:{type(encoded_representations)}")
    return encoded_representations

def subset_accuracy(predict, target):
    return ((predict == target).sum(1) == target.shape[1]).sum() / predict.shape[0]

def One_Error(logits, labels):
    _, indices = torch.max(logits, dim=1)
    cnt1 = 0
    for i in range(logits.shape[0]):
        cnt1 += (labels[i][indices[i]] == 0)
    
    return cnt1.item() / logits.shape[0]

