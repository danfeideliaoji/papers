# VAMS 修正版入口：读取原始数据 → 分折 → 训练集预处理 → 验证选模 → 最终测试。
# 启动脚本调用本文件；当前可运行数据为 Emotions，两个视图分别有 64 和 8 个特征。

import argparse
import random
import time
import numpy as np
import torch
import h5py
from skmultilearn.dataset import load_dataset
import scipy.io as sio
import os
from util import k_fold
from sklearn.preprocessing import RobustScaler
import numpy as np


# 读取 ARFF 训练/测试数据并拼接；返回样本特征与多标签矩阵。
def load_arff(dataset):
    X, y, feature_names, label_names = load_dataset(dataset, 'train')
    Xt, yt, feature_namest, label_namest = load_dataset(dataset, 'test')
    xm = torch.cat([torch.from_numpy(X.toarray()), torch.from_numpy(Xt.toarray())], dim=0)
    ym = torch.cat([torch.from_numpy(y.toarray()), torch.from_numpy(yt.toarray())], dim=0)
    return xm, ym


# 兼容普通 MAT、HDF5 MAT 和 scene 的 ARFF 读取方式。
# 每个视图最终应为 [样本数 N, 该视图特征数 D_v]；标签方向由 train() 统一。
def data_loader(path):
    print('loading the mat dataset')
    if args.dataset == 'Iaprtc12.mat' or args.dataset == 'Espgame.mat' or args.dataset == 'Mirflickr.mat':
        mat = h5py.File(path, 'r')
        view = []
        for i in range(6):
            los = mat['data'][0][i]
            dat = np.array(mat[los])
            view.append(torch.from_numpy(dat).transpose(1, 0))
        da = np.array(mat['target'])
        target = torch.from_numpy(da)
    elif args.dataset == 'scene':
        xm, ym = load_arff(args.dataset)
        view = []
        view.append(xm[:, 0:98])
        view.append(xm[:, 98:])
        target = ym
    else:
        mat = sio.loadmat(path)
        print('load ending')
        view = []
        target = mat['target']
        print('view data:', mat['data'].shape)
        for i in range(mat['data'].shape[0]):
            for j in range(mat['data'].shape[1]):
                print('i:{} j:{} shape:{}'.format(i, j, mat['data'][i][j].shape))
        if mat['data'].shape[0] == 1:
            for i in range(mat['data'].shape[1]):
                view.append(mat['data'][0][i].astype(np.float64))
        elif mat['data'].shape[1] == 1:
            for i in range(mat['data'].shape[0]):
                view.append(mat['data'][i][0].astype(np.float64))
        else:
            print('View data type error!')
        print("Generate torch data from view data (numpy) & print")
        for i in range(len(view)):
            t_view = torch.from_numpy(view[i])
            if target.shape[1] == t_view.shape[1] and target.shape[0] != t_view.shape[0]:
                t_view = t_view.permute(1, 0)
            view[i] = t_view
    for i in range(len(view)):
        print('i:{} shape:{}'.format(i, view[i].shape))
    return view, target


# 固定 PyTorch、CUDA、NumPy 与 Python 随机种子，便于重放随机输入投影。
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


# 不在这里归一化或冻结随机投影；所有拟合步骤在训练折内部进行。
def train():
    views, target = data_loader(os.path.join(args.data_root, args.dataset))
    target = torch.as_tensor(target)
    # ARFF 返回 [N,L]，MAT 通常返回 [L,N]；统一后检查样本数。
    if target.shape[0] != views[0].shape[0]:
        target = target.T
    return k_fold(args, views, target)


if __name__ == '__main__':
    # 下方为运行参数：device 选择设备，fresh_start 禁用旧模型读取，checkpoint_dir 指定模型目录。
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--checkpoint_dir', default='./results_corrected/checkpoints')
    parser.add_argument('--fresh_start', action='store_true', help='Do not load previous model parameters')
    parser.add_argument('--dataset', default='emotions.mat', type=str, metavar='N', help='run_data')
    parser.add_argument('--data_root', default='./data', type=str, metavar='PATH', help='root dir')
    parser.add_argument('--batch_size', default=64, type=int, help='number of batch size')  # 64
    parser.add_argument('--category_num', default=6, type=int,
                        help='category_num')
    parser.add_argument('--k', default=5, type=int, help='KNN')
    parser.add_argument('--GPUS', default=1, type=int, help='The number of GPU')
    # 论文报告五折；每个外层训练折另外留出验证集。
    parser.add_argument('--k_fold', default=5, type=int, metavar='H-P', help='n_hidden')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--end_epochs', default=80, type=int, metavar='H-P',
                        help='number of total epochs to run')
    # 修正版可训练投影与频率损失的梯度尺度变大，0.01 在第一折引发常数输出。
    # 固定划分单因素验证：0.001 恢复可区分预测；完整结果仍须另行评估。
    parser.add_argument('--lr', default=0.001, type=float,
                        metavar='H-P', help='initial learning rate')  # 0.75
    parser.add_argument('--nhidden', default=128, type=int,
                        metavar='H-P', help='n_hidden')
    parser.add_argument('--target_dim', default=128, type=int, help='KNN')
    parser.add_argument('--dropout', default=0.4, type=float,
                        metavar='H-P', help='n_hidden')
    parser.add_argument('--threshold', default=0.50, type=float,
                        metavar='H-P', help='threshold')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--warmup', default=0, type=int,
                        metavar='H-P', help='n_hidden')
    parser.add_argument('--t', default=0.85, type=float,
                        metavar='H-P', help='n_hidden')
    parser.add_argument('--at', default=0.4, type=float,
                        metavar='H-P', help='n_hidden')
    parser.add_argument('--seed', default=2020, type=int,
                        metavar='H-P', help='n_hidden')

    # 限制当前进程使用第一张可见显卡；当前运行路径只验证了单 GPU。
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'

    parser.add_argument('--output_dir', default='./results_corrected', help='新运行输出目录，必须为空')
    parser.add_argument('--fold_index', type=int, default=None, help='仅运行指定折（从 1 开始），仍保留 k_fold 划分')
    parser.add_argument('--dev_fraction', type=float, default=0.2, help='外层训练部分的验证比例')
    parser.add_argument('--beta', type=float, default=1.0, help='论文式 11 权重系数；论文未给数值，默认 1')
    args = parser.parse_args()
    best_prec = 0
    category_num = 0
    DATAPATH = os.path.join(args.data_root, args.dataset)
    setup_seed(args.seed)
    print('train begin')
    train()




