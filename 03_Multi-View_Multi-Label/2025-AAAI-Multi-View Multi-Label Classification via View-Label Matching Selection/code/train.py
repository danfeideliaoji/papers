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
from MLP import merge_views
from sklearn.preprocessing import RobustScaler
import numpy as np

def load_arff(dataset):
    X, y, feature_names, label_names = load_dataset(dataset, 'train')
    Xt, yt, feature_namest, label_namest = load_dataset(dataset, 'test')
    xm = torch.cat([torch.from_numpy(X.toarray()), torch.from_numpy(Xt.toarray())], dim=0)
    ym = torch.cat([torch.from_numpy(y.toarray()), torch.from_numpy(yt.toarray())], dim=0)
    return xm, ym
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
            view[i] = torch.from_numpy(view[i])
    for i in range(len(view)):
        print('i:{} shape:{}'.format(i, view[i].shape))
    return view, target

def data_clean(view, target):
    print("data clean")
    eps = 1e-5
    # Z-score
    # for i in range(len(view)):
    #     valid_col = []
    #     # print('less:', (view[i] < 0).sum())
    #     for j in range(view[i].shape[1]):
    #         if not (view[i][:, j] == 0).sum() == view[i][:, j].shape[0]:
    #             valid_col.append(j)
    #         view[i][:, j] = (view[i][:, j] - view[i][:, j].mean()) / (torch.std((view[i][:, j])) + eps)
    # MinMax
    for i in range(len(view)):
        for j in range(view[i].shape[1]):
            if not (view[i][:, j] == 0).sum() == view[i][:, j].shape[0]:
                min_val = view[i][:, j].min()
                max_val = view[i][:, j].max()
                view[i][:, j] = (view[i][:, j] - min_val) / (max_val - min_val + eps)
    return view, target

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def train():
    print('data load begin')
    view, target = data_loader(args.data_root + args.dataset)
    print('data load end')
    target = torch.tensor(target).permute(1, 0)
    # target = torch.tensor(target)
    view, target = data_clean(view, target)  # ******************
    view = merge_views(view, args)  # encoded_view = encoder(view, target, args)


    print('view:{} target:{}'.format(len(view), target.shape))

    t_start = time.time()
    train_acc_sum, train_loss_sum, test_acc_sum, test_loss_sum, acc_cnt_sum, ham_sum, rank_sum, one_sum = k_fold(args,
                view, target)

    # best = max(test_acc_sum)
    ts = time.time() - t_start
    # print('best:', best)
    print(f'time:{ts}')

if __name__ == '__main__':
    '''
    emotions(6),plant(12),scene(6),Yeast(14),human(14),Pascal(20),Corel5k(260),Espgame(268),Iaprtc12(291),Mirflickr(457)
    '''
    # /app/users/MyExperiment/data/
    
    dataset = 'scene.mat'
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('--dataset', default=dataset, type=str, metavar='N', help='run_data')
    parser.add_argument('--data_root', default='./data', type=str, metavar='PATH', help='root dir')
    parser.add_argument('--batch_size', default=64, type=int, help='number of batch size')  # 64
    parser.add_argument('--category_num', default=6, type=int,
                        help='category_num')
    parser.add_argument('--k', default=5, type=int, help='KNN')
    parser.add_argument('--GPUS', default=1, type=int, help='The number of GPU')
    parser.add_argument('--k_fold', default=6, type=int, metavar='H-P', help='n_hidden')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--end_epochs', default=80, type=int, metavar='H-P',
                        help='number of total epochs to run')
    parser.add_argument('--lr', default=0.01, type=float,
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

    os.environ["CUDA_VISIBLE_DEVICES"] = '0'

    args = parser.parse_args()
    best_prec = 0
    category_num = 0
    DATAPATH = args.data_root + args.dataset
    setup_seed(args.seed)
    print('train begin')
    train()




