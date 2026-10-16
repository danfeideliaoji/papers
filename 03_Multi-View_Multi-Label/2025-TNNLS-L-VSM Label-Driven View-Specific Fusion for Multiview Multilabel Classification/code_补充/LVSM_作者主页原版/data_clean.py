import torch
import scipy.io as scio
from sklearn.metrics.pairwise import pairwise_distances
import numpy as np
# from mymodel import GAT, GAT_by_torch, Net
# from RGCN import RGCN
import torch.nn.functional as F
# from util import *
import scipy.io as sio
import pandas as pd
# from torch_geometric.data import Data
import time
import math
import torch.nn as nn
import os
import matplotlib.pyplot as plt
import sys
import h5py
from sklearn.metrics import hamming_loss, coverage_error, label_ranking_loss
from scipy.io import arff
import argparse


def data_loader(path):
    print('loadmat begin')
    # f = h5py.File(path)
    # with h5py.File(path, 'r') as f:
    # print(f.keys())
    # print(f['data'])

    if args.dataset == 'Iaprtc12.mat' or args.dataset == 'Espgame.mat':
        mat = h5py.File(path, 'r')
        view = []
        for i in range(6):
            los = mat['data'][0][i]
            dat = np.array(mat[los])
            view.append(torch.from_numpy(dat).transpose(1, 0))
            # print('dat:', dat.shape)
        # print('tar:', mat['target'])
        # lo = mat['target'][0][0]
        da = np.array(mat['target'])
        target = torch.from_numpy(da)
    elif args.dataset == 'scene.arff' or args.dataset == 'tmc2007.arff':
        data = arff.loadarff(path)
        df = pd.DataFrame(data)
        print(df[0].shape)
        print('df1:', df[1].shape)
        x = input()
    else:
        mat = sio.loadmat(path)
        print('loadmat end')
        view = []
        target = mat['target']
        print('data:', mat['data'].shape)
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
    print('len_view:', len(view))
    for i in range(len(view)):
        print('i:{} shape:{}'.format(i, view[i].shape))
    print('target:', target.shape)
    # x = input()
    return view, target


def data_clean(view, target):
    eps = 1e-5
    for i in range(len(view)):
        valid_col = []
        print('less:', (view[i] < 0).sum())
        #ict = torch.zeros(view[i].shape[1], target.shape[1])
        #for j in range(view[i].shape[0]):
            #for k in range(view[i].shape[1]):
                #dict[k][target[]]
        for j in range(view[i].shape[1]):
            if not (view[i][:, j] == 0).sum() == view[i][:, j].shape[0]:
                valid_col.append(j)
            # print('shape:{} {} {}'.format(view[i].shape, view[i][:,j].shape, view[i][0].shape))
            # print(view[i][:,j])
            #view[i][:,j] = (view[i][:,j] - view[i][:,j].mean() + eps) / (torch.std((view[i][:,j])) + eps)
            #view[i][:,j] -= view[i][:,j].min()
            #view[i][:,j] = view[i][:,j] / (view[i][:,j].max() + eps) + eps
            # print('aft:')
            # print(view[i][:,j])
            if i == 70:
                print('j:{} view:{} shape:{}'.format(j, view[i][:, j], view[i][:, j].shape))
        #view[i] = view[i][:, valid_col]
        print('i:{} viewi:{}'.format(i, view[i].shape))
    # return view, target
    # print(view[5][3195])
    # x = input()
    for i in range(len(view)):
        cnt = 0
        cnt_dim = 0
        print(view[i][0,:30])
        print('i:{} pos:{} num:{} per:{}'.format(i, (view[i] == 0).sum(), view[i].shape[0] * view[i].shape[1],
                                                 (view[i] == 0).sum() / (view[i].shape[0] * view[i].shape[1])))
        for j in range(view[i].shape[0]):
            if (view[i][j] == 0).sum() == view[i][j].shape[0]:
                cnt += 1
        for j in range(view[i].shape[1]):
            if (view[i][:, j] == 0).sum() == view[i][:, j].shape[0]:
                cnt_dim += 1
                # print('i:{} j:{} sum:{} len:{}'.format(i, j, (view[i][j] == 0).sum(), view[i][j].shape[0]))
                # print(view[i][j])
        print('i:{} cnt:{} cnt_dim:{}'.format(i, cnt, cnt_dim))
        # for j in range(view[i].shape[1]):
        # cnt = 0
        # for k in range(view[i].shape[0]):
        # if view[i][k][j] == 0:
        # cnt += 1
        # print('i:{} j:{} sum:{} len:{}'.format(i, j, (view[i][:,j] == 0).sum(), view[i][:,j].shape[0]))
        # view[i][:,j] = (view[i][:,j] - view[i][:,j].mean()) / torch.std(torch.from_numpy(view[i][:,j]))
    return view, target

if __name__ == '__main__':
    # path = 'data/Pascal.mat'
    # torch.set_num_threads(3)
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('--dataset', default='Pascal.mat', type=str, metavar='N', help='run_data')
    parser.add_argument('--data_root', default='data/', type=str, metavar='PATH',
                        help='root dir')
    parser.add_argument('--word2vec', default='data/voc_glove_word2vec.pkl', type=str, metavar='PATH',
                        help='root dir')
    parser.add_argument('--batch_size', default=1, type=int, help='number of batch size')
    parser.add_argument('--k', default=3, type=int, help='KNN')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--end_epochs', default=300, type=int, metavar='H-P',
                        help='number of total epochs to run')
    parser.add_argument('--lr', default=0.001, type=float,
                        metavar='H-P', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--alpha', default=0.9, type=float,
                        metavar='H-P', help='initial learning rate')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('--nhidden', default=256, type=int,
                        metavar='H-P', help='n_hidden')
    parser.add_argument('--k_fold', default=5, type=int,
                        metavar='H-P', help='n_hidden')
    parser.add_argument('--dropout', default=0.2, type=float,
                        metavar='H-P', help='n_hidden')
    parser.add_argument('--layers', default=2, type=int,
                        metavar='H-P', help='n_hidden')
    os.environ["CUDA_VISIBLE_DEVICES"] = ''
    args = parser.parse_args()
    view, target = data_loader(args.data_root + args.dataset)
    data_clean(view, target)
