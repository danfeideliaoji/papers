import torch
import scipy.io as scio
from sklearn.metrics.pairwise import pairwise_distances
import numpy as np
from mymodel import GAT, GAT_by_torch, Net
from RGCN import RGCN
import torch.nn.functional as F
import dgl
import argparse
from dgl.nn.pytorch import RelGraphConv
from util import *
import pandas as pd
from torch_geometric.data import Data
import scipy.io as sio
import torch.nn as nn
def data_loader(path):
    print('loadmat begin')
    mat = sio.loadmat(path)
    print('loadmat end')
    view = []
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

    return view, mat['target']
    # return torch.tensor(view), torch.tensor(mat['target'])
def get_global_edge_new(nins, nview):
    # print('nins:{} nview:{}'.format(nins, nview))
    u = []
    v = []
    for i in range(nview):
        for j in range(i + 1, nview, 1):
            for k in range(nins):
                u.append(i * nins + k)
                v.append(j * nins + k)

                v.append(i * nins + k)
                u.append(j * nins + k)
    return u, v
def construct_graph(x, dim_view):
    ntest, nview = x[0].shape[0], len(dim_view)
    test_g_list = []
    test_feat_list = []
    for i in range(len(x)):
        test_feat = x[i]
        test_feat_list.append(test_feat.float())
        test_ins_num, test_fea_num = test_feat.size()
        # test data
        test_data_dist = pairwise_distances(test_feat, metric='euclidean')  # (5011, 5011)
        test_topk_dist, test_topk_index = torch.from_numpy(test_data_dist).topk(dim=1, k=args.k, largest=False,
                                                                                sorted=True)  # (5011,5)
        # print('topk_index_shape', test_topk_index.shape)
        test_topk_index = test_topk_index[:, 1:]
        test_knn_idx = test_topk_index.reshape(-1, 1).type(torch.long)  # 5011x5=25055, 1
        test_ins_idx = torch.from_numpy(np.array(list(range(test_ins_num))).repeat(args.k - 1).reshape(-1, 1)).type(
            torch.long)  # (25055,1)
        test_edge_idx_ins = torch.cat((test_ins_idx, test_knn_idx), dim=1).transpose(1, 0)
        # print('test_ins_num:{} test_fea_num:{}'.format(test_ins_num, test_fea_num))
        # print('test_data_dist:', test_data_dist)
        # print('test_topk_dist:', test_topk_dist)
        # print('test_knn_idx:', test_knn_idx)
        # print('test_edge_idx_inx', test_edge_idx_ins)
        # test
        test_edges = test_edge_idx_ins[0].int(), test_edge_idx_ins[1].int()
        # print('test_edges:', test_edges)
        # x = input()
        test_g = dgl.graph(test_edges)
        # test_g.ndata['label'] = torch.tensor(Y_test.clone().detach()).float().argmax(1).reshape(-1, 1)  #
        # test_g.ndata['label'] = y.float().argmax(1).reshape(-1, 1).clone().detach()
        test_g.add_edges(test_edge_idx_ins[1].int(), test_edge_idx_ins[0].int())

        # test_edges2 = test_edge_idx_ins[1].int(), test_edge_idx_ins[0].int()

        test_g_list.append(test_g)
    # test
    test_G = dgl.batch(test_g_list)
    test_u, test_v = get_global_edge_new(ntest, nview)
    test_u, test_v = torch.tensor(test_u, dtype=torch.int32), torch.tensor(test_v, dtype=torch.int32)
    test_G.add_edges(test_u, test_v)
    # test_G = dgl.graph((test_u, test_v))
    test_e_type = torch.zeros(test_G.number_of_edges())
    test_e_type[test_G.edge_ids(test_u, test_v).long()] = 1
    return test_feat_list, test_G, test_e_type
def add_test_pre(x_train, y_train, x_test, y_test, model, crit, dim_view):
    model.eval()
    feats, g, e_type = construct_graph(x_test, dim_view)
    with torch.no_grad():
        logits = model(feats, g, e_type)  # (1010, 16)
        test_label = logits
    # print('test_label:{}  y_test:{}'.format(test_label.shape, y_test.shape))
    cnt = 0
    num = 0
    mif1 = 0
    hamnum = 0
    laebl_ham = torch.zeros(test_label.shape[0], test_label.shape[1])
    #return 0, 100, 0, 10, 10, 10, 0, 10, 0, 0
    for i in range(y_test.shape[0]):
        for j in range(y_test.shape[1]):
            if test_label[i][j] > 0:
                laebl_ham[i][j] = 1
            else:
                laebl_ham[i][j] = 0
            mif1 += laebl_ham[i][j] * y_test[i][j] * 2
            if laebl_ham[i][j] != y_test[i][j]:
                hamnum += 1
            if y_test[i][j] == 1:
                num += 1
                if test_label[i][j] >= 0:
                    cnt += 1
    mif1 /= (laebl_ham.sum() + y_test.sum())
    maf1 = macro_f1(laebl_ham, y_test)
    sa = subset_accuracy(laebl_ham, y_test)
    # print('mif1:', mif1)
    # print('acc_cnt:', cnt / num)
    #ap.add(test_label, y_test)
    # print('ap:', ap.value(), ap.value().mean())
    #ham = hamming_loss(y_test, laebl_ham)
    # print('ham:', ham)
    cov = coverage(test_label, y_test)
    # print('cov:', cov)
    rank = ranking_loss(test_label, y_test)
    # print('rank:', rank)
    one = one_error(test_label, y_test)
    # print('one:', one)
    ap2 = avgp(test_label, y_test)
    #loss_test = crit(test_label, y_test)
    loss_test = 5
    return ap2, loss_test, cnt / num, hamnum, rank, one, mif1, cov, sa, maf1
def tt(model, feat_list, G, e_type):
    return model(feat_list, G, e_type)
def train(train_view, Y_train, test_view, Y_test):
    model = RGCN(dim_view, 128, Y_train.shape[1], 2, .2, 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_f = nn.MultiLabelSoftMarginLoss()
    best_acc = 0
    best_epoch = 0
    for epoch in range(args.end_epochs):
        model.train()
        feat_list, G, e_type = construct_graph(train_view, dim_view)
        pred = tt(model, feat_list, G, e_type)  # (5011, 20)
        #loss1 = ((1 - torch.tensor(Y_train)) * pred).sum()
        # loss2 = (-pred*torch.log(pred)).sum()
        loss = loss_f(pred, Y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print('train end')
        if epoch % 10 == 0:
            acc_test2, loss_test2, acc_cnt2, ham2, rank2, one2, mif12, cov2, sa2, maf12 = add_test_pre(train_view,
                                                                                                       Y_train,
                                                                                                       test_view,
                                                                                                       Y_test, model,
                                                                                                       loss_f, dim_view)

    print('best epoch:' + str(best_epoch) + '    ' + 'best acc:' + str(best_acc))
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('--dataset', default='Pascal07_5view.mat', type=str, metavar='N', help='run_data')
    parser.add_argument('--data_root', default='data/', type=str, metavar='PATH',
                        help='root dir')
    parser.add_argument('--word2vec', default='data/voc_glove_word2vec.pkl', type=str, metavar='PATH',
                        help='root dir')
    parser.add_argument('--batch_size', default=1, type=int, help='number of batch size')
    parser.add_argument('--k', default=3, type=int, help='KNN')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--end_epochs', default=500, type=int, metavar='H-P',
                        help='number of total epochs to run')
    parser.add_argument('--lr', default=0.0001, type=float,
                        metavar='H-P', help='initial learning rate')
    parser.add_argument('--alpha', default=0.9, type=float,
                        metavar='H-P', help='initial learning rate')
    parser.add_argument('--nhidden', default=256, type=int,
                        metavar='H-P', help='n_hidden')

    args = parser.parse_args()
    view, target = data_loader(args.data_root + args.dataset)
    target = torch.tensor(target).permute(1, 0)
    dim_view = []  # 统计视图维度
    for i in range(len(view)):
        dim_view.append(view[i].shape[1])
        view[i] = torch.tensor(view[i]).float()
    train_view = view
    test_view = view
    Y_train = target
    Y_test = target
    train(train_view, Y_train, test_view, Y_test)
