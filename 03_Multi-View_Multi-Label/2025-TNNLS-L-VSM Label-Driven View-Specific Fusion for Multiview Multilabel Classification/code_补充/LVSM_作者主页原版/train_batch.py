import torch
import random
import scipy.io as scio
from joblib import Parallel, delayed
from sklearn.metrics.pairwise import pairwise_distances, cosine_similarity
import numpy as np
from mymodel import GAT, GAT_by_torch, Net
from RGCN import RGCN
import torch.nn.functional as F
import dgl
import argparse
from dgl.nn.pytorch import RelGraphConv
from util import *
import scipy.io as sio
import pandas as pd
# from torch_geometric.data import Data
import time
import math
import torch.nn as nn
import os
import copy
import matplotlib.pyplot as plt
import sys
import h5py
from sklearn.metrics import hamming_loss, coverage_error, label_ranking_loss
from scipy.io import arff

from skmultilearn.dataset import load_dataset
from skmultilearn.dataset import available_data_sets
import torch
import math
import copy
from sklearn.metrics import make_scorer
from sklearn import metrics
import pandas as  pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

max_edge = 20


def accuracy_score(y_test, y_pred):
    # y_pred is a numpy array, y_test is a dataframe
    # to compare the two, convert to a single type
    y_test = y_test.to_numpy()

    # shape of test and preds must be equal
    assert y_test.shape == y_pred.shape
    i = 0
    # list of scores for each training sample
    scores = []

    # for each test sample
    while i < len(y_test):
        count = 0
        # count the number of matches in the sample
        # y_test[i] -> row values in test set (true values)
        # y_pred[i] -> row values in predictions set (predicted values)
        for p, q in zip(y_test[i], y_pred[i]):
            if p == q:
                count += 1

        # accuracy score for the sample = no. of correctly predicted labels/total no. of labels
        scores.append(count / y_pred.shape[1])
        i += 1

        # final accuracy = avg. accuracy over all test samples =
    # sum of the accuracy of all training samples/no. of training samples
    return round((sum(scores) / len(y_test)), 5)


class BinaryRelevanceClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, base_classifier=LogisticRegression()):
        self.base_classifier = base_classifier

    def fit(self, X, y):
        """Build a Binary Relevance classifier from the training set (X, y).
        Parameters
        ----------
        X : array-like or sparse matrix, shape = [n_samples, n_features]
            The training input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csc_matrix``.
        y : array-like, shape = [n_samples, n_labels]
            The target values (class labels) as integers or strings.
        """

        # list of individual classifiers
        self.models = []

        # for every class label
        for label in list(y.columns):
            # Check that X and y have correct shape
            x_checked, y_checked = check_X_y(X, y[label])
            # every classifier is independent of the others
            # hence we create a copy of the base classifier instance
            base_model = clone(self.base_classifier)
            # fit the base model - one model each for Y1, Y2....Y14
            basel_model = base_model.fit(x_checked, y_checked)
            # add the fitted model list of individual classifiers
            self.models.append(base_model)

    # The predict function to make a set of predictions for a set of query instances
    def predict(self, X):
        # check if the models list has been set up
        check_is_fitted(self, ['models'])
        X = check_array(X)

        all_preds = pd.DataFrame()
        i = 0
        # list of individual classifier predictions
        preds = []

        # predict against each fitted model - one model per label
        for model in self.models:
            pred = model.predict(X)
            # add the prediction to the dataframe
            preds.append(pd.DataFrame({'Class' + str(i + 1): pred}))
            i += 1

        # dataframe with predictions for all class labels
        all_preds = pd.concat(preds, axis=1)
        # standard sklearn classifiers return predictions as numpy arrays
        # hence convert the dataframe to a numpy array
        return all_preds.to_numpy()

    def predict_proba(self, X):
        # check if the models list has been set up
        check_is_fitted(self, ['models'])
        X = check_array(X)

        all_preds = pd.DataFrame()
        i = 0

        for model in self.models:
            # Call predict_proba of the each base model
            pred = model.predict_proba(X)
            # Add the probabilities of 1 to the dataframe
            all_preds['Class' + str(i + 1)] = [one_prob[1] for one_prob in pred]
            i += 1

        # return probabilities
        return all_preds.to_numpy()


def load_arff(dataset):
    X, y, feature_names, label_names = load_dataset(dataset, 'train')
    Xt, yt, feature_namest, label_namest = load_dataset(dataset, 'test')
    xm = torch.cat([torch.from_numpy(X.toarray()), torch.from_numpy(Xt.toarray())], dim=0)
    ym = torch.cat([torch.from_numpy(y.toarray()), torch.from_numpy(yt.toarray())], dim=0)
    return xm, ym


def divide(x, y, per):
    lenx, leny = x.shape[1], y.shape[1]
    mx, my = math.floor(lenx * per), math.floor(leny * per)
    indicesx = torch.randperm(lenx)
    indicesy = torch.randperm(leny)
    xn = x[:, indicesx[:mx]].detach().clone()
    yn = y[:, indicesy[:my]].detach().clone()
    return xn, yn


def warm_learning(epoch, optimizer):
    lr_list = []
    decay = (epoch + 1) / args.warmup
    decay_pre = (epoch) / args.warmup
    # print('decay:', decay)
    # print('decay_pre:', decay_pre)

    for param_group in optimizer.param_groups:
        # print('lr:', param_group['lr'])
        param_group['lr'] = param_group['lr'] * decay
        if decay_pre > 0:
            param_group['lr'] = param_group['lr'] / decay_pre
        lr_list.append(param_group['lr'])
    # print('lr_list:', lr_list)
    return np.unique(lr_list)


def cos_learning(epoch, optimizer):
    lr_list = []
    # decay = 0.5 * (1 + torch.cos(torch.tensor(self.state['epoch'] * 3.14159265359 / self.state['max_epochs'])))
    # decay_pre = 0.5 * (1 + torch.cos(torch.tensor((self.state['epoch'] - 1) * 3.14159265359 / self.state['max_epochs'])))
    # print('epoch:{} max:{}'.format(self.state['epoch'], self.state['max_epochs']))
    decay = 0.5 * (1 + math.cos(epoch * 3.14159265359 / args.end_epochs))
    decay_pre = 0.5 * (
            1 + math.cos((epoch - 1) * 3.14159265359 / args.end_epochs))
    # print('decay:', decay)
    # print('decay_pre:', decay_pre)
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * decay
        if decay_pre > 0:
            param_group['lr'] = param_group['lr'] / decay_pre
        lr_list.append(param_group['lr'])
    return np.unique(lr_list)


def adjust_learning_rate(epoch, optimizer):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    if epoch < args.warmup:
        return warm_learning(epoch, optimizer)
    return cos_learning(epoch, optimizer)
    lr_list = []
    # decay = 0.5 if sum(self.state['epoch'] == np.array(self.state['epoch_step'])) > 0 else 1.0
    # print('self:{} step:{} {}'.format(self.state['epoch'], self.state['epoch_step'], np.array(self.state['epoch_step'])))
    decay = self.state['lr_decay'] if self.state['epoch'] % self.state['epoch_step'][0] == 0 and self.state[
        'epoch'] != 0 else 1.0
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * decay
        lr_list.append(param_group['lr'])
    return np.unique(lr_list)


def evaluate_new(training, model, feats, g, e_type, labels, crit, thres=0):
    if training:
        model.train()
        logits = model(feats, g, e_type)  # (1010, 16)
        ap2 = avgp(logits - thres, labels)
        return ap2, crit(logits, labels)
    model.eval()
    with torch.no_grad():
        logits = model(feats, g, e_type)  # (1010, 16)
        for i in range(0):
            print('logits:', logits[i])
            print('labels:', labels[i])

        cnt = 0
        num = 0
        for i in range(labels.shape[0]):
            for j in range(labels.shape[1]):
                if labels[i][j] == 1:
                    num += 1
                    if logits[i][j] >= thres:
                        cnt += 1
        print('acc_cnt:', cnt / num)

        ap.add(logits - thres, labels)
        # print(ap.value())
        ap2 = avgp(logits - thres, labels)
        return ap2, crit(logits, labels)
        # return cnt * 1.0 / (labels.shape[0] * labels.shape[1]), crit(logits, labels)


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


def data_loader(path):
    print('loadmat begin')
    # f = h5py.File(path)
    # with h5py.File(path, 'r') as f:
    # print(f.keys())
    # print(f['data'])

    if args.dataset == 'Iaprtc12.mat' or args.dataset == 'Espgame.mat' or args.dataset == 'Mirflickr.mat':
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
    elif args.dataset == 'scene':
        xm, ym = load_arff(args.dataset)
        view = []
        view.append(xm[:, 0:98])
        view.append(xm[:, 98:])
        target = ym.permute(1, 0)
        # data = arff.loadarff(path)
        # df = pd.DataFrame(data)
        # print(df)
        # x = input()
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
        for i in range(len(view)):
            view[i] = torch.from_numpy(view[i])
    print('len_view:', len(view))
    for i in range(len(view)):
        print('i:{} shape:{}'.format(i, view[i].shape))
    print('target:', target.shape)
    # x = input()
    return view, target
    # return torch.tensor(view), torch.tensor(mat['target'])


def construct_graph_k(x, dim_view, training, train_id):
    return construct_graph(x, dim_view, training, train_id)
    ntest, nview = x[0].shape[0], len(dim_view)
    # print('ntest:{} nview:{}'.format(ntest, nview))
    test_u, test_v = get_global_edge_new(ntest, nview)
    test_u, test_v = torch.tensor(test_u, dtype=torch.int32), torch.tensor(test_v, dtype=torch.int32)
    test_edges = test_u, test_v
    test_G = dgl.graph(test_edges)

    test_g_list = []
    test_feat_list = []
    test_e_type = torch.zeros(ntest * (args.k - 1) * nview * 2 + test_G.number_of_edges())
    knn_edge_id_u = []
    knn_edge_id_v = []
    # print('edge_num:', test_G.number_of_edges())
    for i in range(args.k - 1):
        knn_edge_id_v.append([])
        knn_edge_id_u.append([])
    for i in range(len(x)):
        test_feat = x[i]
        test_feat_list.append(test_feat.float())
        test_ins_num, test_fea_num = test_feat.size()
        test_data_dist = pairwise_distances(test_feat, metric='euclidean')  # (5011, 5011)
        test_topk_dist, test_topk_index = torch.from_numpy(test_data_dist).topk(dim=1, k=args.k, largest=False,
                                                                                sorted=True)  # (5011,5)
        # print('top_pre:', test_topk_index)
        test_topk_index += i * ntest
        # print('top_aft:', test_topk_index)
        test_topk_index = test_topk_index[:, 1:]
        test_knn_idx = test_topk_index.reshape(-1, 1).type(torch.long)  # 5011x5=25055, 1
        test_ins_idx = torch.from_numpy(np.array(list(range(test_ins_num))).repeat(args.k - 1).reshape(-1, 1)).type(
            torch.long)  # (25055,1)
        test_edge_idx_ins = torch.cat((test_ins_idx, test_knn_idx), dim=1).transpose(1, 0)
        test_edges = test_edge_idx_ins[0].int(), test_edge_idx_ins[1].int()
        # print('test_dges:', test_edge_idx_ins[0].int(), test_edge_idx_ins[1].int())
        test_g = dgl.graph(test_edges)
        test_G.add_edges(test_edge_idx_ins[1].int(), test_edge_idx_ins[0].int())
        test_g_list.append(test_g)
        test_G.add_edges(test_edge_idx_ins[0].int(), test_edge_idx_ins[1].int())
        for j in range(args.k - 1):
            knn_edge_id_v[j].append(test_topk_index[:, j].reshape(-1, 1).type(torch.long).squeeze())
            knn_edge_id_u[j].append(
                torch.from_numpy(np.array(list(range(ntest)))).reshape(-1, 1).type(torch.long).squeeze())

            knn_edge_id_u[j].append(test_topk_index[:, j].reshape(-1, 1).type(torch.long).squeeze())
            knn_edge_id_v[j].append(
                torch.from_numpy(np.array(list(range(ntest)))).reshape(-1, 1).type(torch.long).squeeze())
            # print('j:{} u:{} v:{}'.format(j, knn_edge_id_u[j][-1].shape, knn_edge_id_v[j][-1].shape))

    # test_G = dgl.batch(test_g_list)

    # print('test_u:', test_u)
    # print('test_v:', test_v)
    # print('edge_num:', test_G.number_of_edges())
    # print('u:', knn_edge_id_u[0].shape)
    # print('v:', knn_edge_id_v[0].shape)
    for i in range(args.k - 1):
        knnv = knn_edge_id_v[i][0]
        knnu = knn_edge_id_u[i][0]
        for j in range(len(knn_edge_id_v[i])):
            if j == 0:
                continue
            knnv = torch.cat((knnv, knn_edge_id_v[i][j]), dim=0)
            knnu = torch.cat((knnu, knn_edge_id_u[i][j]), dim=0)
        knnu = knnu.int()
        knnv = knnv.int()
        # knnv = torch.cat(knn_edge_id_v, dim=0).int()
        # knnu = torch.cat(knn_edge_id_u, dim=0).int()
        # knn_edge_id_u[i] = torch.tensor(knn_edge_id_u[i])
        # knn_edge_id_v[i] = torch.tensor(knn_edge_id_v[i])
        # knnu, knnv = torch.tensor(knnu.clone().detach(), dtype=torch.int32), torch.tensor(knnv.clone().detach(), dtype=torch.int32)
        # for j in range(knnu.shape[0]):
        # print('knnu:{} knnv:{}'.format(knnu[j], knnv[j]))
        # print('knnu:{} knnv:{}'.format(knnu.shape, knnv.shape))

        mid = 23
        # print('knnu:{} knnv:{}'.format(knnu[mid], knnv[mid]))
        # print('edge:', test_G.edge_ids(knnu[0:mid], knnv[0:mid]))
        test_e_type[test_G.edge_ids(knnu, knnv).long()] = i
    # for i in range(args.k - 1):
    # test_e_type[test_G.edge_ids(knn_edge_id_u[i], knn_edge_id_v[i]).long()] = i
    test_e_type[test_G.edge_ids(test_u, test_v).long()] = args.k - 1
    return test_feat_list, test_G, test_e_type


def construct_graph_adj(x, dim_view):
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
        # test_topk_index = test_topk_index[:, 1:]
        test_knn_idx = test_topk_index.reshape(-1, 1).type(torch.long)  # 5011x5=25055, 1
        test_ins_idx = torch.from_numpy(np.array(list(range(test_ins_num))).repeat(args.k - 0).reshape(-1, 1)).type(
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
        test_g.add_edges(test_edge_idx_ins[1].int(), test_edge_idx_ins[0].int())
        # test_g.ndata['label'] = torch.tensor(Y_test.clone().detach()).float().argmax(1).reshape(-1, 1)  #
        # test_g.ndata['label'] = y.float().argmax(1).reshape(-1, 1).clone().detach()
        # test_g.add_edges(test_edge_idx_ins[1].int(), test_edge_idx_ins[0].int())

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
    return test_feat_list, test_G, test_e_type, test_g_list


def euc(x, y):
    return ((x - y) * (x - y)).sum() / x.shape[0]


def construct_graph(x, dim_view, training, train_id):
    ntest, nview = x[0].shape[0], len(dim_view)
    test_g_list = []
    test_feat_list = []
    for i in range(len(x)):
        test_feat = x[i]
        test_feat_list.append(test_feat.float())
        '''
        test_ins_num, test_fea_num = test_feat.size()
        # test data
        test_data_dist = pairwise_distances(test_feat, metric='euclidean')  # (5011, 5011)
        test_topk_dist, test_topk_index = torch.from_numpy(test_data_dist).topk(dim=1, k=args.k, largest=False,
                                                                                sorted=True)  # (5011,5)
        # print('topk_index_shape', test_topk_index.shape)
        #test_topk_index = test_topk_index[:, 1:]
        test_knn_idx = test_topk_index.reshape(-1, 1).type(torch.long)  # 5011x5=25055, 1
        test_ins_idx = torch.from_numpy(np.array(list(range(test_ins_num))).repeat(args.k - 0).reshape(-1, 1)).type(
            torch.long)  # (25055,1)
        test_edge_idx_ins = torch.cat((test_ins_idx, test_knn_idx), dim=1).transpose(1, 0)
        # print('test_ins_num:{} test_fea_num:{}'.format(test_ins_num, test_fea_num))
        # print('test_data_dist:', test_data_dist)
        # print('test_topk_dist:', test_topk_dist)
        # print('test_knn_idx:', test_knn_idx)
        # print('test_edge_idx_inx', test_edge_idx_ins)
        # test
        test_edges = test_edge_idx_ins[0].int(), test_edge_idx_ins[1].int()
        xp = test_edges[0][0:5].type(
            torch.long)
        yp = test_edges[1][0:5].type(
            torch.long)
                for i in range(xp.shape[0]):
            #print('test_edges:', xp[i], yp[i])
            print('label_xp:', y1[xp[i]], xp[i])
            print('label_yp:', y1[yp[i]], yp[i])
            #print('shape:', test_feat[xp[i]].shape)
            print('dis:', euc(test_feat[xp[i]], test_feat[yp[i]]))
            print('xp:', test_feat[xp[i]].shape)
            print('yp:', test_feat[yp[i]].shape)
        '''
        # print('i:{} eva begin:'.format(i))
        t0 = time.time()
        a, b = eva_graph(x, 0, i, training, train_id)
        # print('eva:', time.time() - t0)
        # a = torch.from_numpy(a)
        # b = torch.from_numpy(b)
        a = a.int()
        b = b.int()
        test_edges = a, b
        test_g = dgl.graph(test_edges)
        #print('a:{} b:{} ntest:{}'.format(a.shape, b.shape, ntest))
        # test_g.add_edges(b, a)
        # test_g.add_edges(test_edge_idx_ins[1].int(), test_edge_idx_ins[0].int())
        # test_g.ndata['label'] = torch.tensor(Y_test.clone().detach()).float().argmax(1).reshape(-1, 1)  #
        # test_g.ndata['label'] = y.float().argmax(1).reshape(-1, 1).clone().detach()
        # test_g.add_edges(test_edge_idx_ins[1].int(), test_edge_idx_ins[0].int())

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
    # print('num:{} edge:{}'.format(ntest, test_G.number_of_edges()))
    return test_feat_list, test_G, test_e_type, test_g_list


def add_test(x_train, y_train, x_test, y_test, model, crit, dim_view):
    # acc_test, loss_test = 0, 0
    test_label = torch.zeros(y_test.shape[0], y_test.shape[1])
    ap = AveragePrecisionMeter()
    model.eval()
    for i in range(y_test.shape[0]):
        x_fus = []  # 混合测试集和训练集，用于测试
        for j in range(len(x_test)):
            # print('xtj:', x_train[j].shape)

            x_fusj = torch.cat((x_train[j], x_test[j][i].view(1, -1)), dim=0)
            x_fus.append(x_fusj)
        # y_fus = torch.cat((y_train, y_test[i].view(1,-1)), dim=0)
        feats, g, e_type = construct_graph(x_fus, dim_view)
        with torch.no_grad():
            logits = model(feats, g, e_type)  # (1010, 16)
            test_label[i] = logits[-1]
    cnt = 0
    num = 0
    laebl_ham = torch.zeros(test_label.shape[0], test_label.shape[1])
    for i in range(y_test.shape[0]):
        for j in range(y_test.shape[1]):
            if test_label[i][j] > 0:
                laebl_ham[i][j] = 1
            else:
                laebl_ham[i][j] = 0
            if y_test[i][j] == 1:
                num += 1
                if test_label[i][j] >= 0:
                    cnt += 1

    # print('acc_cnt:', cnt / num)
    ap.add(test_label, y_test)
    # print('ap:', ap.value(), ap.value().mean())
    ham = hamming_loss(y_test, laebl_ham)
    # print('ham:', ham)
    # cov = coverage_error(y_test, test_label)
    # print('cov:', cov)
    rank = label_ranking_loss(y_test, test_label)
    # print('rank:', rank)
    one = one_error(test_label, y_test)
    # print('one:', one)
    return ap.value().mean(), crit(test_label, y_test)


def add_test_whole(x_train, y_train, x_test, y_test, model, crit, dim_view):
    return 0, 10000, 0, 100000, 100000, 100000, 0, 1000000, 0, 0
    # acc_test, loss_test = 0, 0
    # test_label = torch.zeros(y_test.shape[0], y_test.shape[1])
    # ap = AveragePrecisionMeter()
    x_fus = []  # 混合测试集和训练集，用于测试
    for j in range(len(x_test)):
        # print('xtj:', x_train[j].shape)
        x_fusj = torch.cat((x_train[j], x_test[j]), dim=0)
        x_fus.append(x_fusj)
    # y_fus = torch.cat((y_train, y_test), dim=0)
    feats, g, e_type = construct_graph(x_fus, dim_view)
    model.eval()

    with torch.no_grad():
        logits = model(feats, g, e_type)  # (1010, 16)
        test_label = logits[x_train[0].shape[0]:]
    # print('test_label:{}  y_test:{}'.format(test_label.shape, y_test.shape))
    label_ham = torch.zeros(test_label.shape[0], test_label.shape[1])
    # print('logits:', logits > 0, logits.shape)
    label_ham[test_label > 0] = 1
    # print('label_ham:', label_ham.shape)
    # print('y_test:', y_test.shape)
    mif1 = (label_ham * y_test * 2).sum() / (label_ham.sum() + y_test.sum())
    cnt = 0
    sa = subset_accuracy(label_ham, y_test)
    maf1 = macro_f1(label_ham, y_test)
    # print('mif1:', mif1)
    hamnum = (label_ham != y_test).sum() / (y_test.shape[0] * y_test.shape[1])
    # print('acc_cnt:', cnt / num)
    # ap.add(test_label, y_test)
    # print('ap:', ap.value(), ap.value().mean())
    # ham = hamming_loss(y_test, laebl_ham)
    # print('ham:', ham)
    # print('hamnum:', hamnum)
    cov = coverage(test_label, y_test)
    # print('cov:', cov)
    rank = ranking_loss(test_label, y_test)
    # print('rank:', rank)
    one = one_error(test_label, y_test)
    # print('one:', one)
    ap2 = avgp(test_label, y_test)
    # loss_test = crit(test_label, y_test)
    loss_test = 5
    return ap2, loss_test, cnt, hamnum, rank, one, mif1, cov, sa, maf1


def add_test_pre(x_train, y_train, x_test, y_test, model, crit, dim_view, test_id):
    # acc_test, loss_test = 0, 0
    test_label = torch.zeros(y_test.shape[0], y_test.shape[1])
    # ap = AveragePrecisionMeter()
    x_fus = []  # 混合测试集和训练集，用于测试
    model.eval()
    for j in range(len(x_test)):
        # print('xtj:', x_train[j].shape)
        # x_fusj = torch.cat((x_train[j], x_test[j]), dim=0)
        x_fus.append(x_test[j])

    feats, g, e_type, g_list = construct_graph_k(x_fus, dim_view, False, test_id)

    with torch.no_grad():
        logits = model(feats, g, e_type, g_list)  # (1010, 16)
        test_label = logits
    # print('test_label:{}  y_test:{}'.format(test_label.shape, y_test.shape))
    num = 0

    label_ham = torch.zeros(test_label.shape[0], test_label.shape[1])
    # print('logits:', logits > 0, logits.shape)
    label_ham[test_label > 0] = 1
    # print('label_ham:', label_ham.shape)
    # print('y_test:', y_test.shape)
    mif1 = (label_ham * y_test * 2).sum() / (label_ham.sum() + y_test.sum())
    cnt = 0
    sa = subset_accuracy(label_ham, y_test)
    maf1 = macro_f1(label_ham, y_test)
    hamnum = (label_ham != y_test).sum() / (y_test.shape[0] * y_test.shape[1])
    # print('mif1:', mif1)
    # print('acc_cnt:', cnt / num)
    # ap.add(test_label, y_test)
    # print('ap:', ap.value(), ap.value().mean())
    # ham = hamming_loss(y_test, laebl_ham)
    # print('ham:', ham)
    cov = coverage(test_label, y_test)
    # print('cov:', cov)
    rank = ranking_loss(test_label, y_test)
    # print('rank:', rank)
    one = one_error(test_label, y_test)
    # print('one:', one)
    ap2 = avgp(test_label, y_test)
    # loss_test = crit(test_label, y_test)
    loss_test = 5
    return ap2, loss_test, cnt, hamnum, rank, one, mif1, cov, sa, maf1


def compute_pro(test_v, model, dim_view, test_id):
    model.eval()
    feats, g, e_type, g_list = construct_graph_k(test_v, dim_view, False, test_id)
    with torch.no_grad():
        logits = model(feats, g, e_type, g_list)  # (1010, 16)
    return logits


def process_test(x_test, model, dim_view, test_id):
    test_v = []
    l = i * size
    r = min((i + 1) * size, x_test[0].shape[0])
    for j in range(len(x_test)):
        test_v.append(x_test[j])
    test_label = compute_pro(test_v, model, dim_view, test_id)
    return test_label


def test_epoch(x_train, y_train, x_test, y_test, model, crit, dim_view, test_id):
    size = x_test[0].shape[0] // (args.batch_size // 4)
    acc_train, loss = 0, 0
    num = x_test[0].shape[0] // size
    test_label = torch.zeros(y_test.shape[0], y_test.shape[1])

    for i in range(num):
        # print('test_i:', i)
        test_v = []
        l = i * size
        r = min((i + 1) * size, x_test[0].shape[0])
        if i == num - 1:
            r = x_test[0].shape[0]
        for j in range(len(x_test)):
            test_v.append(x_test[j][l:r])
        test_label[l:r] = compute_pro(test_v, model, dim_view, test_id[l:r])

    # print('size:{} num:{}'.format(size, num))

    # test_label = Parallel(n_jobs=4)(delayed(process_test)(x_test, model, dim_view, test_id, size, i) for i in range(num + 1))
    # test_label = torch.cat(test_label, dim=0)

    # for i in range(len(test_label)):
    # print('i:{} test:{}'.format(i, test_label[i].shape))

    # print('test_label:', test_label.shape)
    label_ham = torch.zeros(test_label.shape[0], test_label.shape[1])
    label_ham[test_label > 0] = 1
    mif1 = (label_ham * y_test * 2).sum() / (label_ham.sum() + y_test.sum())
    cnt = 0
    sa = subset_accuracy(label_ham, y_test)
    maf1 = macro_f1(label_ham, y_test)
    hamnum = (label_ham != y_test).sum() / (y_test.shape[0] * y_test.shape[1])
    cov = coverage(test_label, y_test)
    rank = ranking_loss(test_label, y_test)
    one = one_error(test_label, y_test)
    ap2 = avgp(test_label, y_test)
    loss_test = 5
    return ap2, loss_test, cnt, hamnum, rank, one, mif1, cov, sa, maf1


def train_batch(train_view, Y_train, dim_view, model, optimizer, loss_f, test_view, Y_test, train_id, test_id):
    model.train()
    # t0 = time.time()
    feat_list, G, e_type, g_list = construct_graph_k(train_view, dim_view, True, train_id)
    # t1 = time.time()
    # print('graph:', t1 - t0)
    '''
        if torch.rand(1)[0] > 10:
        inputs, targets_a, targets_b, lam = mixup_data(feat_list, Y_train, 0.5)
        logits = model(inputs, G, e_type, g_list)
        loss = mixup_criterion(loss_f, logits, targets_a, targets_b, lam)
    else:
    '''
    logits = model(feat_list, G, e_type, g_list)  # (1010, 16)
    # t2 = time.time()
    # print('model:', t2 - t1)

    # sig = nn.Sigmoid()
    # print('logits: min:{} mean:{} max:{} std:{} sample:{}'.format(logits.min(), logits.mean(), logits.max(), logits.std(), logits[0]))
    # print('logits:{} pro:{} label:{}'.format(logits[0], sig(logits[0]), Y_train[0]))
    # tar = Y_train.copy()
    # Y_train = Y_train.float()
    # Y_train[Y_train==1] = 0.99
    # print('Y', Y_train[0], Y_train[0] == 0)
    loss = loss_f(logits, Y_train)
    acc_train = avgp(logits, Y_train)
    optimizer.zero_grad()
    loss.backward()
    # t3 = time.time()
    # print('back:', t3 - t2)
    optimizer.step()
    # t4 = time.time()
    # print('step:', t4 - t3)
    with torch.no_grad():
        acc_test2, loss_test2, acc_cnt2, ham2, rank2, one2, mif12, cov2, sa2, maf12 = test_epoch(train_view,
                                                                                             Y_train,
                                                                                             test_view,
                                                                                             Y_test, model,
                                                                                             loss_f, dim_view,
                                                                                             test_id)
    t5 = time.time()
    # print('test:', t5 - t4)
    return acc_train, loss, acc_test2, loss_test2, acc_cnt2, ham2, rank2, one2, mif12, cov2, sa2, maf12


def train_epoch(train_view, Y_train, dim_view, model, optimizer, loss_f, test_view, Y_test, train_id, test_id):
    size = train_view[0].shape[0] // args.batch_size
    acc_train, loss = 0, 0
    num = train_view[0].shape[0] // size
    acc_test_b = 0
    acc_cnt_b = 0
    ham_b = 100000
    rank_b = 100000
    one_b = 100000
    mif1_b = 0
    maf1_b = 0
    cov_b = 100000
    sa_b = 0
    loss_test_b = 100000
    for i in range(num):
        train_v = []
        l = i * size
        r = min((i + 1) * size, train_view[0].shape[0])
        # print('l:{} r:{}'.format(l,r))
        for j in range(len(train_view)):
            train_v.append(train_view[j][l:r])
        acc_train_, loss_, acc_test2, loss_test2, acc_cnt2, ham2, rank2, one2, mif12, cov2, sa2, maf12 = train_batch(
            train_v, Y_train[l:r], dim_view, model, optimizer, loss_f, test_view, Y_test, train_id[l:r], test_id)
        acc_test_b = max(acc_test_b, acc_test2)
        acc_cnt_b = max(acc_cnt_b, acc_cnt2)
        ham_b = min(ham_b, ham2)
        rank_b = min(rank_b, rank2)
        one_b = min(one_b, one2)
        mif1_b = max(mif1_b, mif12)
        maf1_b = max(maf12, maf1_b)
        cov_b = min(cov_b, cov2)
        sa_b = max(sa_b, sa2)
        loss_test_b = min(loss_test_b, loss_test2)
        '''
        print('i:', i)
        print('test_acc2:{} test_loss2:{}:'.format(acc_test2, loss_test2))
        print(
            'cnt2:{} ham2:{} rank2:{} one2:{} mif12:{} cov2:{} sa2:{} maf12:{}'.format(acc_cnt2, ham2, rank2, one2,
                                                                                       mif12, cov2, sa2, maf12))
        print('cnt_b:{} ham_b:{} rank_b:{} one_b:{} mif1_b:{} cov_b:{} sa_b:{} maf1_b:{}'.format(acc_cnt_b, ham_b,
                                                                                                 rank_b, one_b,
                                                                                                 mif1_b, cov_b,
                                                                                                 sa_b, maf1_b))
        '''
        acc_train += acc_train_
        loss += loss_
    acc_train /= num
    loss /= num
    return acc_train, loss, acc_test_b, loss_test_b, acc_cnt_b, ham_b, rank_b, one_b, mif1_b, cov_b, sa_b, maf1_b


def get_dict(view, t):
    ma = []
    cosid = []
    for i in range(len(view)):
        # dim_view.append(view[i].shape[1])

        dic = {}
        # print('i:{} shape:{}'.format(i, view[i].shape))
        for j in range(view[i].shape[0]):
            dic[hash(view[i][j].numpy().tobytes())] = j
        # for j in range(view[i].shape[0]):
        # print('i:{} j:{} hash:{} dic:{}'.format(i, j, hash(view[i][j].numpy().tobytes()), dic[hash(view[i][j].numpy().tobytes())]))
        # print('view:', view[i][j].shape)
        # view[i] = torch.tensor(view[i])
        ma.append(dic)
        # print('dic:', len(dic))
        cosid.append(cosine_similarity(view[i]) >= t[i])
    return ma, cosid


def batch(train_view, Y_train, test_view, Y_test, dim_view, model, optimizer, k_, train_id, test_id):
    '''
    # print('train_view:', len(train_view))
    # for i in range(len(train_view)):
    # print('i:{} shape:{}'.format(i, train_view[i].shape))
    # print('Y_train_shape:', Y_test.shape)
    # print('test_view:', len(test_view))
    # for i in range(len(test_view)):
    # print('i:{} shape:{}'.format(i, test_view[i].shape))
    # print('Y_train_shape:', Y_train.shape)
    # print('dim_view:', dim_view)
    g_list = []
    test_g_list = []
    feat_list = []
    test_feat_list = []
    for i in range(len(train_view)):
        print('----------------------------------------view:', i)
        view_feats = train_view[i]  # (5011, 100)
        test_feat = test_view[i]
        feat_list.append(view_feats.float())
        test_feat_list.append(test_feat.float())

        ins_num, fea_num = view_feats.size()  # 5011 100
        test_ins_num, test_fea_num = test_feat.size()

        # train data
        data_dist = pairwise_distances(view_feats, metric='euclidean')  # (5011, 5011)
        topk_dist, topk_index = torch.from_numpy(data_dist).topk(dim=1, k=args.k, largest=False,
                                                                 sorted=True)  # (5011,5)
        # print('topk_dist:{} {}'.format(topk_dist[0], topk_dist.shape))
        # print('topk_index:{} {}'.format(topk_index[0], topk_index.shape))
        knn_idx = topk_index.reshape(-1, 1).type(torch.long)  # 5011x5=25055, 1
        ins_idx = torch.from_numpy(np.array(list(range(ins_num))).repeat(args.k).reshape(-1, 1)).type(
            torch.long)  # (25055,1) n的全排列顺序出现k次
        # print('ins_idx:', ins_idx[0:20])
        edge_idx_ins = torch.cat((ins_idx, knn_idx), dim=1).transpose(1, 0)  # (2,25055)

        # test data
        test_data_dist = pairwise_distances(test_feat, metric='euclidean')  # (5011, 5011)
        test_topk_dist, test_topk_index = torch.from_numpy(test_data_dist).topk(dim=1, k=args.k, largest=False,
                                                                                sorted=True)  # (5011,5)
        test_knn_idx = test_topk_index.reshape(-1, 1).type(torch.long)  # 5011x5=25055, 1
        test_ins_idx = torch.from_numpy(np.array(list(range(test_ins_num))).repeat(args.k).reshape(-1, 1)).type(
            torch.long)  # (25055,1)
        test_edge_idx_ins = torch.cat((test_ins_idx, test_knn_idx), dim=1).transpose(1, 0)

        # Construct Network
        edges = edge_idx_ins[0].int(), edge_idx_ins[1].int()
        # print('edges:', edges)
        g = dgl.graph(edges)
        # print('g:',g)
        # print('Y_train:', Y_train.shape)
        # g.ndata['label'] = torch.tensor(Y_train.clone().detach()).float().argmax(1).reshape(-1,1)  # [[0], [0], [0], [0], [1], [1], [1], [1], [1], [1], [1]
        g.ndata['label'] = Y_train.float().argmax(1).reshape(-1, 1).clone().detach()  # [[0], [0], [0], [0], [1], [1], [1], [1], [1], [1], [1]
        g_list.append(g)

        # test
        test_edges = test_edge_idx_ins[0].int(), test_edge_idx_ins[1].int()
        test_g = dgl.graph(test_edges)
        # test_g.ndata['label'] = torch.tensor(Y_test.clone().detach()).float().argmax(1).reshape(-1, 1)  #
        test_g.ndata['label'] = Y_test.float().argmax(1).reshape(-1, 1).clone().detach()
        test_g_list.append(test_g)

    G = dgl.batch(g_list)
    #print('G1:',G)
    #print('ntrain:{} nview:{}'.format(ntrain, nview))
    u, v = get_global_edge_new(ntrain, nview)

    u, v = torch.tensor(u, dtype=torch.int32), torch.tensor(v, dtype=torch.int32)
    #print('u:{} {}'.format(u.shape, u.max()))
    #print('v:{} {}'.format(v.shape, v.max()))
    G.add_edges(u, v)
    #print('G2:',G)
    e_type = torch.zeros(G.number_of_edges())
    e_type[G.edge_ids(u, v).long()] = 1

    # test
    test_G = dgl.batch(test_g_list)
    test_u, test_v = get_global_edge_new(ntest, nview)
    test_u, test_v = torch.tensor(test_u, dtype=torch.int32), torch.tensor(test_v, dtype=torch.int32)
    test_G.add_edges(test_u, test_v)
    test_e_type = torch.zeros(test_G.number_of_edges())
    test_e_type[test_G.edge_ids(test_u, test_v).long()] = 1
    '''
    '''
    X_train = train_view
    X_test = test_view
    br_clf = BinaryRelevanceClassifier(LogisticRegression())
    # fit
    x_t = torch.cat(X_train, dim=1)
    print('x_t:', x_t.shape)
    br_clf.fit(pd.DataFrame(x_t.numpy()), pd.DataFrame(Y_train.numpy()))
    # predict
    Y_pred = br_clf.predict(pd.DataFrame(torch.cat(X_test, dim=1).numpy()))
    print(Y_pred[0:5])
    print(Y_test[0:5])
    Y_pred = torch.from_numpy(Y_pred)
    #Y_test = torch.from_numpy(Y_test)
    print('ap:', avgp(Y_pred, Y_test))
    print('sa:', subset_accuracy(Y_pred, Y_test))
    #print('correct:{} num:{}'.format((Y_pred == Y_test).sum(), Y_pred.shape[0] * Y_test.shape[1]))
    print("y_pred.shape: " + str(Y_pred.shape))
    print("Accuracy of Binary Relevance Classifier: " + str(accuracy_score(pd.DataFrame(Y_test), pd.DataFrame(Y_pred))))
    xpp = input()
    '''
    '''
    train_id = torch.zeros(train_view[0].shape[0])
    test_id = torch.zeros(test_view[0].shape[0])
    for i in range(train_id.shape[0]):
        train_id[i] = i
    for i in range(test_id.shape[0]):
        test_id[i] = i
    '''
    # train_view1, Y_train1, train_id1 = shuffle_id(train_view, Y_train, train_id)
    # test_view, Y_test, test_id = shuffle_id(test_view, Y_test, test_id)
    # print('trainid1:', train_id1)
    # print('eee:', (Y_train1 == Y_train[train_id1.type(torch.long)]).sum())
    # print('eee2:', (Y_train1[train_id1.type(torch.long)] == Y_train).sum(), Y_train.shape[0] * Y_train.shape[1])
    # print('eee4:', (train_view1[0][train_id1.type(torch.long)] == train_view[0]).sum(), train_view[0].shape[0] * train_view[0].shape[1])

    # test_feat_list, test_G, test_e_type = construct_graph(test_view, Y_test, dim_view)
    train_view, Y_train, train_id = shuffle_id(train_view, Y_train, train_id)
    test_view, Y_test, test_id = shuffle_id(test_view, Y_test, test_id)
    # maf.clear()
    # cosidf.clear()
    # ma, cosid = get_dict(train_view, thres)

    # maf.append(ma)
    # cosidf.append(cosid)
    # print('cosidf:', cosidf[0].shape)
    # ma, cosid = get_dict(test_view, thres)
    # maf.append(ma)
    # cosidf.append(cosid)
    # ma, cosid = get_dict(train_view, thres)
    # cosid = torch.tensor(cosid)
    # print('cos:', cosid[0])
    # print('mmm:', cosid[0] == cosidf[0][0][train_id1.type(torch.long)])
    # print('mmm2:', cosidf[0][0][train_id1.type(torch.long)] == cosid[0])
    # train_id = train_id.type(torch.long)
    # test_id = test_id.type(torch.long)

    for i in range(len(dim_view)):
        eva_graph(train_view, Y_train, i, True, train_id)
        eva_graph(test_view, Y_test, i, False, test_id)
    # xo = input()

    for j in range(1):
        train_view, Y_train, train_id = shuffle_id(train_view, Y_train, train_id)
        test_view, Y_test, test_id = shuffle_id(test_view, Y_test, test_id)
        for i in range(len(dim_view)):
            eva_graph(train_view, Y_train, i, True, train_id)
            eva_graph(test_view, Y_test, i, False, test_id)
    # xo = input()
    best_acc = 0
    best_epoch = 0
    # print('model2:', model)
    # print('feat_list_shape:', len(feat_list))
    # for i in range(len(feat_list)):
    # print('i:{} shape:{}'.format(i, feat_list[i].shape))
    # feat_list[i] = feat_list[i].float()
    # print('G3:', G)
    # print('e_type_shape', e_type.shape)
    # print('Y_train0:', Y_train[0])
    # aaa = input()
    acc_train_b = 0
    loss_train_b = 0
    acc_test_b = 0
    loss_test_b = 0
    train_a = []
    train_l = []
    test_a = []
    test_l = []

    acc_cnt_b = 0
    ham_b = 100000
    rank_b = 100000
    one_b = 100000
    mif1_b = 0
    maf1_b = 0
    cov_b = 100000
    sa_b = 0
    test_cnt = []
    test_ham = []
    test_rank = []
    test_one = []
    epoch_cnt = 0
    epoch_ham = 0
    epoch_rank = 0
    epoch_one = 0
    epoch_mif1 = 0
    epoch_maf1 = 0
    epoch_cov = 0
    epoch_sa = 0
    t_start = time.time()
    adj = cal_adj(Y_train)
    '''
    prior = np.load('prior/scene.npz')
    print('prior:', len(prior))
    print(prior)
    print(prior['arr_0'].shape, prior['arr_0'].min(), prior['arr_0'].mean(), prior['arr_0'].max())
    print(prior['arr_1'].shape, prior['arr_1'].min(), prior['arr_1'].mean(), prior['arr_1'].max())
    print(train_view[0].shape, train_view[0].min(), train_view[0].mean(), train_view[0].max())
    print(train_view[1].shape, train_view[1].min(), train_view[1].mean(), train_view[1].max())
    compare(prior['arr_0'], train_view[0], Y_train)
    '''
    embeddings = []
    for i in range(len(train_view)):
        embeddings.append(compute_embedding(train_view[i], Y_train))
    model.get_adj(adj, args.at)
    model.get_embeddings(embeddings)
    model.get_gc()
    # for i in range(len(prior)):
    # print(prior[i].shape, prior[i].min(), prior[i].mean(), prior[i].max())
    # asdf = input()
    for epoch in range(args.end_epochs):
        # lr = adjust_learning_rate(epoch, optimizer)
        # print('lr:', lr)
        t0 = time.time()
        train_view, Y_train, train_id = shuffle_id(train_view, Y_train, train_id)
        test_view, Y_test, test_id = shuffle_id(test_view, Y_test, test_id)

        # eva_graph(train_view, Y_train, 0, True, train_id)
        # eva_graph(test_view, Y_test, 0, False, test_id)
        # continue
        acc_test, loss_test, acc_cnt, ham, rank, one, mif1, cov, sa, maf1 = 0, 10000, 0, 100000, 100000, 100000, 0, 1000000, 0, 0
        t1 = time.time()
        # print('shuffle:', t1 - t0)
        # feat_list, G, e_type = construct_graph(train_view, dim_view)
        # model.train()
        # pred = model(feat_list, G, e_type)  # (5011, 20)
        # print('pred:', pred.shape)
        # print('pred0:', pred[0])
        # print('Y_train0:', Y_train[0])
        loss_f = nn.MultiLabelSoftMarginLoss()
        # loss_f = AsymmetricLossOptimized(gamma_neg=4, gamma_pos=0, clip=0.05, disable_torch_grad_focal_loss=True)
        # loss_f = nn.BCEWithLogitsLoss()
        # print('Y_train:', Y_train.shape)
        # loss1 = ((1 - Y_train) * pred).sum()
        # loss2 = (-pred*torch.log(pred)).sum()
        # t0 = time.time()
        # print('t_graph:', t0 - t_1)
        # acc_train, loss_train = evaluate_new(True, model, feat_list, G, e_type, Y_train, loss_f)

        # t1 = time.time()
        # print('t_train:', t1 - t0)
        acc_train, loss_train, acc_test2, loss_test2, acc_cnt2, ham2, rank2, one2, mif12, cov2, sa2, maf12 = train_epoch(
            train_view, Y_train, dim_view, model, optimizer, loss_f, test_view, Y_test, train_id, test_id)

        acc_train_b = max(acc_train, acc_train_b)
        loss_train_b = max(loss_train, loss_train_b)
        if maf1 > maf1_b or maf12 > maf1_b:
            maf1_b = max(maf1_b, max(maf12, maf1))
            epoch_maf1 = epoch
        if sa > sa_b or sa2 > sa_b:
            sa_b = max(sa_b, max(sa, sa2))
            epoch_sa = epoch
        if cov < cov_b or cov2 < cov_b:
            epoch_cov = epoch
        if mif1 > mif1_b or mif12 > mif1_b:
            epoch_mif1 = epoch
        if acc_cnt > acc_cnt_b or acc_cnt2 > acc_cnt_b:
            epoch_cnt = epoch
        if ham < ham_b or ham2 < ham_b:
            epoch_ham = epoch
        if rank < rank_b or rank2 < rank_b:
            epoch_rank = epoch
        if one < one_b or one2 < one_b:
            epoch_one = epoch
        cov_b = min(cov, cov_b)
        cov_b = min(cov2, cov_b)
        acc_test_b = max(acc_test, acc_test_b)
        loss_test_b = min(loss_test, loss_test_b)
        acc_test_b = max(acc_test2, acc_test_b)
        loss_test_b = min(loss_test2, loss_test_b)
        acc_cnt_b = max(acc_cnt, acc_cnt_b)
        ham_b = min(ham_b, ham)
        rank_b = min(rank_b, rank)
        one_b = min(one_b, one)
        mif1_b = max(mif1, mif1_b)
        acc_cnt_b = max(acc_cnt2, acc_cnt_b)
        ham_b = min(ham_b, ham2)
        rank_b = min(rank_b, rank2)
        one_b = min(one_b, one2)
        mif1_b = max(mif12, mif1_b)
        '''
        print('fir')
        print('test_acc2:{} test_loss2:{}:'.format(acc_test2, loss_test2))
        print(
            'cnt2:{} ham2:{} rank2:{} one2:{} mif12:{} cov2:{} sa2:{} maf12:{}'.format(acc_cnt2, ham2, rank2, one2,
                                                                                       mif12, cov2, sa2, maf12))
        print('cnt_b:{} ham_b:{} rank_b:{} one_b:{} mif1_b:{} cov_b:{} sa_b:{} maf1_b:{}'.format(acc_cnt_b, ham_b,
                                                                                                 rank_b, one_b,
                                                                                                 mif1_b, cov_b,
                                                                                                 sa_b, maf1_b))
        print('best_epoch: cnt:{} ham:{} rank:{} one:{} mif1:{} cov:{} sa:{} maf1:{}'.format(epoch_cnt, epoch_ham,
                                                                                             epoch_rank,
                                                                                             epoch_one, epoch_mif1,
                                                                                             epoch_cov, epoch_sa,
                                                                                             epoch_maf1))
        '''
        # t2 = time.time()
        # print('t_back:', t2 - t1)
        # optimizer.step()
        # t3 = time.time()
        # print('t_step:', t3 - t2)

        # print('acc_train:', acc_train)
        # print('test:',Y_test, Y_test.shape)
        # print('test_max:', Y_test.argmax(1), Y_test.argmax(1).shape)
        # acc, pred_index = evaluate(False, model, test_feat_list, test_G, test_e_type, Y_test)
        if epoch < 0:
            acc_test, loss_test, acc_cnt, ham, rank, one, mif1, cov, sa, maf1 = 0, 10000, 0, 100000, 100000, 100000, 0, 1000000, 0, 0
            acc_test2, loss_test2, acc_cnt2, ham2, rank2, one2, mif12, cov2, sa2, maf12 = 0, 10000, 0, 100000, 100000, 100000, 0, 1000000, 0, 0
            acc_test3, loss_test3 = 0, 0
        else:
            acc_test, loss_test, acc_cnt, ham, rank, one, mif1, cov, sa, maf1 = add_test_whole(train_view, Y_train,
                                                                                               test_view,
                                                                                               Y_test, model,
                                                                                               loss_f, dim_view)

            # acc_test2, loss_test2, acc_cnt2, ham2, rank2, one2, mif12, cov2, sa2, maf12 = add_test_pre(train_view,
            # Y_train,test_view,Y_test, model, loss_f, dim_view) acc_test3, loss_test3= add_test(train_view, Y_train,
            # test_view, Y_test, model, loss_f, dim_view)
        # t4 = time.time()
        # print('t_test:', t4 - t3)
        if maf1 > maf1_b or maf12 > maf1_b:
            maf1_b = max(maf1_b, max(maf12, maf1))
            epoch_maf1 = epoch
        if sa > sa_b or sa2 > sa_b:
            sa_b = max(sa_b, max(sa, sa2))
            epoch_sa = epoch
        if cov < cov_b or cov2 < cov_b:
            epoch_cov = epoch
        if mif1 > mif1_b or mif12 > mif1_b:
            epoch_mif1 = epoch
        if acc_cnt > acc_cnt_b or acc_cnt2 > acc_cnt_b:
            epoch_cnt = epoch
        if ham < ham_b or ham2 < ham_b:
            epoch_ham = epoch
        if rank < rank_b or rank2 < rank_b:
            epoch_rank = epoch
        if one < one_b or one2 < one_b:
            epoch_one = epoch
        cov_b = min(cov, cov_b)
        cov_b = min(cov2, cov_b)
        acc_test_b = max(acc_test, acc_test_b)
        loss_test_b = min(loss_test, loss_test_b)
        acc_test_b = max(acc_test2, acc_test_b)
        loss_test_b = min(loss_test2, loss_test_b)
        acc_cnt_b = max(acc_cnt, acc_cnt_b)
        ham_b = min(ham_b, ham)
        rank_b = min(rank_b, rank)
        one_b = min(one_b, one)
        mif1_b = max(mif1, mif1_b)
        acc_cnt_b = max(acc_cnt2, acc_cnt_b)
        ham_b = min(ham_b, ham2)
        rank_b = min(rank_b, rank2)
        one_b = min(one_b, one2)
        mif1_b = max(mif12, mif1_b)
        # acc, pred_index = evaluate(model, test_feat_list, test_G, test_e_type, Y_test.argmax(1))
        # print('acc:', acc)
        acc = acc_test
        acc = max(acc, acc_test2)
        # acc = max(acc, acc_test3)
        if best_acc < acc:
            best_acc = acc
            best_epoch = epoch
            # beat_pred_index = pred_index
        # print('epoch: ' + str(epoch) + '  loss:' + str(loss.item()) + '    acc:' + str(acc))
        train_a.append(acc_train)
        train_l.append(loss_train)
        test_a.append(acc_test)
        test_l.append(loss_test)
        test_cnt.append(acc_cnt_b)
        test_ham.append(ham_b)
        test_rank.append(rank_b)
        test_one.append(one_b)
        if epoch % 1 == 0:
            print('epoch:{}/{}'.format(epoch, args.end_epochs))
            print('train_acc:{} train_loss:{}'.format(acc_train, loss_train))
            print('test_acc:{} test_loss:{}:'.format(acc_test, loss_test))
            print('test_acc2:{} test_loss2:{}:'.format(acc_test2, loss_test2))
            # print('test_acc3:{} test_loss3:{}:'.format(acc_test3, loss_test3))
            print('best:{} epoch:{}'.format(best_acc, best_epoch))
            print('cnt:{} ham:{} rank:{} one:{} mif1:{} cov:{} sa:{} maf1:{}'.format(acc_cnt, ham, rank, one, mif1, cov,
                                                                                     sa, maf1))
            print(
                'cnt2:{} ham2:{} rank2:{} one2:{} mif12:{} cov2:{} sa2:{} maf12:{}'.format(acc_cnt2, ham2, rank2, one2,
                                                                                           mif12, cov2, sa2, maf12))
            print('cnt_b:{} ham_b:{} rank_b:{} one_b:{} mif1_b:{} cov_b:{} sa_b:{} maf1_b:{}'.format(acc_cnt_b, ham_b,
                                                                                                     rank_b, one_b,
                                                                                                     mif1_b, cov_b,
                                                                                                     sa_b, maf1_b))
            print('best_epoch: cnt:{} ham:{} rank:{} one:{} mif1:{} cov:{} sa:{} maf1:{}'.format(epoch_cnt, epoch_ham,
                                                                                                 epoch_rank,
                                                                                                 epoch_one, epoch_mif1,
                                                                                                 epoch_cov, epoch_sa,
                                                                                                 epoch_maf1))
            ts = time.time() - t_start
            print('time:{} remain:{}'.format(ts, (args.end_epochs - epoch - 1) * ts / (epoch + 1)))
            sys.stdout.flush()
    plt.clf()
    plt.ylim(0, 1)
    ax1 = plt.plot(k_ * 2 + 0)
    draw(train_a, test_a, pic + '/acc' + str(k_) + '.png')
    plt.clf()
    ax2 = plt.plot(k_ * 2 + 1)
    draw(train_l, test_l, pic + '/loss' + str(k_) + '.png')
    '''
    ax1 = plt.plot(k_ * 6 + 1)
    draw(train_a, test_a, pic + '/acc' + str(k_) + '.png')
    ax2 = plt.plot(k_ * 6 + 2)
    draw(train_l, test_l, pic + '/loss' + str(k_) + '.png')
    ax3 = plt.plot(k_ * 6 + 3)
    draw(train_l, test_cnt, pic + '/loss' + str(k_) + '.png')
    ax4 = plt.plot(k_ * 6 + 4)
    draw(train_l, test_ham, pic + '/loss' + str(k_) + '.png')
    ax5 = plt.plot(k_ * 6 + 5)
    draw(train_l, test_rank, pic + '/loss' + str(k_) + '.png')
    ax6 = plt.plot(k_ * 6 + 0)
    draw(train_l, test_one, pic + '/loss' + str(k_) + '.png')
    '''

    # print('best epoch:' + str(best_epoch) + '    ' + 'best acc:' + str(best_acc))
    return acc_train_b, loss_train_b, acc_test_b, loss_test_b, acc_cnt_b, ham_b, rank_b, one_b, mif1_b, cov_b, sa_b, maf1_b


def get_k_fold_data(k, i, X, y):
    # 返回第i折交叉验证时所需要的训练和测试数据，分开放，X_train为训练数据，X_test为验证数据
    assert k > 1
    fold_size = X.shape[0] // k  # 每份的个数:数据总条数/折数（向下取整）

    X_train, y_train = None, None
    for j in range(k):
        idx = slice(j * fold_size, (j + 1) * fold_size)  # slice(start,end,step)切片函数 得到测试集的索引
        # print('fold_size:{} idx:{}'.format(fold_size, idx))
        X_part, y_part = X[idx, :], y[idx]  # 只对第一维切片即可
        if j == i:  # 第i折作test
            X_test, y_test = X_part, y_part
        elif X_train is None:
            X_train, y_train = X_part, y_part
        else:
            X_train = torch.cat((X_train, X_part), dim=0)  # 其他剩余折进行拼接 也仅第一维
            y_train = torch.cat((y_train, y_part), dim=0)
    return X_train, y_train, X_test, y_test


def ok(t, data, idc, target):
    acc = args.t
    mul = max_edge
    c, d = (data >= t).nonzero()
    # print('c:', c)
    a, b = eva_graph((c, d), target, -1, True, idc)
    print('t:{} a:{} b:{}'.format(t, a, b))
    return a > acc and b < mul


def find_t(data, idc, target):
    l = 0.8
    r = 1
    eps = 1e-3
    while (r - l > eps):
        mid = (r + l) / 2
        if ok(mid, data, idc, target):
            r = mid
        else:
            l = mid
    return mid


def k_fold(k, data, target):
    idc = torch.zeros(target.shape[0])
    for i in range(target.shape[0]):
        idc[i] = i

    vi = []
    # data, target, idc = shuffle_id(data, target, idc)
    r = min(1000000, data[0].shape[0])
    indces = torch.randperm(data[0].shape[0])
    for i in range(indces.shape[0]):
        indces[i] = i
    for i in range(len(data)):
        vi.append(data[i][indces[0:r]])
        # vi.append(data[i])
    global cosidf

    tp = 'thres'
    if not os.path.exists(tp):
        os.makedirs(tp)
    tp2 = tp + '/' + args.dataset + '_t' + str(args.t)
    tt = []
    if not os.path.exists(tp2):
        cosidf = make_cos(vi)
        for i in range(len(data)):
            th = find_t(cosidf[i], idc[indces[0:r]], target[indces[0:r]])
            # th = find_t(cosidf[i], idc, target)
            tt.append(th)
        _write_conds(tt, tp2)
    tt = _read_conds(tp2)
    print('thres:', tt)
    cos = 'cos'
    if not os.path.exists(cos):
        os.makedirs(cos)
    cosp = cos + '/' + str(args.dataset) + '_t' + str(args.t)
    if not os.path.exists(cosp):
        cosidf = make_cos(data)
        for i in range(len(data)):
            cosidf[i] = (cosidf[i] >= tt[i]).nonzero()
        _write_conds(cosidf, cosp)

    cosidf = _read_conds(cosp)

    # print('sdaf')
    data, target, idc = shuffle_id(data, target, idc)
    train_loss_sum, test_loss_sum = 0, 0
    train_acc_sum, test_acc_sum = 0, 0
    acc_cnt_sum, ham_sum, rank_sum, one_sum = 0, 0, 0, 0
    cov_sum = 0
    mif1_sum = 0
    sa_sum = 0
    # train_view, Y_train, test_view, Y_test, dim_view
    train_a = []
    train_l = []
    test_a = []
    test_l = []
    acc_cnt_l = []
    ham_l, rank_l, one_l, mif1_l, cov_l = [], [], [], [], []
    sa_l = []
    maf1_l = []
    for i in range(k):
        # print('i:', i)
        train_view = []
        test_view = []
        dim_view = []
        y_train = None
        y_test = None

        for j in range(len(data)):
            dim_view.append(data[j].shape[1])
        model = RGCN(dim_view, args.nhidden, target.shape[1], args.layers, args.dropout, 2)
        # print('model:', model)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        dim_view = []  # 统计视图维度
        for j in range(len(data)):
            print('i:{} k:{} j:{} len:{}'.format(i, k, j, len(data)))
            train_v, y_train, test_v, y_test = get_k_fold_data(k, i, data[j], target)  # 获取第i折交叉验证的训练和验证数据
            print('X_train:{} y_train:{} X_test:{} y_test:{}'.format(train_v.shape, y_train.shape, test_v.shape,
                                                                     y_test.shape))
            train_view.append(train_v)
            test_view.append(test_v)
            dim_view.append(test_v.shape[1])
        print('idc:', idc.shape)
        train_id, y_train, test_id, y_test = get_k_fold_data(k, i, idc.reshape(-1, 1), target)
        train_id = train_id.squeeze()
        test_id = test_id.squeeze()
        train_view, y_train, train_id = shuffle_id(train_view, y_train, train_id)
        test_view, y_test, test_id = shuffle_id(test_view, y_test, test_id)
        # shuffle(data, target)
        # ttt(train_view, y_train, data[0].shape[0])
        # xpp = input()
        # return
        # print('train_view:{} y_train:{} test_view:{} y_test:{}'.format(len(train_view), y_train.shape, len(test_view),y_test.shape))
        acc_train, loss_train, acc_test, loss_test, acc_cnt_b, ham_b, rank_b, one_b, mif1_b, cov_b, sa_b, maf1_b = batch(
            train_view,
            y_train,
            test_view,
            y_test,
            dim_view,
            model,
            optimizer,
            i + 2, train_id, test_id)
        print('i:{} acc_train:{} loss_train:{} acc_test:{} loss_test:{}'.format(i, acc_train, loss_train, acc_test,
                                                                                loss_test))
        train_loss_sum += loss_train
        test_loss_sum += loss_test
        train_acc_sum += acc_train
        test_acc_sum += acc_test
        acc_cnt_sum += acc_cnt_b
        ham_sum += ham_b
        rank_sum += rank_b
        one_sum += one_b
        mif1_sum += mif1_b
        cov_sum += cov_b
        sa_sum += sa_b
        train_a.append(acc_train)
        train_l.append(loss_train)
        test_a.append(acc_test)
        test_l.append(loss_test)
        acc_cnt_l.append(acc_cnt_b)
        ham_l.append(ham_b)
        rank_l.append(rank_b)
        one_l.append(one_b)
        mif1_l.append(mif1_b)
        cov_l.append(cov_b)
        sa_l.append(sa_b)
        maf1_l.append(maf1_b)
    print_std(test_a, 'ap')
    print_std(ham_l, 'hamming loss')
    print_std(rank_l, 'ranking loss')
    print_std(one_l, 'one error')
    print_std(mif1_l, 'micro-F1')
    print_std(cov_l, 'coverage')
    print_std(sa_l, 'subset accuracy')
    print_std(maf1_l, 'macro_f1')
    sa_sum /= k
    train_acc_sum /= k
    test_acc_sum /= k
    train_loss_sum /= k
    test_loss_sum /= k
    acc_cnt_sum /= k
    ham_sum /= k
    rank_sum /= k
    one_sum /= k
    mif1_sum /= k
    cov_sum /= k
    print('train_acc:{} train_loss:{}'.format(train_acc_sum, train_loss_sum))
    print('test_acc:{} test_loss:{}:'.format(test_acc_sum, test_loss_sum))
    print('acc_cnt:{} ham:{} rank:{} one:{} mif1:{} cov:{}'.format(acc_cnt_sum, ham_sum, rank_sum, one_sum, mif1_sum,
                                                                   cov_sum))
    fig = plt.figure(1)  # 如果不传入参数默认画板1
    # 第2步创建画纸，并选择画纸1
    ax1 = plt.subplot(2, 1, 1)
    draw(train_a, test_a, pic + '/acc.png')
    ax2 = plt.subplot(2, 1, 2)
    draw(train_l, test_l, pic + '/loss.png')
    return train_acc_sum, train_loss_sum, test_acc_sum, test_loss_sum, acc_cnt_sum, ham_sum, rank_sum, one_sum


def get_shu(data, indce):
    res = torch.zeros(data.shape[0], data.shape[1])
    res = data[indce]
    # for i in range(data.shape[0]):
    # res[i] = copy.deepcopy(data[indce[i]])
    # print('res:', res[i])
    # print('data:', data[indce[i]])
    # print(res[i] == data[indce[i]])
    # sdfp = input()

    return res


def shuffle(x, y):
    # return x,y
    num_ins = x[0].shape[0]
    # print('num_ins:', num_ins)
    indces = torch.randperm(num_ins)
    a = []
    for i in range(len(x)):
        a.append(get_shu(x[i], indces))
    b = get_shu(y, indces)
    return a, b


def shuffle_id(x, y, ids):
    # return x,y, ids
    num_ins = x[0].shape[0]
    # print('num_ins:', num_ins)
    indces = torch.randperm(num_ins)
    # seed = indces[0] + 2
    # print(indces)
    a = []
    for i in range(len(x)):
        # print('i:{} len:{}'.format(i, len(x)))
        # random.seed(seed)
        # random.shuffle(x[i])
        a.append(get_shu(x[i], indces))
        # print('equa:', a[i] == x[i][indces])
    # random.seed(seed)
    # random.shuffle(y)
    # random.seed(seed)
    # random.shuffle(ids)
    # return x, y, ids
    # print('a')
    b = get_shu(y, indces)
    # print('b')
    c = get_shu(ids.reshape(-1, 1), indces).squeeze()
    # print('c')
    # d = torch.zeros(num_ins)
    # for i in range(num_ins):
    # print('i:{} c:{}'.format(i, c[i]))
    # d[c[i].type(torch.long)] = i
    # print('equb:', b == y[indces])
    # print('equc:', c == ids[indces])
    return a, b, c


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
    # colors1 = '#00CED1'  # 点的颜色
    # colors2 = '#DC143C'
    # area = np.pi * 4 ** 2  # 点面积
    # plt.scatter(x_list, train_l, s=area, c=colors1, alpha=0.4, label='类别A')
    # plt.scatter(x_list, test_l, s=area, c=colors2, alpha=0.4, label='类别B')
    plt.plot(x_list, train_l, color='r', linewidth=1, alpha=0.6, label='train')
    plt.plot(x_list, test_l, color='b', linewidth=1, alpha=0.6, label='test')
    plt.legend()
    plt.savefig(path, dpi=300)


def data_clean(view, target):
    eps = 1e-5
    for i in range(len(view)):
        valid_col = []
        print('less:', (view[i] < 0).sum())
        # ict = torch.zeros(view[i].shape[1], target.shape[1])
        # for j in range(view[i].shape[0]):
        # for k in range(view[i].shape[1]):
        # dict[k][target[]]
        for j in range(view[i].shape[1]):
            if not (view[i][:, j] == 0).sum() == view[i][:, j].shape[0]:
                valid_col.append(j)
            # print('shape:{} {} {}'.format(view[i].shape, view[i][:,j].shape, view[i][0].shape))
            # print(view[i][:,j])
            view[i][:, j] = (view[i][:, j] - view[i][:, j].mean()) / (torch.std((view[i][:, j])) + eps)
            # view[i][:, j] = (view[i][:, j] - view[i][:, j].min()) / (view[i][:, j].max() - view[i][:, j].min() + eps)
            # view[i][:, j] = view[i][:, j] / (view[i][:, j].max() + eps) + eps
            # print('aft:')
            # print(view[i][:,j])
            if i == 70:
                print('j:{} view:{} shape:{}'.format(j, view[i][:, j], view[i][:, j].shape))
        # for j in range(view[i].shape[0]):
        # view[i][j] = (view[i][j] - view[i][j].mean()) / (torch.std((view[i][j])) + eps)
        # view[i] = view[i][:, valid_col]

        print('i:{} viewi:{}'.format(i, view[i].shape))
    # return view, target
    # print(view[5][3195])
    # x = input()
    for i in range(len(view)):
        cnt = 0
        cnt_dim = 0
        print(view[i][0, :30])
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


def construct_knn(x):
    test_feat = x
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
    return test_edge_idx_ins[0], test_edge_idx_ins[1]


def construct_cos(x, t=0.85):
    a, b = (cosine_similarity(x, x) >= t).nonzero()
    return torch.from_numpy(a), torch.from_numpy(b)
    # cos = nn.CosineSimilarity(dim=0, eps=1e-6)
    cos2 = nn.CosineSimilarity(dim=1, eps=1e-6)
    a = []
    b = []
    dis = cosine_similarity(x, x)
    disy = (dis >= t).nonzero()

    print('dis:', dis.shape)
    print('disy:', disy)
    for i in range(x.shape[0]):
        cp = cos2(x[i].reshape(1, -1), x)
        # print(cp[0:5])
        # print(dis[i][0:5])
        cpy = (cp >= t).nonzero()
        a.append(torch.tensor(i).repeat(cpy.shape[0]))
        b.append(cpy)
        '''
        print(cpy)
        for j in range(i + 1, x.shape[0]):
            co = cos(x[i], x[j])
            #print(co, cp[j])
            if co >= t:
                print('i:{} j:{} co:{}'.format(i, j, co))
                a.append(i)
                b.append(j)
        '''

    return torch.cat(a, dim=0), torch.cat(b, dim=0).squeeze()


def eva_graph(X, y, id, training, trid=None):
    # id = 1
    # print('id:',id)
    trid = trid.type(torch.long)
    # for i in range(len(X)):
    # X[i] = X[i][trid]
    # ma, co = get_dict(X, thres)
    # print('co:', co[id].shape)
    # print('cosidf:', len(cosidf))
    # print('cosid:', cosid.shape)
    # print(co[id][0])
    # print(co[id][trid[0]])
    # print(cosid[id][0])
    # print(cosid[id][trid[0]])
    # print('sum1:', (co[id] == cosid[id]).sum(), co[id].shape[0] * co[id].shape[1])
    # print('sum2:', (co[id][trid] == cosid[id]).sum(), co[id].shape[0] * co[id].shape[1])
    # print('sum3:', (co[id] == cosid[id][trid]).sum(), co[id].shape[0] * co[id].shape[1])
    # print('sum4:', (co[id][trid] == cosid[id][trid]).sum(), co[id].shape[0] * co[id].shape[1])
    # x = X[id]
    # a, b = construct_cos(x, args.t)
    # print('sdfsdf:', (cosid[0][ma[0][hash(x[0].numpy().tobytes())]]).shape)
    # matri = np.zeros((x.shape[0], x.shape[0]))
    # print('total:', total_num)
    # print('x:', x.shape)
    # print('cosid:', cosid[id].shape)
    # print('ma:', len(ma[id]))
    # print('trid:', trid.shape)
    if id == -1:
        matri = X
    else:
        matri = cosidf[id]
    # print('matri:', matri.shape)
    # print(matri == cosid[id])
    # for i in range(x.shape[0]):
    # print('ma:', ma[id][hash(x[i].numpy().tobytes())])
    # matri[i] = cosid[id][ma[id][hash(x[i].numpy().tobytes())]]
    # print('cosid:', cosid[id][trid])
    # matri[i] = cosid[id][trid]
    # print('c:', cosid[id][i])
    # b.append((cosid[0][ma[0][hash(x[i].numpy().tobytes())]]).nonzero())
    # a.append(torch.tensor(i).repeat(b[-1].shape[0]))
    # print(matri.shape)
    # print(matri.nonzero())
    # print('matri:', matri.shape)
    # print('non:', matri.nonzero())
    a, b = matri
    if id == -1:
        print('b:', b.shape, trid.shape)
    if b.shape[0] > trid.shape[0] * max_edge and id == -1:
        return 0, b.shape[0] / trid.shape[0]
    # print('a:{} b:{}'.format(a.shape, b.shape))
    # print('total:', a.shape, b.shape, trid.shape)
    # print('trid:', trid.min())
    c, d = [], []
    dii = {}  # 从原编号域->现编号域
    for i in range(trid.shape[0]):
        dii[trid[i].item()] = i
    # for key in dii:
    # print('key:', key, dii[key])
    # c = [dii[i] for i in range(a[i].shape[0]) and dii.has_key(a[i])]
    # d = [dii[i] for i in b and dii.has_key(i)]
    for i in range(a.shape[0]):
        # if dii.has_key(a[i]) and dii.has_key(b[i]):
        if a[i] in dii and b[i] in dii:
            c.append(dii[a[i]])
            d.append(dii[b[i]])
    c = torch.tensor(c)
    d = torch.tensor(d)
    # for i in range(a.shape[0]):
    # print('i:', i)
    # c[i] = dii[a[i]]
    # d[i] = dii[b[i]]
    # return a, b
    #
    if id != -1:
        # print('c:{} trid:{}'.format(c.shape, trid.shape))
        if c.shape[0] < trid.shape[0]:
            c = torch.zeros(trid.shape[0])
            d = torch.zeros(trid.shape[0])
            for i in range(c.shape[0]):
                c[i] = i
                d[i] = i
        return c, d
    a = c.type(torch.long)
    b = d.type(torch.long)

    acc = 0
    cnt = 0
    equal = 0
    acc2 = 0
    # print('ac:', (y[a] == 1).sum(), (y[b] == 1).sum(), ((y[a] == 1) == (y[a] == 1)).sum())
    # print('ya:', (y[a] == 1).shape, ((y[a] == 1) == (y[a] == 1)).shape)
    # a0, a1 = (y[a] == 1).nonzero()
    # yb = (y[b] == 1).nonzero()
    # a0 = torch.tensor(a0)
    # a1 = torch.tensor(a1)
    # print(a0.shape)
    # print(a1.shape)
    # acc2 = (yb[a0, a1] == 1).sum()
    # y = torch.from_numpy(y)

    acc2 = torch.masked_select(y[b] == 1, y[a] == 1).sum()
    '''
    for i in range(a.shape[0]):
        #acc2 += ((y[b[i]] == 1) == (y[a[i]] == 1)).sum()
        for j in range(y.shape[1]):
            if y[a[i]][j] == 1:
                cnt += 1
                if y[b[i]][j] == 1:
                    acc += 1
            if y[a[i]][j] == y[b[i]][j]:
                equal += 1
    '''
    # a_b = (y[a] == y[b])
    # a_1 = (y[a] == 1)
    # print('ab:{} a1:{}'.format(a_b.sum(), a_1.sum()))
    # acc2 = (((y[a] == y[b]) == (y[a] == 1)) == (y[b] == 1)).sum()
    cnt2 = (y[a] == 1).sum()
    # equal2 = (y[a] == y[b]).sum()
    # print('acc2:{} cnt2:{} equal2:{}'.format(acc2, cnt2, equal2))
    # print('equal2:', equal2)
    # sum = a.shape[0] * y.shape[1]
    # print('acc:{} cnt:{} per:{}'.format(acc, cnt, acc / cnt))
    # print('equal:{} sum:{} per:{}'.format(equal, sum, equal / sum))
    return acc2 / cnt2, b.shape[0] / trid.shape[0]


def ttt(view, target):
    l = 0
    r = 10
    for i in range(len(view)):
        view[i] = view[i][l:r]
    target = target[l:r]
    eva_graph(view, target, 1)


def train():
    print('data load begin')
    view, target = data_loader(args.data_root + args.dataset)
    print('data load end')
    target = torch.tensor(target).permute(1, 0)
    print('positive:', target.sum() / (target.shape[0] * target.shape[1]))
    view, target = data_clean(view, target)
    dim_view = []  # 统计视图维度
    # tol = 100
    # for i in range(len(view)):
    # view[i] = view[i][0:tol]
    # target = target[0:tol]

    for i in range(len(view)):
        dim_view.append(view[i].shape[1])
        # view[i] = torch.rand(view[i].shape[0], view[i].shape[1])
        # dic = {}
        # print('i:{} shape:{}'.format(i, view[i].shape))
        # for j in range(view[i].shape[0]):
        # dic[hash(view[i][j].numpy().tobytes())] = j
        # for j in range(view[i].shape[0]):
        # print('i:{} j:{} hash:{} dic:{}'.format(i, j, hash(view[i][j].numpy().tobytes()), dic[hash(view[i][j].numpy().tobytes())]))
        # print('view:', view[i][j].shape)
        # view[i] = torch.tensor(view[i])
        # ma.append(dic)
        # print('dic:', len(dic))
        # cosid.append(cosine_similarity(view[i], view[i]) >= t[i])
    # ttt(view, target, view[0].shape[0])
    # return
    # model = RGCN(dim_view, 128, target.shape[1], 2, 0.2, 2)
    # print('model1:', model)
    # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # print('view:{} target:{}'.format(len(view), target.shape))

    # view, target = shuffle(view, target)
    t_start = time.time()
    best = 0
    print('view:', view[0][0:20], view[0].shape)
    num_k_fold = 1
    for epoch in range(num_k_fold):
        #
        # print('view:', view[0][0:20], view[0].shape)
        # print('num_k_fold:{}/{}'.format(epoch, num_k_fold))
        train_acc_sum, train_loss_sum, test_acc_sum, test_loss_sum, acc_cnt_sum, ham_sum, rank_sum, one_sum = k_fold(
            args.k_fold, view, target)

        best = max(best, test_acc_sum)
        ts = time.time() - t_start
        # print('best:', best)
        # print('time:{} remain:{}'.format(ts, (num_k_fold - epoch - 1) * ts / (epoch + 1)))


def test_code():
    # u, v = get_global_edge_new(5, 2)
    # print(u, v)
    '''
    test_shuffle
    n = 4
    len = 3
    x = []
    y = torch.randn(n, 2)
    for i in range(len):
        x.append(torch.randn(n, 4))
        print(x[i])
    print(y)
    a, b = shuffle(x, y)
    for i in range(len):
        print(a[i])
    print(b)
    '''
    x = [
        torch.tensor([[0, 0, 1], [0, 0, 1]]),
        torch.tensor([[0, 1, 1], [0, 1, 1]]),
        torch.tensor([[0, 2, 1], [0, 2, 1]]),
        torch.tensor([[3, 2, 5], [3, 2, 5]]),
        torch.tensor([[4, 2, 6], [4, 2, 6]]),
    ]
    y = torch.tensor([[0, 1],
                      [1, 1],
                      [1, 0],
                      [0, 1],
                      [1, 0]])
    dim_view = [3]
    construct_graph_k(x, dim_view)


def make_cos(view, t=None):
    cosid = []
    for i in range(len(view)):
        # t[i] = find_t(view[i])
        cosid.append(cosine_similarity(view[i]))
        # a, b = cosid[-1].nonzero()
        # print('a:{} b:{}'.format(a, b))
    return cosid


def _read_conds(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)


def _write_conds(dat, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(dat, f)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    # torch.set_num_threads(3)

    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('--dataset', default='emotions.mat', type=str, metavar='N', help='run_data')
    parser.add_argument('--data_root', default='../data/', type=str, metavar='PATH',
                        help='root dir')
    parser.add_argument('--word2vec', default='data/voc_glove_word2vec.pkl', type=str, metavar='PATH',
                        help='root dir')
    parser.add_argument('--batch_size', default=1, type=int, help='number of batch size')
    parser.add_argument('--k', default=2, type=int, help='KNN')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--end_epochs', default=100, type=int, metavar='H-P',
                        help='number of total epochs to run')
    parser.add_argument('--lr', default=0.001, type=float,
                        metavar='H-P', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--alpha', default=0.9, type=float,
                        metavar='H-P', help='initial learning rate')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('--nhidden', default=128, type=int,
                        metavar='H-P', help='n_hidden')
    parser.add_argument('--k_fold', default=5, type=int,
                        metavar='H-P', help='n_hidden')
    parser.add_argument('--dropout', default=0.2, type=float,
                        metavar='H-P', help='n_hidden')
    parser.add_argument('--layers', default=2, type=int,
                        metavar='H-P', help='n_hidden')
    parser.add_argument('--warmup', default=0, type=int,
                        metavar='H-P', help='n_hidden')
    parser.add_argument('--t', default=0.85, type=float,
                        metavar='H-P', help='n_hidden')
    parser.add_argument('--at', default=0.4, type=float,
                        metavar='H-P', help='n_hidden')
    parser.add_argument('--seed', default=2021, type=int,
                        metavar='H-P', help='n_hidden')
    os.environ["CUDA_VISIBLE_DEVICES"] = ''
    args = parser.parse_args()
    pic = 'record'
    if not os.path.exists(pic):
        os.makedirs(pic)
    pic = pic + '/' + args.dataset + '_lr-' + str(args.lr) + '_epoch-' + str(args.end_epochs) + '_k-' + str(
        args.k) + '_nhidden-' + str(args.nhidden) + '_k-flod-' + str(args.k_fold) + '_batch-size-' + str(
        args.batch_size) + '_warmup-' + str(args.warmup) + '_t-' + str(args.t) + '_at-' + str(args.at) + '_seed' + str(
        args.seed)
    if not os.path.exists(pic):
        os.makedirs(pic)
    sys.stdout = open(pic + '/console.txt', 'w')
    setup_seed(args.seed)
    '''
    maf = []
    #cosidf = []
    total_num = 593
    if args.dataset == 'emotions.mat' or args.dataset == 'Emotions.mat':
        thres = [0.85, 0.999]
    elif args.dataset == 'Yeast.mat':
        thres = [0.85, 0.999]
    '''
    print('train begin')
    train()
    # test_code()
    sys.stdout.close()
