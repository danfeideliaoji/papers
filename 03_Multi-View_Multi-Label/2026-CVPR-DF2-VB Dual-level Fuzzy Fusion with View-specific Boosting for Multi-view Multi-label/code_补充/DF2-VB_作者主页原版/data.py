import numpy as np
import scipy.io as sio
from torch.utils.data import Dataset

from sklearn.preprocessing import StandardScaler, MinMaxScaler, normalize, scale
import torch

import h5py
import scipy.sparse as scsp


class MultiViewDataset(Dataset):
    def __init__(self, data_name, data_X, data_Y, normalize=True):
        super(MultiViewDataset, self).__init__()
        self.data_name = data_name

        self.X = dict()

        if data_name in ['Emotions', 'Human']:
            self.num_views = data_X.shape[1]
            for v in range(self.num_views):
                # print(data_X[0][v].shape)
                if normalize is True:
                    # self.X[v] = self.normalize(data_X[0][v])
                    self.X[v] = StandardScaler().fit_transform(data_X[0][v].astype(np.float32))
                else:
                    self.X[v] = data_X[0][v]
                print(self.X[v].shape)
                self.Y = data_Y.T
        elif data_name in ['Corel5k', 'Pascal']:
            self.num_views = data_X.shape[0]
            for v in range(self.num_views):
                if normalize is True:
                    # self.X[v] = self.normalize(data_X[v][0])
                    self.X[v] = StandardScaler().fit_transform(data_X[v][0].astype(np.float32))
                    # self.X[v] = data_X[v][0]
                else:
                    self.X[v] = data_X[v][0]
                self.Y = data_Y.T
        elif data_name in ['Scene']:  # Scene, Human
            self.num_views = data_X.shape[1]

            for v in range(self.num_views):
                if normalize is True:
                    # self.X[v] = self.normalize(data_X[0][v])
                    self.X[v] = StandardScaler().fit_transform(data_X[0][v].astype(np.float32))
                    # print(self.X[v].shape)
                else:
                    self.X[v] = data_X[0][v]

            self.Y = data_Y
        else:  # Human
            self.num_views = data_X.shape[1]
            for v in range(self.num_views):
                if normalize is True:
                    self.X[v] = self.normalize(data_X[0][v])
                    # self.X[v] = StandardScaler().fit_transform(data_X[0][v].astype(np.float32))

                    print('X Shape:', v, self.X[v].shape)
                    print(self.X[v])
                else:
                    self.X[v] = data_X[0][v]
            # print('X Shape:', data_X.shape)
            # print('Y Shape:', data_Y.shape)
            self.Y = data_Y.T
        # self.Y = np.squeeze(self.Y)
        # 调整标签值
        # if np.min(self.Y) == 1:
        #     self.Y = self.Y - 1
        self.Y = self.Y.astype(dtype=np.int64)
        self.num_classes = self.Y.shape[1]
        print(self.num_classes)
        self.dims = self.get_dims()

    # 获取下标index在所有视图下的数据与对应的标签
    def __getitem__(self, index):
        data = dict()
        # print(len(self.X))
        for v_num in range(len(self.X)):
            # print(self.X[v_num].shape)
            data[v_num] = (self.X[v_num][index]).astype(np.float32)
        # print(self.Y)
        # print(self.Y.shape)
        target = self.Y[index]
        return data, target, index

    def __len__(self):
        return len(self.X[0])

    def get_dims(self):
        dims = []
        for view in range(self.num_views):
            dims.append([self.X[view].shape[1]])
        return np.array(dims)

    @staticmethod
    def normalize(x, min=0):
        if min == 0:
            scaler = MinMaxScaler((0, 1))
        else:  # min=-1
            scaler = MinMaxScaler((-1, 1))
        norm_x = scaler.fit_transform(x)
        return norm_x

    def postprocessing(self, index, addNoise=False, sigma=0, ratio_noise=0.5, addConflict=False, ratio_conflict=0.5):
        if addNoise:
            self.addNoise(index, ratio_noise, sigma=sigma)
        if addConflict:
            self.addConflict(index, ratio_conflict)
        pass

    def addNoise(self, index, ratio, sigma):
        selects = np.random.choice(index, size=int(ratio * len(index)), replace=False)
        for i in selects:
            views = np.random.choice(np.array(self.num_views), size=np.random.randint(self.num_views), replace=False)
            for v in views:
                self.X[v][i] = np.random.normal(self.X[v][i], sigma)
        pass

    def addConflict(self, index, ratio):
        records = dict()
        for c in range(self.num_classes):
            i = np.where(self.Y == c)[0][0]
            temp = dict()
            for v in range(self.num_views):
                temp[v] = self.X[v][i]
            records[c] = temp
        selects = np.random.choice(index, size=int(ratio * len(index)), replace=False)
        for i in selects:
            v = np.random.randint(self.num_views)
            self.X[v][i] = records[(self.Y[i] + 1) % self.num_classes][v]
        pass


def Scene(normalize=False):
    data_path = 'scene.mat'
    print('Load data...')
    data = sio.loadmat(data_path)
    data_X = data['data']
    data_Y = data['target']

    print(data_X.shape)
    print(data_X)

    return MultiViewDataset("Scene", data_X, data_Y, normalize=normalize)

    return data_X, data_Y
