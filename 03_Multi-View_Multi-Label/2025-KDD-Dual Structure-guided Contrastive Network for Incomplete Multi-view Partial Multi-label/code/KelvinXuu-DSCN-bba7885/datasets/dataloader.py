from sklearn.preprocessing import MinMaxScaler, StandardScaler, Normalizer
import torch
from torch.utils.data import DataLoader, random_split, Dataset
import scipy.io as sio
import numpy as np


transform_dict = {
    "corel5k": StandardScaler(),
    "espgame": StandardScaler(),
    "iaprtc12": StandardScaler(),
    "mirflickr": StandardScaler(),
    "pascal07": StandardScaler(),
}


class MatDataset(Dataset):
    def __init__(self, args, mode="train"):
        flag= False
        if args.missing_rate_data==0.5 and  args.missing_rate_label==0:
            args.missing_rate_label = 0.5
            flag = True
        self.transform = transform_dict[args.dataset_name]

        datas = sio.loadmat("../data/{}/{}_six_view.mat".format(args.dataset_name, args.dataset_name))
        datafold = sio.loadmat("../data/{}/{}_six_view_MaskRatios_{}_LabelMaskRatio_{}_TraindataRatio_{}.mat".format(
            args.dataset_name, args.dataset_name, args.missing_rate_data, args.missing_rate_label, args.training_ratio))
        X = datas['X'][0]
        labels = datas['label']
        self.mask_data = datafold['folds_data'][0][args.fold_idx]
        self.mask_label = datafold['folds_label'][0][args.fold_idx]
        folds_sample_index = datafold['folds_sample_index'][0][args.fold_idx].reshape(-1) - 1
        folds_sample_index = np.array(folds_sample_index, dtype=np.int32)

        self.mask_data = self.mask_data[folds_sample_index]
        self.mask_label = self.mask_label[folds_sample_index]
        labels = labels[folds_sample_index]
        if flag:
            self.mask_label = np.ones_like(self.mask_label)
            args.missing_rate_label = 0

        self.total_samples = X[0].shape[0]
        self.view_num = X.shape[0]
        self.class_num = labels.shape[1]

        train_size = int(args.training_ratio * self.total_samples)
        val_size = int((1 - args.training_ratio) / 2 * self.total_samples)
        test_size = self.total_samples - train_size - val_size

        self.data = []
        self.labels = []

        # size
        if mode == "train":
            self.size = train_size
        elif mode == "val":
            self.size = val_size
        else:
            self.size = test_size

        if mode == "train":
            self.labels = labels[:train_size]
            self.mask_data = self.mask_data[:train_size]
            self.mask_label = self.mask_label[:train_size] 
            self.labels = self.labels * self.mask_label
        elif mode == "val":
            self.labels = labels[train_size:train_size + val_size]
            self.mask_data = self.mask_data[train_size:train_size + val_size]
            self.mask_label = self.mask_label[train_size:train_size + val_size]
        else:
            self.labels = labels[train_size + val_size:]
            self.mask_data = self.mask_data[train_size + val_size:]
            self.mask_label = self.mask_label[train_size + val_size:]
        self.labels = torch.tensor(self.labels * 1.0).to(torch.float32)

        # data
        for i in range(X.shape[0]):
            X[i] = X[i][folds_sample_index, :]
            data_matrix = self.transform.fit_transform(X[i])

            if mode == "train":
                data = data_matrix[:train_size] * self.mask_data[:, i][:, np.newaxis]
            elif mode == "val":
                data = data_matrix[train_size:train_size + val_size] * self.mask_data[:, i][:, np.newaxis]
            else:
                data = data_matrix[train_size + val_size:] * self.mask_data[:, i][:, np.newaxis]

            data = torch.tensor(data).to(torch.float32)
            self.data.append(data)

        self.mask_data = torch.tensor(self.mask_data).to(torch.float32)
        self.mask_label = torch.tensor(self.mask_label).to(torch.float32)
        self.indices = torch.tensor(range(self.size))

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        datas = [data[idx] for data in self.data]
        label = self.labels[idx]
        mask_data = self.mask_data[idx]
        mask_label = self.mask_label[idx]
        index = self.indices[idx]

        return datas, label, mask_data, mask_label, index


def get_dataloader(args, dataset, mode="train"):
    if mode == "train":
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    else:
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    return dataloader

