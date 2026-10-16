import scipy.io
import numpy as np
import os.path as osp
from pathlib import Path
import pickle
import torch
from torch.utils.data import Dataset
from sklearn import preprocessing
import h5py

def save_k_fold_Multi_View_dataset(name, index, type, features, hypergraph, labels): # features is a list
    new_dir = osp.join('Multi_View_dataset_k_fold_pickle', name, 'fold' + str(index), type)
    Path(new_dir).mkdir(parents=True, exist_ok=True)
    for i,feature_view in enumerate(features):
        with open(osp.join(new_dir, 'features_view'+str(i)+'.pickle'), 'wb') as f:
            pickle.dump(feature_view, f)
    with open(osp.join(new_dir, 'hypergraph.pickle'), 'wb') as f:
        pickle.dump(hypergraph, f)
    with open(osp.join(new_dir, 'labels.pickle'), 'wb') as f:
        pickle.dump(labels, f)

def convert_k_fold_mat_to_pickle(mat_file_path, name:str, index:int, type:str, num_views): # Convert the num_views views of the specified dataset into features, hypergraph, and labels
    data = scipy.io.loadmat(osp.join(mat_file_path, name, 'fold' + str(index), type + '.mat'))
    features_view = []
    for i in range(num_views):
        feature_view_idx = data['data'+str(i)]
        print(f'the view_{i} feature shape is {feature_view_idx.shape}')
        features_view.append(feature_view_idx)
        # print(feature_view_idx[0][0].dtype)
    labels = data['target']
    labels = labels.T
    # The label matrix used here is the transpose of the normal label matrix
    # Build hypergraph from label matrix. Since there are no instances without any labels in multi-view multi-label classification, we don't need to consider isolated points.
    hyperedge_index = None
    # Used to test if there are any samples without labels
    # for classes in labels.T:
    #     if np.array_equal(classes, np.zeros(6)):
    #         print("ok")
    labels = labels.T
    print('the label shape is {}'.format(labels.shape))
    save_k_fold_Multi_View_dataset(name,index, type, features_view, hyperedge_index, labels)
    print(f'Finish preprocess other dataset: {name} / {type}')
    print(' ')

class MyDataset_k_fold(Dataset):
    def __init__(self, num_views: int, name: str, index: int, type:str, device: str = 'cuda:0', scalers = None):
        self.scalers = scalers
        self.name = name
        self.type = type
        self.num_views = num_views
        self.device = device
        # Note: modify this
        self.dataset_dir = osp.join('Multi_View_dataset_k_fold_pickle', self.name, 'fold' + str(index), self.type)
        self.features = [] # is a list
        self.hypergraph = None
        self.labels = None
        self.hyperedge_index = None
        self.num_nodes = None
        self.num_edges = None
        self.load_dataset()
        self.preprocess_dataset()

    def load_dataset(self):
        if self.type == 'train' and self.scalers is None:
            self.scalers = []

        for i in range(self.num_views):
            with open(osp.join(self.dataset_dir, 'features_view' + str(i) + '.pickle'), 'rb') as f:
                feature_view = pickle.load(f)
                # print(feature_view[0][0].dtype)
                self.features.append(feature_view)
        with open(osp.join(self.dataset_dir, 'hypergraph.pickle'), 'rb') as f:
            self.hypergraph = pickle.load(f)
            # print(self.hypergraph)
        with open(osp.join(self.dataset_dir, 'labels.pickle'), 'rb') as f:
            self.labels = pickle.load(f)

    def preprocess_dataset(self):
        self.features = [torch.as_tensor(f).to(torch.float32).to(self.device) for f in self.features]
        self.labels = torch.LongTensor(self.labels).to(self.device)
        self.num_nodes = self.features[0].shape[0]
        return self

    def __len__(self):
        # Return the number of samples in the dataset
        return self.features[0].shape[0]

    def __getitem__(self, index):
        # Return a single sample and its label
        return [self.features[i][index] for i in range(self.num_views)], self.labels[index]

def generate_k_fold_from_mat_emotions_plant_human(dataset_mat_name:str, seed:int=8848, k=10):
    # Split the original mat file into three mat files according to the mask. Actually equivalent to dictionaries with keys: data(i) and target. Note: the target here is transposed (correct shape)
    # Returns the number of views in the mat data
    dataset_mat_dir = osp.join('Multi_View_dataset', dataset_mat_name)
    data = scipy.io.loadmat(dataset_mat_dir)
    feature_data = data['data']
    label_data = data['target']
    label_data = label_data.T
    num_views = feature_data.shape[1]

    for i in range(num_views):
        scaler = preprocessing.StandardScaler().fit(feature_data[0][i])
        feature_data[0][i] = scaler.transform(feature_data[0][i])

    n_samples = (feature_data[0][0]).shape[0]
    indices = np.arange(n_samples)
    # Shuffle data order
    np.random.seed(seed)
    np.random.shuffle(indices)
    # Calculate number of samples per fold
    fold_sizes = np.full(k, n_samples // k, dtype=int)
    fold_sizes[:n_samples % k] += 1  # Handle remainder
    # Generate indices for each fold
    current = 0
    folds = []
    for fold_size in fold_sizes:
        start, stop = current, current + fold_size
        folds.append(indices[start:stop])
        current = stop
    all_indices = np.concatenate(folds)
    assert np.array_equal(np.sort(all_indices), np.arange(n_samples)), "Sample allocation error"
    # Generate training and validation set indices
    train_val = []
    for i in range(k):
        val_indices = folds[i]
        train_indices = np.concatenate(folds[:i] + folds[i + 1:])
        train_val.append((train_indices, val_indices))
    for idx, train_val_split in enumerate(train_val):
        train_data = {}
        val_data = {}
        for i in range(num_views):
            feature_data_train_view = feature_data[0][i][train_val_split[0]]
            train_data['data' + str(i)] = feature_data_train_view
            feature_data_val_view = feature_data[0][i][train_val_split[1]]
            val_data['data' + str(i)] = feature_data_val_view
        train_data['target'] = label_data[train_val_split[0]]
        val_data['target'] = label_data[train_val_split[1]]
        save_dir_train = osp.join('Multi_View_dataset_k_fold', dataset_mat_name, 'fold'+str(idx))
        Path(save_dir_train).mkdir(parents=True, exist_ok=True)
        scipy.io.savemat(osp.join(save_dir_train, 'train.mat'), train_data)
        scipy.io.savemat(osp.join(save_dir_train, 'val.mat'), val_data)
    print('Finish generating k-fold')

def generate_k_fold_from_mat_scene(dataset_mat_name:str, seed:int=0, k=10):
    # Split the original mat file into three mat files according to the mask. Actually equivalent to dictionaries with keys: data(i) and target. Note: the target here is transposed (correct shape)
    # Returns the number of views in the mat data
    dataset_mat_dir = osp.join('Multi_View_dataset', dataset_mat_name)
    data = scipy.io.loadmat(dataset_mat_dir)
    feature_data = data['data']
    label_data = data['target']
    num_views = feature_data.shape[1]

    for i in range(num_views):
        scaler = preprocessing.StandardScaler().fit(feature_data[0][i])
        feature_data[0][i] = scaler.transform(feature_data[0][i])

    n_samples = (feature_data[0][0]).shape[0]
    indices = np.arange(n_samples)
    # Shuffle data order
    np.random.seed(seed)
    np.random.shuffle(indices)
    # Calculate number of samples per fold
    fold_sizes = np.full(k, n_samples // k, dtype=int)
    fold_sizes[:n_samples % k] += 1  # Handle remainder
    # Generate indices for each fold
    current = 0
    folds = []
    for fold_size in fold_sizes:
        start, stop = current, current + fold_size
        folds.append(indices[start:stop])
        current = stop
    all_indices = np.concatenate(folds)
    assert np.array_equal(np.sort(all_indices), np.arange(n_samples)), "Sample allocation error"
    # Generate training and validation set indices
    train_val = []
    for i in range(k):
        val_indices = folds[i]
        train_indices = np.concatenate(folds[:i] + folds[i + 1:])
        train_val.append((train_indices, val_indices))
    for idx, train_val_split in enumerate(train_val):
        train_data = {}
        val_data = {}
        for i in range(num_views):
            feature_data_train_view = feature_data[0][i][train_val_split[0]]
            train_data['data' + str(i)] = feature_data_train_view
            feature_data_val_view = feature_data[0][i][train_val_split[1]]
            val_data['data' + str(i)] = feature_data_val_view
        train_data['target'] = label_data[train_val_split[0]]
        val_data['target'] = label_data[train_val_split[1]]
        save_dir_train = osp.join('Multi_View_dataset_k_fold', dataset_mat_name, 'fold'+str(idx))
        Path(save_dir_train).mkdir(parents=True, exist_ok=True)
        scipy.io.savemat(osp.join(save_dir_train, 'train.mat'), train_data)
        scipy.io.savemat(osp.join(save_dir_train, 'val.mat'), val_data)
    print('Finish generating k-fold')

def generate_k_fold_from_mat_Yeast_corel5k_Pascal(dataset_mat_name:str, seed:int=3407, k=10):
    # Split the original mat file into three mat files according to the mask. Actually equivalent to dictionaries with keys: data(i) and target. Note: the target here is transposed (correct shape)
    # Returns the number of views in the mat data
    dataset_mat_dir = osp.join('Multi_View_dataset', dataset_mat_name)
    data = scipy.io.loadmat(dataset_mat_dir)
    feature_data = data['data']
    label_data = data['target']
    label_data = label_data.T
    num_views = feature_data.shape[0]

    for i in range(num_views):
        scaler = preprocessing.StandardScaler().fit(feature_data[i][0])
        feature_data[i][0] = scaler.transform(feature_data[i][0])
        # print(feature_data[i][0][0][0].dtype)

    n_samples = (feature_data[0][0]).shape[0]
    indices = np.arange(n_samples)
    # Shuffle data order
    np.random.seed(seed)
    np.random.shuffle(indices)
    # Calculate number of samples per fold
    fold_sizes = np.full(k, n_samples // k, dtype=int)
    fold_sizes[:n_samples % k] += 1  # Handle remainder
    # Generate indices for each fold
    current = 0
    folds = []
    for fold_size in fold_sizes:
        start, stop = current, current + fold_size
        folds.append(indices[start:stop])
        current = stop
    all_indices = np.concatenate(folds)
    assert np.array_equal(np.sort(all_indices), np.arange(n_samples)), "Sample allocation error"
    # Generate training and validation set indices
    train_val = []
    for i in range(k):
        val_indices = folds[i]
        train_indices = np.concatenate(folds[:i] + folds[i + 1:])
        train_val.append((train_indices, val_indices))
    for idx, train_val_split in enumerate(train_val):
        train_data = {}
        val_data = {}
        for i in range(num_views):
            feature_data_train_view = feature_data[i][0][train_val_split[0]]
            train_data['data' + str(i)] = feature_data_train_view
            feature_data_val_view = feature_data[i][0][train_val_split[1]]
            val_data['data' + str(i)] = feature_data_val_view
        train_data['target'] = label_data[train_val_split[0]]
        val_data['target'] = label_data[train_val_split[1]]
        save_dir_train = osp.join('Multi_View_dataset_k_fold', dataset_mat_name, 'fold'+str(idx))
        Path(save_dir_train).mkdir(parents=True, exist_ok=True)
        scipy.io.savemat(osp.join(save_dir_train, 'train.mat'), train_data)
        scipy.io.savemat(osp.join(save_dir_train, 'val.mat'), val_data)
    print('Finish generating k-fold')

def generate_k_fold_from_mat_iaprtc12_epsgame_mirflickr(dataset_mat_name:str, seed:int=0, k=10):
    # Split the original mat file into three mat files according to the mask. Actually equivalent to dictionaries with keys: data(i) and target. Note: the target here is transposed (correct shape)
    # Returns the number of views in the mat data
    dataset_mat_dir = osp.join('Multi_View_dataset', dataset_mat_name)
    data = h5py.File(dataset_mat_dir, 'r')
    feature_data = []
    for i in range(6):
        feature_data.append(np.array(data[data['data'][0][i]]).astype(np.float32).T)

    label_data = np.array(data['target']).astype(np.int32).T
    num_views = data['data'].shape[1]

    for i in range(num_views):
        scaler = preprocessing.StandardScaler().fit(feature_data[i])
        feature_data[i] = scaler.transform(feature_data[i])

    n_samples = (feature_data[0]).shape[0]
    indices = np.arange(n_samples)
    # Shuffle data order
    np.random.seed(seed)
    np.random.shuffle(indices)
    # Calculate number of samples per fold
    fold_sizes = np.full(k, n_samples // k, dtype=int)
    fold_sizes[:n_samples % k] += 1  # Handle remainder
    # Generate indices for each fold
    current = 0
    folds = []
    for fold_size in fold_sizes:
        start, stop = current, current + fold_size
        folds.append(indices[start:stop])
        current = stop
    all_indices = np.concatenate(folds)
    assert np.array_equal(np.sort(all_indices), np.arange(n_samples)), "Sample allocation error"
    # Generate training and validation set indices
    train_val = []
    for i in range(k):
        val_indices = folds[i]
        train_indices = np.concatenate(folds[:i] + folds[i + 1:])
        train_val.append((train_indices, val_indices))
    for idx, train_val_split in enumerate(train_val):
        train_data = {}
        val_data = {}
        for i in range(num_views):
            feature_data_train_view = feature_data[i][train_val_split[0]]
            train_data['data' + str(i)] = feature_data_train_view
            feature_data_val_view = feature_data[i][train_val_split[1]]
            val_data['data' + str(i)] = feature_data_val_view
        train_data['target'] = label_data[train_val_split[0]]
        val_data['target'] = label_data[train_val_split[1]]
        save_dir_train = osp.join('Multi_View_dataset_k_fold', dataset_mat_name, 'fold'+str(idx))
        Path(save_dir_train).mkdir(parents=True, exist_ok=True)
        scipy.io.savemat(osp.join(save_dir_train, 'train.mat'), train_data)
        scipy.io.savemat(osp.join(save_dir_train, 'val.mat'), val_data)
    print('Finish generating k-fold')

if __name__ == '__main__':
    generate_k_fold_from_mat_emotions_plant_human(dataset_mat_name='emotions', seed=8848, k=5)
    for i in range(5):
        convert_k_fold_mat_to_pickle('Multi_View_dataset_k_fold', 'emotions', i, 'train', 2)
        convert_k_fold_mat_to_pickle('Multi_View_dataset_k_fold', 'emotions', i, 'val', 2)

