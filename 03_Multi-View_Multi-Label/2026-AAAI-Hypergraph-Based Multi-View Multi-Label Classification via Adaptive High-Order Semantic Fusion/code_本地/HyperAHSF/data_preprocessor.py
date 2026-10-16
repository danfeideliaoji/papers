"""多视图多标签数据的格式转换、K 折划分与 PyTorch 数据集封装。

原始数据可能来自传统 MATLAB v7 ``.mat`` 文件或 HDF5 格式的 MATLAB
v7.3 文件。预处理流程先按视图读取特征、进行 StandardScaler 标准化和
K 折划分，再把每折数据转为 pickle；训练阶段由 ``MyDataset_k_fold``
一次性将整折数据加载到指定设备。
"""

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
    """把一折数据按视图保存为 pickle 文件。

    Args:
        name (str): 数据集名称，对应输出目录的第一层。
        index (int): 折编号。
        type (str): 数据划分名称，通常为 ``'train'`` 或 ``'val'``。
        features (list[numpy.ndarray]): 多视图特征列表，第 ``v`` 项形状为 ``[N, D_v]``。
        hypergraph (torch.Tensor | numpy.ndarray | None): 可选的预构造超图；当前主流程保存为 ``None``，模型会
            根据特征和标签动态构图。
        labels (numpy.ndarray | torch.Tensor): 多标签矩阵，形状为 ``[N, C]``。
    """
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
    """把一折 ``.mat`` 数据转换为训练阶段直接读取的 pickle 结构。

    ``.mat`` 文件需要包含 ``data0`` ... ``data{V-1}`` 和 ``target``。
    特征按视图分别保存，标签最终保持 ``[N, C]`` 布局。

    Args:
        mat_file_path (str | pathlib.Path): K 折 MAT 文件所在的根目录。
        name (str): 数据集名称。
        index (int): 要转换的折编号。
        type (str): 数据划分名称，通常是 ``'train'`` 或 ``'val'``。
        num_views (int): 需要读取的视图数量 ``V``。
    """
    data = scipy.io.loadmat(osp.join(mat_file_path, name, 'fold' + str(index), type + '.mat'))
    features_view = []
    for i in range(num_views):
        feature_view_idx = data['data'+str(i)]
        print(f'the view_{i} feature shape is {feature_view_idx.shape}')
        features_view.append(feature_view_idx)
    labels = data['target']
    labels = labels.T
    # 当前版本不在预处理阶段构图；真正的训练/测试超图由模型动态生成。
    hyperedge_index = None
    # 与上方转置配对，使保存布局恢复为模型使用的 [N, C]。
    labels = labels.T
    print('the label shape is {}'.format(labels.shape))
    save_k_fold_Multi_View_dataset(name,index, type, features_view, hyperedge_index, labels)
    print(f'Finish preprocess other dataset: {name} / {type}')
    print(' ')

class MyDataset_k_fold(Dataset):
    """一折多视图数据集。

    与常规逐样本 DataLoader 用法不同，训练代码直接读取 ``features`` 和
    ``labels``，一次性对整折数据进行超图传播。``__getitem__`` 仍保留标准
    Dataset 接口，便于需要时按样本访问。
    """

    def __init__(self, num_views: int, name: str, index: int, type:str, device: str = 'cuda:0', scalers = None):
        """定位指定折的数据，加载后转换为目标设备上的张量。

        Args:
            num_views (int): 视图数 ``V``。
            name (str): 数据集名称。
            index (int): 折编号。
            type (str): ``'train'`` 或 ``'val'``。
            device (str | torch.device): 张量所在的 CPU/GPU 设备。
            scalers (list[sklearn.preprocessing.StandardScaler] | None): 预留的标准化器列表。当前 pickle 已在生成阶段完成
                标准化，因此加载阶段不会调用其中的 scaler。
        """
        self.scalers = scalers
        self.name = name
        self.type = type
        self.num_views = num_views
        self.device = device
        # 约定目录：.../<dataset>/fold<i>/<split>/。
        self.dataset_dir = osp.join('Multi_View_dataset_k_fold_pickle', self.name, 'fold' + str(index), self.type)
        self.features = []  # 长度为 V；每项最终为 [N, D_v] 的 float32 Tensor。
        self.hypergraph = None
        self.labels = None
        self.hyperedge_index = None
        self.num_nodes = None
        self.num_edges = None
        self.load_dataset()
        self.preprocess_dataset()

    def load_dataset(self):
        """从当前折目录读取各视图特征、占位超图和标签。"""
        if self.type == 'train' and self.scalers is None:
            self.scalers = []

        for i in range(self.num_views):
            with open(osp.join(self.dataset_dir, 'features_view' + str(i) + '.pickle'), 'rb') as f:
                feature_view = pickle.load(f)
                self.features.append(feature_view)
        with open(osp.join(self.dataset_dir, 'hypergraph.pickle'), 'rb') as f:
            self.hypergraph = pickle.load(f)
        with open(osp.join(self.dataset_dir, 'labels.pickle'), 'rb') as f:
            self.labels = pickle.load(f)

    def preprocess_dataset(self):
        """将 NumPy/pickle 数据转为设备上的 PyTorch 张量。

        Returns:
            当前数据集实例，便于链式调用。
        """
        # torch.as_tensor 尽量复用原数组内存；to(float32) 统一类型，最后一个 to 迁移到 CPU/GPU。
        self.features = [torch.as_tensor(f).to(torch.float32).to(self.device) for f in self.features]
        # LongTensor 表示 64 位整数张量，适合保存离散标签或索引。
        self.labels = torch.LongTensor(self.labels).to(self.device)
        self.num_nodes = self.features[0].shape[0]
        return self

    def __len__(self):
        """返回当前折的样本数 N。"""
        return self.features[0].shape[0]

    def __getitem__(self, index):
        """返回一个样本在全部视图中的特征列表及其多标签向量。

        Args:
            index (int): 样本下标，范围为 ``0..N-1``。

        Returns:
            tuple[list[torch.Tensor], torch.Tensor]: 视图特征列表和形状 ``[C]`` 的标签。
        """
        return [self.features[i][index] for i in range(self.num_views)], self.labels[index]

def generate_k_fold_from_mat_emotions_plant_human(dataset_mat_name:str, seed:int=8848, k=10):
    """为 emotions/plant/human 风格的 MATLAB cell 数组生成 K 折数据。

    该类文件以 ``data[0][v]`` 保存第 ``v`` 个视图，以 ``target`` 保存标签。
    每折轮流作为验证集，其余折拼成训练集，并写出 ``train.mat`` 与
    ``val.mat``。

    Args:
        dataset_mat_name (str): 原始 MAT 文件路径或数据集名称。
        seed (int): 打乱样本下标的随机种子。
        k (int): K 折划分的折数。

    Notes:
        当前实现先在全部样本上拟合 StandardScaler，再进行 K 折划分；这与
        原始实验保持一致，但严格评估时会引入验证集统计量泄漏。
    """
    dataset_mat_dir = osp.join('Multi_View_dataset', dataset_mat_name)
    data = scipy.io.loadmat(dataset_mat_dir)
    feature_data = data['data']
    label_data = data['target']
    label_data = label_data.T
    num_views = feature_data.shape[1]

    # 每个视图独立按特征列做零均值、单位方差标准化。
    for i in range(num_views):
        scaler = preprocessing.StandardScaler().fit(feature_data[0][i])
        feature_data[0][i] = scaler.transform(feature_data[0][i])

    n_samples = (feature_data[0][0]).shape[0]
    indices = np.arange(n_samples)
    # 固定随机种子后打乱样本编号，保证折划分可复现。
    np.random.seed(seed)
    np.random.shuffle(indices)
    # 尽可能均匀分配样本；前若干折各多接收一个余数样本。
    fold_sizes = np.full(k, n_samples // k, dtype=int)
    fold_sizes[:n_samples % k] += 1
    # 按 fold_sizes 顺序切分打乱后的下标。
    current = 0
    folds = []
    for fold_size in fold_sizes:
        start, stop = current, current + fold_size
        folds.append(indices[start:stop])
        current = stop
    all_indices = np.concatenate(folds)
    assert np.array_equal(np.sort(all_indices), np.arange(n_samples)), "Sample allocation error"
    # 第 i 折作为验证集，其余 k-1 折拼接为训练集。
    train_val = []
    for i in range(k):
        val_indices = folds[i]
        train_indices = np.concatenate(folds[:i] + folds[i + 1:])
        train_val.append((train_indices, val_indices))
    # 将每个视图和标签按当前训练/验证下标切片并保存。
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
    """为 Scene 风格数据生成 K 折训练/验证文件。

    数据布局同 ``generate_k_fold_from_mat_emotions_plant_human``，区别在于
    ``target`` 已经是当前流程需要的样本优先布局，因此不额外转置。

    Args:
        dataset_mat_name (str): 原始 MAT 文件路径或数据集名称。
        seed (int): 打乱样本下标的随机种子。
        k (int): K 折划分的折数。
    """
    dataset_mat_dir = osp.join('Multi_View_dataset', dataset_mat_name)
    data = scipy.io.loadmat(dataset_mat_dir)
    feature_data = data['data']
    label_data = data['target']
    num_views = feature_data.shape[1]

    # 标准化在划分前完成，以保持原代码实验行为。
    for i in range(num_views):
        scaler = preprocessing.StandardScaler().fit(feature_data[0][i])
        feature_data[0][i] = scaler.transform(feature_data[0][i])

    n_samples = (feature_data[0][0]).shape[0]
    indices = np.arange(n_samples)
    # 生成可复现的随机折下标。
    np.random.seed(seed)
    np.random.shuffle(indices)
    # 让各折大小之差不超过 1。
    fold_sizes = np.full(k, n_samples // k, dtype=int)
    fold_sizes[:n_samples % k] += 1
    # 切分折下标并检查每个样本恰好出现一次。
    current = 0
    folds = []
    for fold_size in fold_sizes:
        start, stop = current, current + fold_size
        folds.append(indices[start:stop])
        current = stop
    all_indices = np.concatenate(folds)
    assert np.array_equal(np.sort(all_indices), np.arange(n_samples)), "Sample allocation error"
    # 轮流使用一折作为验证集。
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
    """为 Yeast/Corel5k/Pascal 风格数据生成 K 折文件。

    这类 MATLAB cell 的视图维位于第一轴，因此第 ``v`` 个视图通过
    ``feature_data[v][0]`` 访问；其余划分逻辑与前述函数相同。

    Args:
        dataset_mat_name (str): 原始 MAT 文件路径或数据集名称。
        seed (int): 打乱样本下标的随机种子。
        k (int): K 折划分的折数。
    """
    dataset_mat_dir = osp.join('Multi_View_dataset', dataset_mat_name)
    data = scipy.io.loadmat(dataset_mat_dir)
    feature_data = data['data']
    label_data = data['target']
    label_data = label_data.T
    num_views = feature_data.shape[0]

    # 各视图独立标准化；当前行为仍是在全数据上 fit 后再划分。
    for i in range(num_views):
        scaler = preprocessing.StandardScaler().fit(feature_data[i][0])
        feature_data[i][0] = scaler.transform(feature_data[i][0])

    n_samples = (feature_data[0][0]).shape[0]
    indices = np.arange(n_samples)
    # 固定种子并打乱样本编号。
    np.random.seed(seed)
    np.random.shuffle(indices)
    # 计算各折大小并分配余数。
    fold_sizes = np.full(k, n_samples // k, dtype=int)
    fold_sizes[:n_samples % k] += 1
    # 依次生成每折下标。
    current = 0
    folds = []
    for fold_size in fold_sizes:
        start, stop = current, current + fold_size
        folds.append(indices[start:stop])
        current = stop
    all_indices = np.concatenate(folds)
    assert np.array_equal(np.sort(all_indices), np.arange(n_samples)), "Sample allocation error"
    # 为每个验证折收集其余训练折。
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
    """为 MATLAB v7.3/HDF5 格式的大型多视图数据生成 K 折文件。

    HDF5 中的 ``data`` 保存对各视图数据集的引用。读取后转置为 ``[N,D_v]``，
    标签也转换成 ``[N,C]``，再沿用相同的标准化和 K 折保存流程。

    Args:
        dataset_mat_name (str): HDF5/MAT v7.3 文件路径或数据集名称。
        seed (int): 打乱样本下标的随机种子。
        k (int): K 折划分的折数。
    """
    dataset_mat_dir = osp.join('Multi_View_dataset', dataset_mat_name)
    data = h5py.File(dataset_mat_dir, 'r')
    feature_data = []
    # 解引用每个视图并转为 float32；原文件以特征维在前，因此需要转置。
    for i in range(6):
        feature_data.append(np.array(data[data['data'][0][i]]).astype(np.float32).T)

    label_data = np.array(data['target']).astype(np.int32).T
    num_views = data['data'].shape[1]

    # 每个视图独立标准化。
    for i in range(num_views):
        scaler = preprocessing.StandardScaler().fit(feature_data[i])
        feature_data[i] = scaler.transform(feature_data[i])

    n_samples = (feature_data[0]).shape[0]
    indices = np.arange(n_samples)
    # 可复现地打乱样本编号。
    np.random.seed(seed)
    np.random.shuffle(indices)
    # 均匀分配折大小。
    fold_sizes = np.full(k, n_samples // k, dtype=int)
    fold_sizes[:n_samples % k] += 1
    # 生成各折索引并验证没有遗漏或重复分配。
    current = 0
    folds = []
    for fold_size in fold_sizes:
        start, stop = current, current + fold_size
        folds.append(indices[start:stop])
        current = stop
    all_indices = np.concatenate(folds)
    assert np.array_equal(np.sort(all_indices), np.arange(n_samples)), "Sample allocation error"
    # 轮流将每一折作为验证集。
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
    # 默认预处理 emotions：生成 5 折 MAT 文件，再转换为训练直接读取的 pickle。
    generate_k_fold_from_mat_emotions_plant_human(dataset_mat_name='emotions', seed=8848, k=5)
    for i in range(5):
        convert_k_fold_mat_to_pickle('Multi_View_dataset_k_fold', 'emotions', i, 'train', 2)
        convert_k_fold_mat_to_pickle('Multi_View_dataset_k_fold', 'emotions', i, 'val', 2)

