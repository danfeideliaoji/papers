"""HyperAHSF 的五折训练入口。

本模块负责读取数据集配置、逐折创建训练/验证数据、初始化模型与优化器，
并在每个 epoch 后计算八项多标签分类指标。模型内部一次处理整折数据，
因此这里没有使用 ``DataLoader`` 进行小批量训练。
"""

import torch
import random
import argparse
import yaml
from tqdm import tqdm
import numpy as np
import torch.nn as nn
from pathlib import Path
from utils import PrecomputedContrastiveLoss
from metrics import test_metrics
from data_preprocessor import MyDataset_k_fold
from model import HGMVMLC_Model

def fix_seed(seed):
    """固定 Python、NumPy 与 PyTorch 的随机状态，尽量保证实验可复现。

    Args:
        seed (int): 随机种子。

    Notes:
        关闭 cuDNN benchmark 并启用确定性算法会牺牲部分运行速度，但可以减少
        同一配置重复运行时的数值波动。
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.Generator().manual_seed(seed)

def train_model(dataset, model, Optim):
    """对当前折的完整训练集执行一次前向传播和参数更新。

    Args:
        dataset (MyDataset_k_fold): 当前训练折；``features`` 是长度为视图数 ``V``
            的列表，每项形状为 ``[N, D_v]``，``labels`` 形状为 ``[N, C]``。
        model (HGMVMLC_Model): 当前训练模型。
        Optim (torch.optim.Optimizer): 已绑定模型参数的 PyTorch 优化器。

    Notes:
        ``pcl`` 在主程序中按当前训练折创建，是本函数读取的模块级变量。
        总损失由多标签二元交叉熵与预计算分组对比损失相加得到。
    """
    fix_seed(0)
    model.train()
    Optim.zero_grad(set_to_none=True)
    features = dataset.features
    labels = dataset.labels
    # output 为未过 sigmoid 的多标签 logits；后两项是公共/特异表示。
    output, _, specific_representations = model(features)
    fix_seed(0)
    # torch.sigmoid 把 logits 映射到 (0,1) 概率；BCELoss 再逐标签计算二元交叉熵。
    criterion_cla = nn.BCELoss()
    cla_loss = criterion_cla(torch.sigmoid(output), labels.to(torch.float32))
    cl_loss = pcl(specific_representations)
    loss = cla_loss + cl_loss
    loss.backward()
    Optim.step()

if __name__ == '__main__':
    # 配置来源见 YAML 与审查记录；Corel5k 参数不是已核实的作者实验设置。
    parser = argparse.ArgumentParser('')
    parser.add_argument('--dataset', type=str, default='emotions')
    parser.add_argument('--device', type=int, default=0)
    # 保留作者的五折与完整 epoch 默认值，同时支持较小规模的冒烟验证。
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--cpu', action='store_true', help='Explicit CPU validation run')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--data', type=str, default=None, help='Raw Corel5k MAT; cached global-scaled folds are bypassed')
    parser.add_argument('--split-seed', type=int, default=3407)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    with (Path(__file__).resolve().parent / 'config.yaml').open('r', encoding='utf-8') as file:
        params = yaml.safe_load(file)[args.dataset]
    # 使用副本覆盖 epoch，避免直接修改从 YAML 读取的原始配置对象。
    if args.epochs is not None:
        params = dict(params)
        params['epochs'] = args.epochs
    if args.dataset == 'corel5k':
        from corel5k_cv import run_corel5k
        run_corel5k(args, params)
        raise SystemExit(0)
    fix_seed(0)
    all_metrics = []
    k = args.folds
    print('lr=', params['lr'], 'confidence bar=', params['confidence_bar'],
          'threshold=', params['thr'], 'test_threshold=', params['test_thr'], 'w_intra=', params['w_intra'])
    for i in range(k):
        fix_seed(0)
        device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
        print('===================================fold:' + str(i) + '========================================')
        # 训练集与验证集必须使用相同的数据折编号；scalers 字段保留给预处理流程。
        train_dataset = MyDataset_k_fold(params['num_views'], params['name'], i, 'train', device=device, scalers=None)
        val_dataset = MyDataset_k_fold(params['num_views'], params['name'], i, 'val', device=device, scalers=train_dataset.scalers)
        model = HGMVMLC_Model(dataset=train_dataset, params=params, device=device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])
        # 根据本折训练标签预先分组；同一完整标签组合且样本数不少于 2 的样本为正组。
        pcl = PrecomputedContrastiveLoss(train_dataset.labels, params['w_intra'])
        # 每个 epoch 保存 8 个指标，列顺序与 metrics.all_metrics 保持一致。
        fold_metrics = np.zeros((params['epochs'], 8))
        for epoch in tqdm(range(params['epochs'])):
            train_model(dataset=train_dataset, model=model, Optim=optimizer)
            results = test_metrics(model=model, test_features=val_dataset.features,
                                   labels=val_dataset.labels, params=params, mode='val')
            for idx, item in enumerate(results):
                fold_metrics[epoch][idx] = item[1]
        all_metrics.append(fold_metrics)
        del model, optimizer, pcl, train_dataset, val_dataset
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # [fold][epoch, metric] → [epoch, metric, fold]，再沿折维求均值与样本标准差。
    all_metrics = np.stack(all_metrics, axis=-1)
    means = np.mean(all_metrics, axis=2)
    stds = np.std(all_metrics, axis=2, ddof=1)
    for epoch in range(params['epochs']):
        print(f"Epoch {epoch}:")
        print("Means:", " ".join([f"{m:.3f}" for m in means[epoch]]))
        print("Stds :", " ".join([f"{s:.3f}" for s in stds[epoch]]))

    metric_names = ['Hamming Loss', 'Average Precision', 'One-Error', 'Ranking Loss', 'Subset Accuracy', 'Coverage', 'Macro-F1', 'Micro-F1']
    best_ap_epoch = int(np.argmax(means[:, 1]))
    best_mic_epoch = int(np.argmax(means[:, 7]))
    final_epoch = params['epochs'] - 1

    print("\n" + "=" * 75)
    print(f"5-Fold Cross-Validation Summary for Dataset: {params['name']}")
    print("=" * 75)
    col1 = "Best AP (Ep " + str(best_ap_epoch) + ")"
    col2 = "Best Mic (Ep " + str(best_mic_epoch) + ")"
    col3 = "Final (Ep " + str(final_epoch) + ")"
    print(f"{'Metric':<20} {col1:<22} {col2:<22} {col3}")
    print("-" * 75)
    for idx, name in enumerate(metric_names):
        ap_val = f"{means[best_ap_epoch, idx]:.4f} +/- {stds[best_ap_epoch, idx]:.4f}"
        mic_val = f"{means[best_mic_epoch, idx]:.4f} +/- {stds[best_mic_epoch, idx]:.4f}"
        fin_val = f"{means[final_epoch, idx]:.4f} +/- {stds[final_epoch, idx]:.4f}"
        print(f"{name:<20} {ap_val:<22} {mic_val:<22} {fin_val}")
    print("=" * 75)
