import torch
import random
import argparse
import yaml
from tqdm import tqdm
import numpy as np
import torch.nn as nn
from utils import PrecomputedContrastiveLoss
from metrics import test_metrics
from data_preprocessor import MyDataset_k_fold
from model import HGMVMLC_Model

def fix_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.Generator().manual_seed(seed)

def train_model(dataset, model, Optim):
    fix_seed(0)
    model.train()
    Optim.zero_grad(set_to_none=True)
    features = dataset.features
    labels = dataset.labels
    output, common_representations, specific_representations = model(features)
    fix_seed(0)
    criterion_cla = nn.BCELoss()
    cla_loss = criterion_cla(torch.sigmoid(output), labels.to(torch.float32))
    cl_loss = pcl(specific_representations)
    loss = cla_loss + cl_loss
    loss.backward()
    Optim.step()

if __name__ == '__main__':
    parser = argparse.ArgumentParser('')
    parser.add_argument('--dataset', type=str, default='emotions')
    parser.add_argument('--device', type=int, default=0)
    args = parser.parse_args()
    with open('config.yaml', 'r', encoding='utf-8') as file:
        params = yaml.safe_load(file)[args.dataset]
    fix_seed(0)
    all_metrics = []
    k = 5
    print('lr=', params['lr'], 'confidence bar=', params['confidence_bar'],
          'threshold=', params['thr'], 'test_threshold=', params['test_thr'], 'w_intra=', params['w_intra'])
    for i in range(k):
        fix_seed(0)
        device = torch.device("cuda:0")
        print('===================================fold:' + str(i) + '========================================')
        train_dataset = MyDataset_k_fold(params['num_views'], params['name'], i, 'train', device=device, scalers=None)
        val_dataset = MyDataset_k_fold(params['num_views'], params['name'], i, 'val', device=device, scalers=train_dataset.scalers)
        model = HGMVMLC_Model(dataset=train_dataset, params=params, device=device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])
        pcl = PrecomputedContrastiveLoss(train_dataset.labels, params['w_intra'])
        fold_metrics = np.zeros((params['epochs'], 8))
        for epoch in tqdm(range(params['epochs'])):
            train_model(dataset=train_dataset, model=model, Optim=optimizer)
            results = test_metrics(model=model, test_features=val_dataset.features,
                                   labels=val_dataset.labels, params=params, mode='val')
            for idx, item in enumerate(results):
                fold_metrics[epoch][idx] = item[1]
        all_metrics.append(fold_metrics)
    all_metrics = np.stack(all_metrics, axis=-1)
    means = np.mean(all_metrics, axis=2)
    stds = np.std(all_metrics, axis=2, ddof=1)
    for epoch in range(params['epochs']):
        print(f"Epoch {epoch}:")
        print("Means:", " ".join([f"{m:.3f}" for m in means[epoch]]))
        print("Stds :", " ".join([f"{s:.3f}" for s in stds[epoch]]))