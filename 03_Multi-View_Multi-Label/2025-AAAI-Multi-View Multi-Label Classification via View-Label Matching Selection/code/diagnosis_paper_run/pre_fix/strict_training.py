"""修正版 VAMS：训练集拟合预处理/邻居库，验证集选模，测试集最终评估。

默认五折；外层训练部分再留出 20% 验证集。内层训练样本是唯一拟合来源。
这是修正后的归纳协议，不沿用旧版按测试折挑最高分的结果。
"""
import csv
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import KFold, train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import (label_ranking_average_precision_score, label_ranking_loss,
                             hamming_loss, f1_score, coverage_error)
from torch import nn
from torch.nn import functional as F
from paper_gn import _Model
from MLP import MLP

PROTOCOL = 'vams_paper_alignment_train_dev_test_v3'


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_splits(n, folds, seed, dev_fraction):
    # KFold 保留除不尽的尾部样本，每个样本恰好进入一次外层测试折。
    splits = []
    for fold, (outer, test) in enumerate(KFold(folds, shuffle=True, random_state=seed).split(np.arange(n))):
        train, dev = train_test_split(outer, test_size=dev_fraction, random_state=seed + fold)
        splits.append((np.sort(train), np.sort(dev), np.sort(test)))
    return splits


def fit_scalers(views, train_ids):
    # 只看训练行，常数列用 1 作分母；验证/测试允许超出 [0,1]，不重新拟合。
    return [(v[train_ids].amin(0), (v[train_ids].amax(0) - v[train_ids].amin(0)).clamp_min(1e-5))
            for v in views]


def transform(views, ids, scalers, device):
    return [((v[ids] - lo) / span).float().to(device) for v, (lo, span) in zip(views, scalers)]


def label_frequency_loss(scores, labels, frequency, beta):
    # 式 (11)：正样本 exp(beta*(1-pc))，负样本 exp(beta*pc)。
    # 论文印出的 L 缺少负号；最小化时使用负对数似然，按元素均值控制梯度尺度。
    weight = torch.exp(beta * (labels * (1 - frequency) + (1 - labels) * frequency))
    return (weight * F.binary_cross_entropy(scores, labels, reduction='none')).mean()


def knn_indices(reference, query, k, exclude_self=False):
    # 近邻索引是离散选择，不求导；邻居节点特征仍通过当前投影回传梯度。
    if k < 1 or len(reference) < k + int(exclude_self):
        raise ValueError('训练样本数不足以构建指定 k 的邻居库')
    result = []
    for v in range(reference.shape[1]):
        bank = reference[:, v].detach().cpu().numpy()
        q = query[:, v].detach().cpu().numpy()
        ids = NearestNeighbors(n_neighbors=k + int(exclude_self), metric='cosine').fit(bank).kneighbors(q, return_distance=False)
        if exclude_self:
            # 排除自己的真实行号，重复特征时不能盲目丢弃第一名。
            ids = np.stack([row[row != i][:k] for i, row in enumerate(ids)])
        result.append(torch.as_tensor(ids, dtype=torch.long))
    return torch.stack(result)


class VAMS(nn.Module):
    def __init__(self, input_dims, labels, args):
        super().__init__()
        if args.nhidden != 128 or args.target_dim != 128:
            raise ValueError('当前 GN Block 的内部维度为 128，请保持 nhidden/target_dim=128')
        self.input_projections = nn.ModuleList([MLP(d, args.nhidden, args.dropout) for d in input_dims])
        # 标签身份与真实标签值无关；将 one-hot 投影注册为可学习模块。
        self.label_projection = MLP(labels, args.target_dim, args.dropout)
        self.register_buffer('label_identity', torch.eye(labels))
        self.graph = _Model(conv_layers=getattr(args, 'conv_layers', 3))
        self.views = len(input_dims)
        self.labels = labels

    def encode(self, raw_views):
        return torch.stack([p(v) for p, v in zip(self.input_projections, raw_views)], dim=1)

    def predict_batch(self, samples, reference, neighbors, batch_num, args):
        # 构图仅需要标签数，始终传入全零占位；验证/测试真实标签不进入 forward。
        placeholder = samples.new_zeros(len(samples), self.views, self.labels)
        edges, _ = self.graph(samples, self.label_projection(self.label_identity)[None], placeholder,
                              args, neighbors[None], reference[None], batch_num)
        return edges.reshape(len(samples), self.views, self.labels).mean(1)


def metrics(labels, scores, threshold):
    y = labels.detach().cpu().numpy()
    s = scores.detach().cpu().numpy()
    pred = s >= threshold
    return {'LRAP': float(label_ranking_average_precision_score(y, s)),
            'Hamming': float(hamming_loss(y, pred)),
            'Ranking': float(label_ranking_loss(y, s)),
            'Coverage': float(coverage_error(y, s) - 1),
            'Micro_F1': float(f1_score(y, pred, average='micro', zero_division=0)),
            'Macro_F1': float(f1_score(y, pred, average='macro', zero_division=0))}


@torch.no_grad()
def predict(model, train_views, query_views, args):
    model.eval()
    reference, query = model.encode(train_views), model.encode(query_views)
    neighbors = knn_indices(reference, query, args.k)
    return torch.cat([model.predict_batch(query[start:start + args.batch_size], reference, neighbors, b, args)
                      for b, start in enumerate(range(0, len(query), args.batch_size))])


def save_csv(path, rows):
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def run(args, views, labels):
    device = torch.device('cuda' if args.device != 'cpu' and torch.cuda.is_available() else 'cpu')
    if args.device == 'cuda' and device.type != 'cuda':
        raise RuntimeError('请求了 CUDA，但当前环境无法使用 GPU')
    if args.resume:
        raise ValueError('修正版不能载入旧检查点；当前入口只支持从头训练')
    if not (0 < args.dev_fraction < 1) or args.end_epochs < 1 or args.batch_size < 1 or args.beta < 0:
        raise ValueError('请检查 dev_fraction、end_epochs、batch_size 和 beta')
    if getattr(args, 'conv_layers', 3) < 1 or getattr(args, 'weight_decay', 0.0) < 0 or getattr(args, 'norm_regularizer', 0.0) < 0:
        raise ValueError('层数必须大于零，正则系数不能为负')
    torch.set_num_threads(1)
    views = [v.float() for v in views]
    labels = labels.float()
    n = len(labels)
    if any(len(v) != n for v in views) or not torch.isfinite(labels).all() or not all(torch.isfinite(v).all() for v in views):
        raise ValueError('视图样本数不一致或数据包含非有限值')
    if not torch.all((labels == 0) | (labels == 1)):
        raise ValueError('标签必须是 0/1 多标签矩阵')
    selected_fold = getattr(args, 'fold_index', None)
    if selected_fold is not None and not 1 <= selected_fold <= args.k_fold:
        raise ValueError('fold_index 必须在 1 到 k_fold 之间')
    selected_folds = [selected_fold] if selected_fold else list(range(1, args.k_fold + 1))
    root = Path(args.output_dir)
    # 新运行不得覆盖既有结果或旧 Model_Best_GPU；另选输出目录即可重跑。
    root.mkdir(parents=True, exist_ok=True)
    if any(root.iterdir()):
        raise FileExistsError(f'输出目录非空：{root}，请用 --output_dir 指定新目录')
    (root / 'checkpoints').mkdir()
    source = root / 'source'
    source.mkdir()
    hashes = {}
    for name in ('demo.py', 'strict_training.py', 'paper_gn.py', 'paper_graph.py', 'GNBlock_model.py', 'conv_layer.py', 'build_graph.py', 'MLP.py', 'util.py'):
        data = Path(__file__).with_name(name).read_bytes()
        (source / name).write_bytes(data)
        hashes[name] = hashlib.sha256(data).hexdigest()
    data_path = Path(args.data_root) / args.dataset
    manifest = {'protocol': PROTOCOL, 'args': vars(args), 'source_sha256': hashes,
                'data_sha256': hashlib.sha256(data_path.read_bytes()).hexdigest() if data_path.is_file() else None,
                'device': str(device), 'samples': n, 'selection': 'dev_LRAP',
                'outer_folds': args.k_fold, 'selected_folds': selected_folds,
                'test_evaluations_per_fold': 1, 'neighbor_reference': 'inner_train_only',
                'knn_refresh': 'each training epoch and each evaluation',
                'architecture': 'separate_view_neighbor_messages_two_stage_edges_shared_undirected_state',
                'formula_4_label_pooling': 'sum_literal_no_1_over_Q',
                'auxiliary_neighbor_nodes': 'fixed_features_with_trainable_input_projection',
                'status': 'running'}
    (root / 'manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf-8')
    start_time = time.time()
    epoch_rows, fold_rows = [], []
    splits = make_splits(n, args.k_fold, args.seed, args.dev_fraction)
    (root / 'splits.json').write_text(json.dumps([{'fold': i+1, 'train': tr.tolist(), 'dev': dv.tolist(), 'test': te.tolist()}
                                                for i, (tr, dv, te) in enumerate(splits)], indent=2), encoding='utf-8')
    for fold, (tr, dv, te) in enumerate(splits, 1):
        # 只限制执行折数，不改变五折划分、样本比例或每折随机种子。
        if fold not in selected_folds:
            continue
        seed_all(args.seed + fold - 1)
        scalers = fit_scalers(views, tr)
        train_views = transform(views, tr, scalers, device)
        dev_views = transform(views, dv, scalers, device)
        y_train, y_dev = labels[tr].to(device), labels[dv]
        frequency = y_train.mean(0)
        model = VAMS([v.shape[1] for v in views], labels.shape[1], args).to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=.9, weight_decay=getattr(args, 'weight_decay', 0.0))
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=.933)
        best_score = -float('inf')
        checkpoint_path = root / 'checkpoints' / f'fold_{fold}.pth'
        print(f'Fold {fold}/{args.k_fold}: train={len(tr)}, dev={len(dv)}, test={len(te)}, device={device}', flush=True)
        for epoch in range(1, args.end_epochs + 1):
            model.eval()
            with torch.no_grad():
                embedded = model.encode(train_views)
                neighbors = knn_indices(embedded, embedded, args.k, exclude_self=True)
            model.train()
            loss_sum = 0.
            for batch, begin in enumerate(range(0, len(tr), args.batch_size)):
                optimizer.zero_grad(set_to_none=True)
                # 每批重算可求导投影，不能复用跨优化步骤的计算图或 detach 特征。
                reference = model.encode(train_views)
                scores = model.predict_batch(reference[begin:begin + args.batch_size], reference, neighbors, batch, args)
                loss = label_frequency_loss(scores, y_train[begin:begin + args.batch_size], frequency, args.beta)
                # 论文式 (11) 未包含正则项；默认不加作者代码的双重正则。
                reg = getattr(args, 'norm_regularizer', 0.0)
                if reg:
                    loss = loss + reg * sum(p.norm() for p in model.parameters())
                if not torch.isfinite(loss):
                    raise FloatingPointError('损失非有限值')
                loss.backward()
                optimizer.step()
                loss_sum += float(loss.detach()) * len(scores)
            dev_scores = predict(model, train_views, dev_views, args)
            dev_metrics = metrics(y_dev, dev_scores, args.threshold)
            epoch_rows.append({'fold': fold, 'epoch': epoch, 'train_loss': loss_sum / len(tr),
                               **{'dev_' + k: v for k, v in dev_metrics.items()}})
            if dev_metrics['LRAP'] > best_score:
                best_score = dev_metrics['LRAP']
                torch.save({'protocol': PROTOCOL, 'state_dict': model.state_dict(), 'epoch': epoch,
                            'dev_LRAP': best_score, 'scalers': scalers, 'frequency': frequency.cpu(),
                            'split': {'train': tr, 'dev': dv, 'test': te}, 'args': vars(args),
                            'input_dims': [v.shape[1] for v in views], 'labels': labels.shape[1]}, checkpoint_path)
            scheduler.step()
            save_csv(root / 'epoch_metrics.csv', epoch_rows)
            print(f'  epoch {epoch}: loss={loss_sum / len(tr):.4f}, dev_LRAP={dev_metrics["LRAP"]:.6f}', flush=True)
        # 训练/选模全部结束后才创建测试输入、加载验证集选择的模型并测试一次。
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['state_dict'])
        test_views = transform(views, te, scalers, device)
        test_scores = predict(model, train_views, test_views, args)
        test_metrics = metrics(labels[te], test_scores, args.threshold)
        fold_rows.append({'fold': fold, 'selected_epoch': checkpoint['epoch'], 'dev_LRAP': best_score, **test_metrics})
        np.savez_compressed(root / f'fold_{fold}_predictions.npz', sample_ids=te, labels=labels[te].numpy(), predictions=test_scores.cpu().numpy())
        save_csv(root / 'fold_test_metrics.csv', fold_rows)
        print(f'  final test LRAP={test_metrics["LRAP"]:.6f} (dev-selected epoch {checkpoint["epoch"]})', flush=True)
    summary = {'status': 'complete', 'protocol': PROTOCOL, 'seconds': time.time() - start_time,
               'folds': len(fold_rows), 'outer_folds': args.k_fold,
               'selected_folds': selected_folds, 'epochs': args.end_epochs,
               'scope': 'selected_folds' if len(fold_rows) < args.k_fold else 'all_folds',
               'mean': {k: float(np.mean([r[k] for r in fold_rows])) for k in test_metrics},
               'std': {k: (float(np.std([r[k] for r in fold_rows], ddof=1)) if len(fold_rows) > 1 else None) for k in test_metrics}}
    (root / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    manifest['status'] = 'complete'
    (root / 'manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary
