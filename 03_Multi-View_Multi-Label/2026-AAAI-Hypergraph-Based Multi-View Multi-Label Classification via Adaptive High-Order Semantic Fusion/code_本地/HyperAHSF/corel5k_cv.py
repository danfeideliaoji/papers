"""Corel5k protocol repair: fixed epochs, fold-local scaling, saved evidence.

The retained Corel5k hyperparameters are Gemini's history-exposed adaptation,
NOT a verified author configuration. This runner cannot undo past tuning.
"""

import csv
import hashlib
import json
import os
import random
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import numpy as np
import scipy.io
import sklearn
import torch
import torch_geometric
from sklearn.preprocessing import StandardScaler

from metrics import all_metrics
from model import HGMVMLC_Model
from utils import PrecomputedContrastiveLoss

ROOT = Path(__file__).resolve().parent
METRICS = ['HM', 'AP', 'OE', 'RL', 'subset', 'coverage', 'macro', 'micro']


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False),
                    encoding='utf-8')


def load_raw(path, params):
    mat = scipy.io.loadmat(path)
    views = [np.asarray(v) for v in mat['data'].ravel()]
    labels = np.asarray(mat['target'])
    n = views[0].shape[0]
    if labels.shape == (params['num_class'], n):
        labels = labels.T
    if len(views) != params['num_views'] or labels.shape != (n, params['num_class']):
        raise ValueError('Raw view count or label dimensions disagree with config.')
    if not np.isin(labels, [0, 1]).all():
        raise ValueError('Expected binary labels; do not silently binarize.')
    for view, width in zip(views, params['inputs_sizes']):
        if view.shape != (n, width) or not np.isfinite(view).all():
            raise ValueError('Invalid raw feature shape or nonfinite feature values.')
    return views, labels.astype(np.int64, copy=False)


def five_fold_indices(n, seed=3407):
    ids = np.random.RandomState(seed).permutation(n)
    folds = np.array_split(ids, 5)
    return [(np.concatenate(folds[:i] + folds[i + 1:]), folds[i]) for i in range(5)]


def scale_fold(views, train_ids, test_ids):
    train, test, scalers = [], [], []
    for view in views:
        scaler = StandardScaler().fit(view[train_ids])
        train.append(scaler.transform(view[train_ids]).astype(np.float32))
        test.append(scaler.transform(view[test_ids]).astype(np.float32))
        scalers.append(scaler)
    return train, test, scalers


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def independent_ap_micro(logits, labels, predictions):
    """Direct float64 ranking counts, including sklearn's empty-label convention."""
    ap_rows = []
    for scores, y in zip(np.asarray(logits, dtype=np.float64), labels):
        positive = np.flatnonzero(y)
        if positive.size == 0 or positive.size == y.size:
            ap_rows.append(1.0)
            continue
        precisions = []
        for c in positive:
            above = scores >= scores[c]
            precisions.append(np.count_nonzero(above & (y == 1)) / np.count_nonzero(above))
        ap_rows.append(float(np.mean(precisions)))
    tp = np.count_nonzero((labels == 1) & (predictions == 1))
    denominator = np.count_nonzero(labels) + np.count_nonzero(predictions)
    return float(np.mean(ap_rows)), float(2 * tp / denominator) if denominator else 0.0


def run_corel5k(args, params):
    if not 1 <= args.folds <= 5 or params['epochs'] < 1:
        raise ValueError('--folds must be 1..5 and epochs must be positive.')
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    torch.set_num_threads(args.threads)
    device = torch.device('cpu' if args.cpu else f'cuda:{args.device}')
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable. Use --cpu explicitly for a CPU smoke run.')
    if device.type == 'cuda':
        torch.cuda.set_device(device)
    output_base = Path(params.get('artifact_root', ROOT / 'research_outputs'))
    output = Path(args.output) if args.output else output_base / (
        'corel5k_corrected_' + datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M%S'))
    output.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(output.parent).free < args.folds * 100 * 1024**2:
        raise OSError('Insufficient output disk space; choose a free volume with --output.')
    output.mkdir(parents=True, exist_ok=False)
    data_path = Path(args.data) if args.data else ROOT / 'Multi_View_dataset' / 'corel5k.mat'
    started = time.perf_counter()
    manifest = dict(status='preflight', dataset='corel5k', params=params,
                    config_provenance='Gemini local adaptation; author Corel5k config unverified',
                    claim_status='EXPLORATORY_HISTORY_EXPOSED_NOT_PAPER_REPRODUCTION',
                    protocol='five outer folds; train-only StandardScaler; fixed final epoch; no test selection',
                    split_seed=args.split_seed, model_seed=args.seed, folds_executed=args.folds,
                    epoch_selection=f"fixed_final_epoch_{params['epochs']}",
                    test_threshold_selection='fixed config; not scanned in this run',
                    data_path=str(data_path.resolve()), data_sha256=sha256(data_path),
                    code_sha256={name: sha256(ROOT / name) for name in
                                 ['train.py', 'corel5k_cv.py', 'eval_5folds.py', 'model.py',
                                  'utils.py', 'metrics.py', 'config.yaml',
                                  'experiments/hgnn_ablation/tensor_ops.py']},
                    environment=dict(python=sys.version, executable=sys.executable,
                                     torch=torch.__version__, sklearn=sklearn.__version__,
                                     numpy=np.__version__, scipy=scipy.__version__,
                                     torch_geometric=torch_geometric.__version__,
                                     device=str(device), threads=args.threads,
                                     gpu=torch.cuda.get_device_name(device) if device.type == 'cuda' else None),
                    inherited_boundaries=['source HGNN edge readout is mean; paper Eq8 describes sum',
                                          'source contrastive prototype averages normalized rows',
                                          'source BCE reduction is mean',
                                          'test_thr=99 leaves only self loops within each view'])
    write_json(output / 'manifest.json', manifest)
    try:
        views, labels = load_raw(data_path, params)
        manifest.update(samples=len(labels), num_labels=labels.shape[1],
                        positive_labels=int(labels.sum()), zero_label_rows=int((labels.sum(1) == 0).sum()),
                        empty_label_AP_convention='AP=1 as in sklearn/source implementation')
        full_corel_data = (len(labels) == 4999 and labels.shape[1] == 260 and
                          [x.shape[1] for x in views] == [100, 1000, 512, 4096, 4096, 4096])
        manifest['full_corel5k_shape'] = full_corel_data
        if not full_corel_data:
            print('SMOKE/FIXTURE: input is not the complete 4999-sample Corel5k dataset.', flush=True)
        splits = five_fold_indices(len(labels), args.split_seed)
        np.savez_compressed(output / 'splits.npz', **{
            f'fold{i}_{part}': ids for i, split in enumerate(splits)
            for part, ids in zip(['train', 'test'], split)})
        manifest['split_sha256'] = sha256(output / 'splits.npz')
        manifest['status'] = 'running'
        write_json(output / 'manifest.json', manifest)  # Persist rules/IDs before any metric access.
        print(f'Corel5k: {args.folds}/5 fixed outer folds, {params["epochs"]} epochs, {device}', flush=True)
        print('EXPLORATORY: retained settings were previously tuned on outer folds.', flush=True)
        print(f'Artifacts: {output.resolve()}', flush=True)
        records = []
        for fold, (train_ids, test_ids) in enumerate(splits[:args.folds]):
            fold_dir = output / f'fold{fold}'
            fold_dir.mkdir()
            train_x, test_x, scalers = scale_fold(views, train_ids, test_ids)
            np.savez_compressed(fold_dir / 'scalers.npz', **{
                f'view{v}_{key}': getattr(scaler, key + '_') for v, scaler in enumerate(scalers)
                for key in ['mean', 'scale', 'var']})
            dataset = SimpleNamespace(features=[torch.as_tensor(x, device=device) for x in train_x],
                                      labels=torch.as_tensor(labels[train_ids], device=device),
                                      num_views=params['num_views'])
            del train_x
            seed_everything(args.seed)  # Once per fresh fold, never reset inside epochs.
            model = HGMVMLC_Model(dataset, params, device)
            initial_activation = float(model.common_HGNN.act.weight.detach().cpu().item())
            if initial_activation != 0.25:
                raise AssertionError('Fresh-fold PReLU initialization was contaminated.')
            optimizer = torch.optim.AdamW(model.parameters(), lr=params['lr'],
                                          weight_decay=params['weight_decay'])
            pcl = PrecomputedContrastiveLoss(dataset.labels, params['w_intra']).to(device)
            criterion = torch.nn.BCELoss()  # Keep source objective/reduction; no new weighting.
            torch.save(model.state_dict(), fold_dir / 'initial_state.pt')
            initial_state_sha = sha256(fold_dir / 'initial_state.pt')
            if records and initial_state_sha != records[0]['initial_state_sha256']:
                raise AssertionError('Same seed did not produce identical initial parameters across folds.')
            fold_started = time.perf_counter()
            with (fold_dir / 'training.csv').open('w', encoding='utf-8', newline='') as handle:
                writer = csv.writer(handle)
                writer.writerow(['epoch', 'loss', 'classification', 'contrastive'])
                for epoch in range(1, params['epochs'] + 1):
                    model.train()
                    optimizer.zero_grad(set_to_none=True)
                    logits, _, cross_view_repr = model(dataset.features)
                    bce = criterion(torch.sigmoid(logits), dataset.labels.float())
                    contrastive = pcl(cross_view_repr)
                    loss = bce + contrastive
                    if not torch.isfinite(loss):
                        raise FloatingPointError(f'Nonfinite training loss: fold {fold}, epoch {epoch}')
                    loss.backward()
                    optimizer.step()
                    writer.writerow([epoch, loss.item(), bce.item(), contrastive.item()])
                    handle.flush()
                    if epoch == 1 or epoch % 10 == 0 or epoch == params['epochs']:
                        print(f'fold {fold} epoch {epoch}: train loss={loss.item():.6f}', flush=True)
            # No held-out inference, checkpoint selection or threshold scan during training.
            model.eval()
            with torch.no_grad():
                test_features = [torch.as_tensor(x, device=device) for x in test_x]
                out, _, _ = model.evaluate(test_features)
                probabilities = torch.sigmoid(out).cpu().numpy()
                logits_np = out.cpu().numpy()
            if not np.isfinite(logits_np).all():
                raise FloatingPointError(f'Nonfinite predictions in fold {fold}')
            predictions = (probabilities > params['confidence_bar']).astype(np.int64)
            truth = labels[test_ids]
            metrics = dict(all_metrics(logits_np, predictions, truth))
            ap, micro = independent_ap_micro(logits_np, truth, predictions)
            np.savez_compressed(fold_dir / 'predictions.npz', sample_ids=test_ids,
                                logits=logits_np, probabilities=probabilities,
                                labels=truth, predictions=predictions)
            np.testing.assert_allclose([metrics['AP'], metrics['micro']], [ap, micro], rtol=0, atol=1e-12)
            torch.save(dict(state_dict=model.state_dict(), params=params, epoch=params['epochs'],
                            train_graph=[g.cpu() for g in model.View_hyperedge_index],
                            common_train_graph=model.common_hyperedge_index.cpu()),
                       fold_dir / 'final_checkpoint.pt')
            record = dict(fold=fold, metrics=metrics, independent_AP=ap, independent_MicroF1=micro,
                          train_samples=len(train_ids), test_samples=len(test_ids),
                          activation_initial_value=initial_activation,
                          initial_state_sha256=initial_state_sha,
                          elapsed_seconds=time.perf_counter() - fold_started,
                          peak_cuda_memory_bytes=torch.cuda.max_memory_allocated(device)
                          if device.type == 'cuda' else None)
            write_json(fold_dir / 'result.json', record)
            records.append(record)
            print(f'fold {fold}: AP={ap:.6f}, Micro-F1={micro:.6f}', flush=True)
            del model, optimizer, pcl, dataset, test_features, out, logits, loss, bce, contrastive
            if device.type == 'cuda':
                torch.cuda.empty_cache()
        values = np.array([[r['metrics'][key] for key in METRICS] for r in records])
        summary = dict(status='complete', folds_executed=len(records), total_outer_folds=5,
                       claim_status=manifest['claim_status'],
                       validation_scope='five_fold_fixed_epoch' if args.folds == 5 and full_corel_data
                       else 'partial_fold_or_subset_smoke',
                       epochs_per_fold=params['epochs'], full_corel5k_shape=full_corel_data,
                       means=dict(zip(METRICS, values.mean(axis=0).tolist())),
                       sample_stds=dict(zip(METRICS, values.std(axis=0, ddof=1).tolist()))
                       if len(records) > 1 else None, records=records,
                       wall_seconds=time.perf_counter() - started)
        write_json(output / 'summary.json', summary)
        manifest['status'] = 'complete'
        write_json(output / 'manifest.json', manifest)
        print(json.dumps(summary['means'], ensure_ascii=False), flush=True)
        print('Sample SD (ddof=1):', summary['sample_stds'], flush=True)
        return output
    except Exception as error:
        manifest.update(status='failed', error=f'{type(error).__name__}: {error}')
        write_json(output / 'manifest.json', manifest)
        raise
