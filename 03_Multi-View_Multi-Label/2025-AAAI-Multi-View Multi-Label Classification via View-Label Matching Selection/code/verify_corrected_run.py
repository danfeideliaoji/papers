"""用运行时源码快照重放修正版结果，核对验证选模和最终测试预测。"""
import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('output_dir')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    cli = parser.parse_args()
    root = Path(cli.output_dir).resolve()
    manifest = json.loads((root / 'manifest.json').read_text(encoding='utf-8'))
    assert manifest['status'] == 'complete'
    for name, expected in manifest['source_sha256'].items():
        assert hashlib.sha256((root / 'source' / name).read_bytes()).hexdigest() == expected
    sys.path.insert(0, str(root / 'source'))
    from strict_training import VAMS, predict, transform, fit_scalers, metrics, make_splits, PROTOCOL
    import demo
    args = SimpleNamespace(**manifest['args'])
    data_path = Path(args.data_root) / args.dataset
    if manifest['data_sha256'] is not None:
        assert hashlib.sha256(data_path.read_bytes()).hexdigest() == manifest['data_sha256']
    demo.args = args
    views, labels = demo.data_loader(str(Path(args.data_root) / args.dataset))
    labels = torch.as_tensor(labels).float()
    if labels.shape[0] != len(views[0]):
        labels = labels.T
    views = [v.float() for v in views]
    device = torch.device(cli.device)
    torch.set_num_threads(1)
    with (root / 'epoch_metrics.csv').open(encoding='utf-8-sig') as f:
        epochs = list(csv.DictReader(f))
    with (root / 'fold_test_metrics.csv').open(encoding='utf-8-sig') as f:
        folds = list(csv.DictReader(f))
    selected_folds = manifest.get('selected_folds', list(range(1, args.k_fold + 1)))
    # v3 向量化 CUDA index_add 的并行累加存在 FP32 舍入波动。
    # 已用重复 GPU / CPU 重放定位到约 1e-6 至 3e-6；放宽概率误差时，
    # 仍要求标签排序、阈值分类完全一致，并逐项核对全部指标。
    aligned = manifest['protocol'] in ('vams_paper_alignment_train_dev_test_v3','vams_paper_alignment_train_dev_test_v4',
                                       'vams_paper_alignment_train_dev_test_v5_deterministic')
    prediction_atol = 5e-6 if aligned else 1e-6
    replay_errors = []
    assert [int(row['fold']) for row in folds] == selected_folds
    assert len(epochs) == len(selected_folds) * args.end_epochs
    expected_splits = make_splits(len(labels), args.k_fold, args.seed, args.dev_fraction)
    all_test_ids = []
    for row in folds:
        fold = int(row['fold'])
        checkpoint = torch.load(root / 'checkpoints' / f'fold_{fold}.pth', map_location='cpu', weights_only=False)
        assert checkpoint['protocol'] == PROTOCOL
        tr, dv, te = (checkpoint['split'][k] for k in ('train', 'dev', 'test'))
        for actual, expected in zip((tr, dv, te), expected_splits[fold - 1]):
            np.testing.assert_array_equal(actual, expected)
        assert [int(r['epoch']) for r in epochs if int(r['fold']) == fold] == list(range(1, args.end_epochs + 1))
        all_test_ids.extend(te.tolist())
        assert not (set(tr) & set(dv) or set(tr) & set(te) or set(dv) & set(te))
        selection = [r for r in epochs if int(r['fold']) == fold]
        best = max(selection, key=lambda r: float(r['dev_LRAP']))
        assert int(best['epoch']) == checkpoint['epoch'] == int(row['selected_epoch'])
        assert abs(float(best['dev_LRAP']) - checkpoint['dev_LRAP']) < 1e-7
        scalers = checkpoint['scalers']
        torch.testing.assert_close(checkpoint['frequency'], labels[tr].mean(0))
        for expected, actual in zip(fit_scalers(views, tr), scalers):
            for a, b in zip(expected, actual):
                torch.testing.assert_close(a, b)
        model = VAMS(checkpoint['input_dims'], checkpoint['labels'], args).to(device)
        model.load_state_dict(checkpoint['state_dict'])
        train_views = transform(views, tr, scalers, device)
        scores = predict(model, train_views, transform(views, te, scalers, device), args)
        with np.load(root / f'fold_{fold}_predictions.npz') as saved:
            np.testing.assert_array_equal(te, saved['sample_ids'])
            np.testing.assert_array_equal(labels[te].numpy(), saved['labels'])
            actual = scores.cpu().numpy()
            expected = saved['predictions']
            np.testing.assert_allclose(actual, expected, atol=prediction_atol, rtol=1e-6)
            if aligned:
                np.testing.assert_array_equal(actual >= args.threshold, expected >= args.threshold)
                np.testing.assert_array_equal(np.argsort(actual, axis=1), np.argsort(expected, axis=1))
            replay_errors.append(float(np.abs(actual - expected).max()))
        for name, value in metrics(labels[te], scores, args.threshold).items():
            assert abs(value - float(row[name])) < 1e-6
        dev_scores = predict(model, train_views, transform(views, dv, scalers, device), args)
        assert abs(metrics(labels[dv], dev_scores, args.threshold)['LRAP'] - checkpoint['dev_LRAP']) < 1e-6
    expected_test_ids = sorted(i for f in selected_folds for i in expected_splits[f - 1][2])
    assert sorted(all_test_ids) == expected_test_ids
    report = {'status': 'PASS', 'folds': len(folds), 'outer_folds': args.k_fold,
              'selected_folds': selected_folds, 'epochs': len(epochs),
              'test_rows': len(all_test_ids), 'source_snapshot_hashes': 'verified',
              'train_only_scalers': 'verified', 'dev_selection_and_scores': 'replayed',
              'test_predictions_and_metrics': 'replayed',
              'prediction_atol': prediction_atol, 'prediction_rtol': 1e-6,
              'max_prediction_abs_error': max(replay_errors),
              'threshold_and_label_order': 'identical' if aligned else 'covered_by_metrics'}
    (root / 'verification.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
