"""固定的小范围优化实验：每折内部验证选择 SGD/Adam，测试不参与选择。"""
from runtime_config import configure_determinism, runtime_metadata
import argparse
import hashlib
import json
import shutil
import time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
import demo
from strict_training import (VAMS, make_splits, fit_scalers, transform, seed_all,
                             knn_indices, label_frequency_loss, predict, metrics, save_csv)

PROTOCOL = 'vams_v5_deterministic_inner_validation_optimizer_selection'
GRID = ['sgd', 'adam']
SOURCE_FILES = ['deterministic_search.py', 'runtime_config.py', 'demo.py', 'strict_training.py',
                'paper_gn.py', 'paper_graph.py', 'GNBlock_model.py', 'conv_layer.py',
                'build_graph.py', 'MLP.py', 'util.py']


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')


def state_hash(model):
    h = hashlib.sha256()
    for name, value in model.state_dict().items():
        h.update(name.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output_dir', required=True)
    cli = p.parse_args()
    # 同一预先固定网格应用到每折：不从第一折选全局配置，以免跨折传递标签信息。
    baseline = Path(__file__).resolve().parent / 'search_base_args.json'
    args = SimpleNamespace(**json.loads(baseline.read_text(encoding='utf-8')))
    args.output_dir = cli.output_dir
    args.end_epochs = 120
    args.fold_index = None
    assert args.dataset == 'scene.mat' and args.k_fold == 5
    root = Path(cli.output_dir)
    root.mkdir(exist_ok=False)
    (root/'source').mkdir()
    (root/'checkpoints').mkdir()
    (root/'candidate_checkpoints').mkdir()
    hashes = {}
    for name in SOURCE_FILES + ['search_base_args.json']:
        content = Path(__file__).with_name(name).read_bytes()
        (root/'source'/name).write_bytes(content)
        hashes[name] = hashlib.sha256(content).hexdigest()
    configure_determinism()
    device = torch.device('cuda')
    if not torch.cuda.is_available():
        raise RuntimeError('本次已固定 CUDA，请检查原 GPU 环境')
    demo.args = args
    raw, labels = demo.data_loader(str(Path(args.data_root)/args.dataset))
    raw = [v.float() for v in raw]
    labels = torch.as_tensor(labels).float()
    if labels.shape[0] != len(raw[0]): labels = labels.T
    assert all(len(v) == len(labels) and torch.isfinite(v).all() for v in raw)
    assert torch.all((labels == 0) | (labels == 1))
    manifest = {'protocol': PROTOCOL, 'status': 'running', 'args': vars(args),
                'source_sha256': hashes, 'data_sha256': hashlib.sha256((Path(args.data_root)/args.dataset).read_bytes()).hexdigest(),
                'runtime': runtime_metadata(), 'samples': len(labels), 'outer_folds': 5,
                'selected_folds': [1, 2, 3, 4, 5], 'candidate_grid': GRID,
                'selection': 'inner_dev_LRAP_candidate_and_epoch',
                'threshold': .5, 'gamma': .933, 'minimum_epochs': 80, 'maximum_epochs': 120,
                'patience': 20, 'test_evaluations_per_fold': 1,
                'hard_stop': 'only_two_predeclared_optimizers_no_grid_expansion_after_test',
                'claim_scope': 'exploratory_optimization_on_previously_evaluated_dataset_not_an_independent_confirmation'}
    write_json(root/'manifest.json', manifest)
    splits = make_splits(len(labels), args.k_fold, args.seed, args.dev_fraction)
    write_json(root/'splits.json', [{'fold': i+1, 'train': tr.tolist(), 'dev': dv.tolist(), 'test': te.tolist()}
                                  for i, (tr, dv, te) in enumerate(splits)])
    start_time = time.time()
    epochs, trials, tests = [], [], []
    for fold, (tr, dv, te) in enumerate(splits, 1):
        scalers = fit_scalers(raw, tr)
        tv, vv = transform(raw, tr, scalers, device), transform(raw, dv, scalers, device)
        yt = labels[tr].to(device)
        frequency = yt.mean(0)
        first_initial_hash = None
        for candidate in GRID:
            seed_all(args.seed + fold - 1)
            model = VAMS([v.shape[1] for v in raw], labels.shape[1], args).to(device)
            initial_hash = state_hash(model)
            if first_initial_hash is None: first_initial_hash = initial_hash
            assert initial_hash == first_initial_hash, '两种优化器必须从相同初始权重开始'
            if candidate == 'sgd':
                optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=.9, weight_decay=0)
            else:
                optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0)
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=.933)
            best, best_epoch = -float('inf'), 0
            checkpoint = root/'candidate_checkpoints'/f'fold_{fold}_{candidate}.pth'
            print(f'Fold {fold}/5 {candidate}: train={len(tr)}, dev={len(dv)}, deterministic CUDA', flush=True)
            for epoch in range(1, 121):
                model.eval()
                with torch.no_grad():
                    emb = model.encode(tv)
                    neighbors = knn_indices(emb, emb, args.k, exclude_self=True)
                model.train()
                loss_sum = 0.
                for batch, s in enumerate(range(0, len(tr), args.batch_size)):
                    optimizer.zero_grad(set_to_none=True)
                    ref = model.encode(tv)
                    scores = model.predict_batch(ref[s:s+args.batch_size], ref, neighbors, batch, args)
                    loss = label_frequency_loss(scores, yt[s:s+args.batch_size], frequency, args.beta)
                    if not torch.isfinite(loss): raise FloatingPointError('损失非有限，停止而不临时改变网格')
                    loss.backward()
                    optimizer.step()
                    loss_sum += float(loss.detach())*len(scores)
                dev_scores = predict(model, tv, vv, args)
                dm = metrics(labels[dv], dev_scores, args.threshold)
                epochs.append({'fold': fold, 'candidate': candidate, 'epoch': epoch,
                               'lr': optimizer.param_groups[0]['lr'], 'train_loss': loss_sum/len(tr),
                               **{'dev_'+k: v for k, v in dm.items()}})
                if dm['LRAP'] > best:
                    best, best_epoch = dm['LRAP'], epoch
                    torch.save({'protocol': PROTOCOL, 'state_dict': model.state_dict(), 'epoch': epoch,
                                'candidate': candidate, 'dev_LRAP': best, 'dev_metrics': dm,
                                'scalers': scalers, 'frequency': frequency.cpu(),
                                'split': {'train': tr, 'dev': dv, 'test': te}, 'args': vars(args),
                                'input_dims': [v.shape[1] for v in raw], 'labels': labels.shape[1],
                                'initial_state_sha256': initial_hash}, checkpoint)
                    np.savez_compressed(root/f'fold_{fold}_{candidate}_dev_predictions.npz',
                                        sample_ids=dv, labels=labels[dv].numpy(), predictions=dev_scores.cpu().numpy())
                scheduler.step()
                save_csv(root/'epoch_metrics.csv', epochs)
                if epoch == 1 or epoch % 5 == 0:
                    print(f'  {candidate} epoch {epoch}: loss={loss_sum/len(tr):.4f}, dev_LRAP={dm["LRAP"]:.6f}, best={best:.6f}', flush=True)
                if epoch >= 80 and epoch - best_epoch >= 20: break
            trials.append({'fold': fold, 'candidate': candidate, 'epochs_run': epoch,
                           'selected_epoch': best_epoch, 'dev_LRAP': best,
                           'initial_state_sha256': initial_hash})
            save_csv(root/'candidate_metrics.csv', trials)
            del model, optimizer, scheduler
        fold_trials = [t for t in trials if t['fold'] == fold]
        winner = max(fold_trials, key=lambda t: t['dev_LRAP'])
        candidate = winner['candidate']
        cp_path = root/'candidate_checkpoints'/f'fold_{fold}_{candidate}.pth'
        shutil.copy2(cp_path, root/'checkpoints'/f'fold_{fold}.pth')
        cp = torch.load(cp_path, map_location=device, weights_only=False)
        model = VAMS(cp['input_dims'], cp['labels'], args).to(device)
        model.load_state_dict(cp['state_dict'])
        # 所有候选训练及内部选模完成后，才构造并计算本折测试预测。
        ts = predict(model, tv, transform(raw, te, scalers, device), args)
        tm = metrics(labels[te], ts, args.threshold)
        tests.append({'fold': fold, 'selected_candidate': candidate, 'selected_epoch': cp['epoch'],
                      'dev_LRAP': cp['dev_LRAP'], **tm})
        np.savez_compressed(root/f'fold_{fold}_predictions.npz', sample_ids=te,
                            labels=labels[te].numpy(), predictions=ts.cpu().numpy())
        save_csv(root/'fold_test_metrics.csv', tests)
        print(f'Fold {fold}: selected {candidate}, epoch {cp["epoch"]}; final test LRAP={tm["LRAP"]:.6f}', flush=True)
        del model
    result = {'status': 'complete', 'protocol': PROTOCOL, 'folds': 5, 'candidates_per_fold': 2,
              'total_epochs': len(epochs), 'seconds': time.time()-start_time,
              'mean': {k: float(np.mean([t[k] for t in tests])) for k in tm},
              'std': {k: float(np.std([t[k] for t in tests], ddof=1)) for k in tm},
              'selected_candidates': [t['selected_candidate'] for t in tests]}
    write_json(root/'summary.json', result)
    manifest['status'] = 'complete'
    write_json(root/'manifest.json', manifest)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__': main()
