"""Scene 差距诊断：只用内层训练/验证数据，不产生新测试成绩。"""
import csv
import hashlib
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch

BASE = Path('results_scene_v4_5fold_80epochs')
OUT = Path('diagnosis_scene_gap')


def save(name, obj):
    (OUT / name).write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding='utf-8')


def main():
    OUT.mkdir(exist_ok=False)
    manifest = json.loads((BASE / 'manifest.json').read_text(encoding='utf-8'))
    for name, expected in manifest['source_sha256'].items():
        assert hashlib.sha256((BASE / 'source' / name).read_bytes()).hexdigest() == expected
    args = SimpleNamespace(**manifest['args'])
    assert hashlib.sha256((Path(args.data_root) / args.dataset).read_bytes()).hexdigest() == manifest['data_sha256']
    sys.path.insert(0, str((BASE / 'source').resolve()))
    from strict_training import VAMS, transform, predict, knn_indices, metrics, seed_all, label_frequency_loss
    import demo
    demo.args = args
    raw, y = demo.data_loader(str(Path(args.data_root) / args.dataset))
    raw = [v.float() for v in raw]
    y = torch.as_tensor(y).T.float()
    torch.set_num_threads(1)
    device = 'cuda'
    with (BASE / 'epoch_metrics.csv').open(encoding='utf-8-sig') as f:
        curves = list(csv.DictReader(f))
    probes = []
    for fold in range(1, 6):
        cp = torch.load(BASE / f'checkpoints/fold_{fold}.pth', map_location='cpu', weights_only=False)
        tr, dv = cp['split']['train'], cp['split']['dev']
        tv, vv = transform(raw, tr, cp['scalers'], device), transform(raw, dv, cp['scalers'], device)
        model = VAMS(cp['input_dims'], cp['labels'], args).to(device)
        model.load_state_dict(cp['state_dict'])
        model.eval()
        with torch.no_grad():
            train_emb, dev_emb = model.encode(tv), model.encode(vv)
            train_nn = knn_indices(train_emb, train_emb, args.k, exclude_self=True)
            def from_emb(query, bank, nn):
                return torch.cat([model.predict_batch(query[s:s+args.batch_size], bank, nn, b, args)
                                  for b, s in enumerate(range(0, len(query), args.batch_size))])
            ts = from_emb(train_emb, train_emb, train_nn)
            vs = predict(model, tv, vv, args)
            # 只用验证集特征的邻居库：检查旧代码与修正版的参考库差异，不使用测试集。
            dn = knn_indices(dev_emb, dev_emb, args.k, exclude_self=True)
            ds = from_emb(dev_emb, dev_emb, dn)
            ym = y[dv]
            candidates = [{'threshold': float(t), 'Micro_F1': metrics(ym, vs, float(t))['Micro_F1']}
                          for t in np.arange(.1, .901, .05)]
            seq = [r for r in curves if int(r['fold']) == fold]
            probes.append({'fold': fold, 'selected_epoch': cp['epoch'],
                           'train_no_self': metrics(y[tr], ts, args.threshold),
                           'dev_train_bank': metrics(ym, vs, args.threshold),
                           'dev_query_bank_no_self': metrics(ym, ds, args.threshold),
                           'train_weighted_bce': float(label_frequency_loss(ts, y[tr].to(device), cp['frequency'].to(device), args.beta)),
                           'dev_weighted_bce': float(label_frequency_loss(vs, ym.to(device), cp['frequency'].to(device), args.beta)),
                           'dev_threshold_probe': {'fixed_0_5': metrics(ym, vs, .5)['Micro_F1'],
                                                   'best_on_grid': max(candidates, key=lambda x: x['Micro_F1']),
                                                   'scope': 'descriptive_validation_only_no_test'},
                           'dev_LRAP_epoch40': float(seq[39]['dev_LRAP']),
                           'dev_LRAP_epoch60': float(seq[59]['dev_LRAP']),
                           'dev_LRAP_epoch80': float(seq[79]['dev_LRAP'])})
        del model
    save('checkpoint_probes.json', {'source_verified': True, 'test_evaluations': 0, 'folds': probes,
                                   'learning_rate_epoch80': args.lr*.933**79,
                                   'parameters': sum(p.numel() for p in VAMS([v.shape[1] for v in raw], y.shape[1], args).parameters()),
                                   'empty_label_rows': int((y.sum(1) == 0).sum())})
    print(json.dumps(probes, ensure_ascii=False, indent=2), flush=True)
    # 在实验开始前冻结唯一干预、折、预算和停止条件；不搜索其他衰减系数。
    plan = {'hypothesis': 'rapid_lr_decay_limits_training_progress', 'fold': 1, 'epochs': 80,
            'intervention': 'only_gamma_0.933_to_0.99', 'baseline': str(BASE),
            'unchanged': 'source model, split, seed, optimizer, learning rate, losses, order, dimensions, k, threshold',
            'selection_metric': 'dev_LRAP', 'test_evaluations': 0,
            'hard_stop': 'one_fold_one_intervention_80epochs_no_rescue_tuning',
            'baseline_dev_LRAP': probes[0]['dev_train_bank']['LRAP'],
            'source_sha256': manifest['source_sha256'], 'data_sha256': manifest['data_sha256']}
    save('schedule_plan.json', plan)
    cp = torch.load(BASE / 'checkpoints/fold_1.pth', map_location='cpu', weights_only=False)
    tr, dv = cp['split']['train'], cp['split']['dev']
    tv, vv = transform(raw, tr, cp['scalers'], device), transform(raw, dv, cp['scalers'], device)
    seed_all(args.seed)
    model = VAMS(cp['input_dims'], cp['labels'], args).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=.9, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=.99)
    freq = cp['frequency'].to(device)
    yt = y[tr].to(device)
    rows = []
    start = time.time()
    for epoch in range(1, 81):
        model.eval()
        with torch.no_grad():
            emb = model.encode(tv)
            nn = knn_indices(emb, emb, args.k, exclude_self=True)
        model.train()
        loss_sum = 0.
        for batch, s in enumerate(range(0, len(tr), args.batch_size)):
            optimizer.zero_grad(set_to_none=True)
            ref = model.encode(tv)
            pred = model.predict_batch(ref[s:s+args.batch_size], ref, nn, batch, args)
            loss = label_frequency_loss(pred, yt[s:s+args.batch_size], freq, args.beta)
            if not torch.isfinite(loss):
                raise FloatingPointError('诊断损失非有限值；按预定规则停止，不调参')
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach())*len(pred)
        score = metrics(y[dv], predict(model, tv, vv, args), args.threshold)
        rows.append({'epoch': epoch, 'train_loss': loss_sum/len(tr), 'lr': optimizer.param_groups[0]['lr'],
                     **{'dev_'+k: v for k, v in score.items()}})
        with (OUT / 'schedule_epoch_metrics.csv').open('w', encoding='utf-8-sig', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        scheduler.step()
        print(f'diagnostic epoch {epoch}: loss={loss_sum/len(tr):.4f}, dev_LRAP={score["LRAP"]:.6f}', flush=True)
    best = max(rows, key=lambda x: x['dev_LRAP'])
    save('schedule_result.json', {'status': 'complete', 'test_evaluations': 0, 'epochs': 80,
                                 'seconds': time.time()-start, 'best': best, 'last': rows[-1],
                                 'baseline_dev_LRAP': plan['baseline_dev_LRAP'],
                                 'dev_LRAP_difference': best['dev_LRAP']-plan['baseline_dev_LRAP'],
                                 'scope': 'single_fold_training_validation_diagnostic_not_a_reproduction_score'})
    print(json.dumps(best), flush=True)


if __name__ == '__main__':
    main()
