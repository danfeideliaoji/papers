"""在完整训练的源码快照上诊断预测塌缩；只读取训练/验证行，不评估测试集。"""
import argparse
import copy
import csv
import hashlib
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch


def variation(tensor, axis):
    return float(tensor.detach().float().std(dim=axis, unbiased=False).square().mean().sqrt())


def inspect(model, raw, y, frequency, args, label):
    from strict_training import knn_indices, label_frequency_loss
    model.eval()
    reference = model.encode(raw)
    neighbors = knn_indices(reference, reference, args.k, True)
    b = min(args.batch_size, len(y))
    stages, handles = {}, []

    def cross(name):
        def hook(module, inp, out):
            tensor = out.reshape(b, model.views, model.labels, -1)
            stages[name] = {'label_rms_std': variation(tensor, 2), 'sample_rms_std': variation(tensor, 0),
                            'rms': float(tensor.detach().square().mean().sqrt())}
        return hook

    handles.append(model.graph.red_edge_cross.register_forward_hook(cross('initial_cross_edges')))
    for i, layer in enumerate((model.graph.pro_layer1, model.graph.pro_layer2), 1):
        handles.append(layer.edge_model.edge_mlp_cross.register_forward_hook(cross(f'layer_{i}_cross_edges')))
        def node_hook(module, inp, out, index=i):
            labels = out[1][0].reshape(b, model.labels, -1)
            stages[f'layer_{index}_label_nodes'] = {'label_rms_std': variation(labels, 1),
                                                   'sample_rms_std': variation(labels, 0)}
        handles.append(layer.node_model.register_forward_hook(node_hook))
    handles.append(model.graph.decoder_edge_cross[3].register_forward_hook(cross('decoder_logits')))
    scores = model.predict_batch(reference[:b], reference, neighbors, 0, args)
    for handle in handles:
        handle.remove()
    task = label_frequency_loss(scores, y[:b], frequency, args.beta)
    regularizer = .005 * sum(p.norm() for p in model.parameters())
    named = list(model.named_parameters())
    data_grads = torch.autograd.grad(task, [p for _, p in named], allow_unused=True)
    reg_grads = torch.autograd.grad(regularizer, [p for _, p in named])
    groups = {}
    for (name, p), data, reg in zip(named, data_grads, reg_grads):
        if name.startswith('input_projections'):
            group = '.'.join(name.split('.')[:2])
        elif name.startswith('label_projection'):
            group = 'label_projection'
        else:
            group = '.'.join(name.split('.')[:3])
        record = groups.setdefault(group, {'task_sq': 0., 'reg_sq': 0., 'wd_sq': 0., 'dot': 0., 'missing_task_tensors': 0})
        record['reg_sq'] += float(reg.square().sum())
        record['wd_sq'] += float((.005 * p.detach()).square().sum())
        if data is not None:
            record['task_sq'] += float(data.square().sum())
            record['dot'] += float((data * reg).sum())
        else:
            record['missing_task_tensors'] += 1
    for record in groups.values():
        record['task_norm'] = record.pop('task_sq') ** .5
        record['reg_norm'] = record.pop('reg_sq') ** .5
        record['weight_decay_norm'] = record.pop('wd_sq') ** .5
        record['reg_to_task_ratio'] = record['reg_norm'] / max(record['task_norm'], 1e-20)
        record['reg_task_cosine'] = record.pop('dot') / max(record['task_norm'] * record['reg_norm'], 1e-20)
    return {'model': label, 'classification_loss': float(task.detach()), 'explicit_regularization_loss': float(regularizer.detach()),
            'score_min': float(scores.detach().min()), 'score_max': float(scores.detach().max()),
            'score_sample_std': variation(scores, 0), 'score_label_std': variation(scores, 1),
            'stages': stages, 'gradient_groups': groups}


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument('--run_dir', default='results_corrected_5fold_fold1_80epochs')
    cli.add_argument('--output_dir', default='diagnosis_fold1')
    cli.add_argument('--experiment', choices=['inspect', 'no_explicit_regularizer', 'low_lr'], default='inspect')
    cli.add_argument('--epochs', type=int, default=33)
    opts = cli.parse_args()
    root = Path(opts.run_dir).resolve()
    output = Path(opts.output_dir)
    output.mkdir(exist_ok=True)
    manifest = json.loads((root / 'manifest.json').read_text(encoding='utf-8'))
    for name, sha in manifest['source_sha256'].items():
        assert hashlib.sha256((root / 'source' / name).read_bytes()).hexdigest() == sha
    # 优先载入确实生成当前坏结果的版本，工作文件变化不能污染根因判断。
    sys.path.insert(0, str(root / 'source'))
    from strict_training import VAMS, seed_all, transform, knn_indices, predict, metrics, label_frequency_loss
    import demo
    checkpoint = torch.load(root / 'checkpoints' / 'fold_1.pth', map_location='cpu', weights_only=False)
    args = SimpleNamespace(**checkpoint['args'])
    demo.args = args
    views, labels = demo.data_loader(str(Path(args.data_root) / args.dataset))
    labels = torch.as_tensor(labels).float()
    if len(labels) != len(views[0]):
        labels = labels.T
    views = [v.float() for v in views]
    train_ids, dev_ids = checkpoint['split']['train'], checkpoint['split']['dev']
    device = torch.device('cuda')
    torch.set_num_threads(1)
    raw = transform(views, train_ids, checkpoint['scalers'], device)
    dev_raw = transform(views, dev_ids, checkpoint['scalers'], device)
    y, dev_y = labels[train_ids].to(device), labels[dev_ids]
    frequency = y.mean(0)
    seed_all(args.seed)
    model = VAMS(checkpoint['input_dims'], checkpoint['labels'], args).to(device)
    if opts.experiment == 'inspect':
        initial = inspect(model, raw, y, frequency, args, 'initial_seed_2020')
        model.load_state_dict(checkpoint['state_dict'])
        chosen = inspect(model, raw, y, frequency, args, 'dev_selected_epoch_33')
        scores = predict(model, raw, dev_raw, args)
        wp, wn = frequency * torch.exp(args.beta * (1-frequency)), (1-frequency) * torch.exp(args.beta*frequency)
        report = {'protocol': manifest['protocol'], 'source_run': str(root), 'test_used': False,
                  'train_label_frequency': frequency.cpu().tolist(),
                  'optimal_constant_per_label': (wp/(wp+wn)).cpu().tolist(),
                  'optimal_shared_constant': float(wp.sum()/(wp+wn).sum()),
                  'initial': initial, 'selected': chosen,
                  'dev_metrics': metrics(dev_y, scores, args.threshold),
                  'dev_scores_per_label_mean': scores.mean(0).cpu().tolist(),
                  'dev_scores_per_label_std': scores.std(0, unbiased=False).cpu().tolist()}
        (output / 'inspection.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        print(json.dumps({k:v for k,v in report.items() if k not in ('initial','selected')}, indent=2), flush=True)
        for record in (initial, chosen):
            print(record['model'], 'task loss', record['classification_loss'], 'reg loss', record['explicit_regularization_loss'])
            print('gradient groups:', json.dumps(record['gradient_groups'], indent=2))
        return
    # 冻结预算的单因素对照；每次只改变显式正则或学习率，不进行测试评估。
    rate = args.lr if opts.experiment != 'low_lr' else args.lr / 10
    regularizer_weight = 0. if opts.experiment == 'no_explicit_regularizer' else .005
    optimizer = torch.optim.SGD(model.parameters(), lr=rate, momentum=.9, weight_decay=.005)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=.933)
    history, best = [], -float('inf')
    started = time.time()
    for epoch in range(1, opts.epochs+1):
        model.eval()
        with torch.no_grad():
            representation = model.encode(raw)
            neighbors = knn_indices(representation, representation, args.k, True)
        model.train()
        tasks, regs = [], []
        for batch, begin in enumerate(range(0, len(y), args.batch_size)):
            optimizer.zero_grad(set_to_none=True)
            representation = model.encode(raw)
            scores = model.predict_batch(representation[begin:begin+args.batch_size], representation, neighbors, batch, args)
            task = label_frequency_loss(scores, y[begin:begin+args.batch_size], frequency, args.beta)
            reg = regularizer_weight * sum(p.norm() for p in model.parameters())
            (task + reg).backward()
            optimizer.step()
            tasks.append(float(task.detach()))
            regs.append(float(reg.detach()))
        scores = predict(model, raw, dev_raw, args)
        values = metrics(dev_y, scores, args.threshold)
        row = {'epoch':epoch, 'task_loss':float(np.mean(tasks)), 'reg_loss':float(np.mean(regs)), **values,
               'score_sample_std': variation(scores,0), 'score_label_std':variation(scores,1),
               'predicted_positives':int((scores>=.5).sum()), 'score_min':float(scores.min()), 'score_max':float(scores.max())}
        history.append(row)
        if values['LRAP'] > best:
            best = values['LRAP']
            torch.save({'state_dict':model.state_dict(), 'epoch':epoch, 'dev_metrics':values,
                        'experiment':opts.experiment, 'test_used':False}, output / f'{opts.experiment}_best.pth')
        scheduler.step()
        print(opts.experiment, 'epoch', epoch, 'dev LRAP', round(values['LRAP'],6), 'micro F1',round(values['Micro_F1'],6),flush=True)
    with (output / f'{opts.experiment}.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer=csv.DictWriter(stream, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    result={'experiment':opts.experiment, 'epochs':opts.epochs, 'seconds':time.time()-started,
            'test_used':False, 'lr':rate, 'explicit_regularizer':regularizer_weight,
            'weight_decay':.005, 'best_dev':max(history,key=lambda r:r['LRAP']), 'last_dev':history[-1]}
    (output / f'{opts.experiment}.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2),flush=True)


if __name__ == '__main__':
    main()
