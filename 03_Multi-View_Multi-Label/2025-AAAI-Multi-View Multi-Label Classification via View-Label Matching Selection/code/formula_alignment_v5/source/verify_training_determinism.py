"""修复回归：独立进程从同一初始权重重复真实训练三轮，核对权重及预测。"""
from runtime_config import configure_determinism
import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from strict_training import VAMS, seed_all, transform, knn_indices, label_frequency_loss, predict
import demo


def worker(path):
    base = Path('results_scene_v4_5fold_80epochs')
    args = SimpleNamespace(**json.loads((base/'manifest.json').read_text())['args'])
    demo.args = args
    raw, y = demo.data_loader('data/scene.mat')
    raw = [v.float() for v in raw]; y = torch.as_tensor(y).T.float()
    cp = torch.load(base/'checkpoints/fold_1.pth', map_location='cpu', weights_only=False)
    tr, dv = cp['split']['train'], cp['split']['dev']
    tv, vv = transform(raw, tr, cp['scalers'], 'cuda'), transform(raw, dv, cp['scalers'], 'cuda')
    seed_all(args.seed)
    assert torch.are_deterministic_algorithms_enabled()
    model = VAMS(cp['input_dims'], cp['labels'], args).to('cuda')
    opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=.9)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=.933)
    losses = []
    for epoch in range(3):
        model.eval()
        with torch.no_grad():
            ref = model.encode(tv); neighbors = knn_indices(ref, ref, args.k, exclude_self=True)
        model.train()
        for batch, s in enumerate(range(0, len(tr), args.batch_size)):
            opt.zero_grad(set_to_none=True)
            ref = model.encode(tv)
            pred = model.predict_batch(ref[s:s+args.batch_size], ref, neighbors, batch, args)
            loss = label_frequency_loss(pred, y[tr[s:s+args.batch_size]].to('cuda'), cp['frequency'].to('cuda'), args.beta)
            assert torch.isfinite(loss)
            loss.backward()
            assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
            opt.step(); losses.append(float(loss.detach()))
        scheduler.step()
    h = hashlib.sha256()
    for name, value in model.state_dict().items():
        h.update(name.encode()); h.update(value.detach().cpu().contiguous().numpy().tobytes())
    np.savez_compressed(path, predictions=predict(model, tv, vv, args).cpu().numpy(),
                        state_sha256=h.hexdigest(), losses=np.array(losses))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--worker')
    p.add_argument('--output_dir', default='formula_alignment_v5')
    args = p.parse_args()
    if args.worker: return worker(args.worker)
    out = Path(args.output_dir); out.mkdir(exist_ok=True)
    env = dict(os.environ, CUBLAS_WORKSPACE_CONFIG=':4096:8', PYTHONUTF8='1', PYTHONDONTWRITEBYTECODE='1')
    for i in range(2):
        subprocess.run([sys.executable, str(Path(__file__).resolve()), '--worker', str(out/f'repeat_{i}.npz')],
                       env=env, check=True)
    with np.load(out/'repeat_0.npz') as a, np.load(out/'repeat_1.npz') as b:
        assert str(a['state_sha256']) == str(b['state_sha256'])
        np.testing.assert_array_equal(a['losses'], b['losses'])
        np.testing.assert_array_equal(a['predictions'], b['predictions'])
        report = {'status': 'PASS', 'independent_processes': 2, 'epochs_each': 3,
                  'identical_final_weights': True, 'identical_batch_losses': True,
                  'max_dev_prediction_change': float(np.abs(a['predictions']-b['predictions']).max()),
                  'test_evaluations': 0, 'full_120_epoch_repeatability': 'not_tested'}
    (out/'training_repeatability.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__': main()
