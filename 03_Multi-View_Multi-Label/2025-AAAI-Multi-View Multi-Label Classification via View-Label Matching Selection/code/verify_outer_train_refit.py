"""核验完整外层训练折的固定轮数重拟合，重放测试预测，不重新选模。"""
import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(path):
    with path.open(encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('run', nargs='?', default='results_scene_v5_refit_outer_train')
    root = Path(p.parse_args().run).resolve()
    manifest = json.loads((root/'manifest.json').read_text(encoding='utf-8'))
    assert manifest['status'] == 'complete'
    assert manifest['protocol'] == 'vams_v5_deterministic_outer_train_refit'
    for name, wanted in manifest['source_sha256'].items():
        assert sha(root/'source'/name) == wanted, name
    sys.path.insert(0, str(root/'source'))
    from runtime_config import configure_determinism
    configure_determinism()
    import numpy as np
    import torch
    import demo
    from strict_training import VAMS, seed_all, make_splits, fit_scalers, transform, predict, metrics
    from deterministic_search import state_hash

    parent = Path(manifest['parent_run'])
    parent_manifest = json.loads((parent/'manifest.json').read_text(encoding='utf-8'))
    for key,value in parent_manifest['args'].items():
        if key != 'output_dir': assert manifest['args'][key] == value, key
    for name,wanted in parent_manifest['source_sha256'].items():
        assert manifest['source_sha256'][name] == wanted
    assert manifest['gamma'] == .933 and manifest['selection'] == 'fixed_from_parent_inner_dev_only'
    assert sha(parent/'candidate_metrics.csv') == manifest['candidate_metrics_sha256']
    assert json.loads((parent/'verification.json').read_text())['status'] == 'PASS'
    trials = rows(parent/'candidate_metrics.csv')
    args = SimpleNamespace(**manifest['args']); demo.args = args
    data = Path(args.data_root)/args.dataset
    assert sha(data) == manifest['data_sha256']
    raw, y = demo.data_loader(str(data)); raw = [v.float() for v in raw]
    y = torch.as_tensor(y).float()
    if y.shape[0] != len(raw[0]): y = y.T
    splits = make_splits(len(y), args.k_fold, args.seed, args.dev_fraction)
    saved_splits = json.loads((root/'splits.json').read_text())
    epochs = rows(root/'epoch_metrics.csv'); tests = rows(root/'fold_test_metrics.csv')
    assert len(tests) == len(splits) == len(manifest['refit_plan']) == 5
    errors = []; all_ids = []
    for fold, ((tr, dv, te), split, choice, test) in enumerate(zip(splits, saved_splits, manifest['refit_plan'], tests), 1):
        winner = max([r for r in trials if int(r['fold']) == fold], key=lambda r: float(r['dev_LRAP']))
        expected_choice = {'fold':fold, 'candidate':winner['candidate'], 'epochs':int(winner['selected_epoch']),
                           'selection_dev_LRAP':float(winner['dev_LRAP']), 'initial_state_sha256':winner['initial_state_sha256']}
        assert choice == expected_choice
        outer = np.sort(np.concatenate([tr, dv]))
        for key, wanted in [('train',outer), ('test',te), ('selection_train',tr), ('selection_dev',dv)]:
            np.testing.assert_array_equal(split[key], wanted)
        assert not set(outer)&set(te) and sorted(np.concatenate([outer,te]).tolist()) == list(range(len(y)))
        curve = [r for r in epochs if int(r['fold']) == fold]
        assert [int(r['epoch']) for r in curve] == list(range(1,choice['epochs']+1))
        for row in curve:
            assert row['candidate'] == choice['candidate'] and np.isfinite(float(row['train_loss']))
            assert abs(float(row['lr'])-args.lr*.933**(int(row['epoch'])-1)) < 1e-12
        cp = torch.load(root/'checkpoints'/f'fold_{fold}.pth', map_location='cpu', weights_only=False)
        assert cp['protocol'] == manifest['protocol'] and cp['epoch'] == choice['epochs']
        assert cp['args'] == manifest['args']
        assert cp['candidate'] == choice['candidate'] and cp['selection_dev_LRAP'] == choice['selection_dev_LRAP']
        assert cp['initial_state_sha256'] == choice['initial_state_sha256']
        # 重拟合不再有独立验证集；旧验证行已加入完整训练折。
        assert set(cp['split']) == {'train','test'}
        for key,wanted in [('train',outer),('test',te)]: np.testing.assert_array_equal(cp['split'][key],wanted)
        for key,wanted in [('train',tr),('dev',dv)]: np.testing.assert_array_equal(cp['selection_split'][key],wanted)
        scalers = fit_scalers(raw, outer)
        for wanted,actual in zip(scalers,cp['scalers']):
            for a,b in zip(wanted,actual): torch.testing.assert_close(a,b,atol=0,rtol=0)
        torch.testing.assert_close(cp['frequency'], y[outer].double().mean(0).float(), atol=3e-8, rtol=0)
        initial = torch.load(parent/'checkpoints'/f'fold_{fold}.pth',map_location='cpu',weights_only=False)
        assert cp['initial_state_sha256'] == initial['initial_state_sha256']
        seed_all(args.seed+fold-1)
        model = VAMS(cp['input_dims'],cp['labels'],args).to('cuda')
        assert state_hash(model) == choice['initial_state_sha256']
        model.load_state_dict(cp['state_dict'])
        assert cp['input_dims'] == [v.shape[1] for v in raw] and cp['labels'] == y.shape[1]
        score = predict(model, transform(raw,outer,scalers,'cuda'), transform(raw,te,scalers,'cuda'),args)
        with np.load(root/f'fold_{fold}_predictions.npz') as saved:
            np.testing.assert_array_equal(saved['sample_ids'],te); np.testing.assert_array_equal(saved['labels'],y[te])
            actual = score.cpu().numpy(); wanted = saved['predictions']
            np.testing.assert_allclose(actual,wanted,atol=5e-6,rtol=1e-6)
            np.testing.assert_array_equal(actual>=args.threshold,wanted>=args.threshold)
            np.testing.assert_array_equal(np.argsort(actual,axis=1),np.argsort(wanted,axis=1))
            errors.append(float(np.abs(actual-wanted).max()))
        assert int(test['fold']) == fold and test['candidate'] == choice['candidate']
        assert int(test['fixed_epochs']) == choice['epochs'] and int(test['train_rows']) == len(outer)
        assert float(test['selection_dev_LRAP']) == choice['selection_dev_LRAP']
        for key,value in metrics(y[te],score,args.threshold).items(): assert abs(value-float(test[key])) < 1e-7
        all_ids.extend(te.tolist()); del model
        print(f'Fold {fold}: fixed refit and test replay verified',flush=True)
    assert sorted(all_ids) == list(range(len(y)))
    summary = json.loads((root/'summary.json').read_text())
    assert summary['status'] == 'complete' and summary['folds'] == 5
    assert summary['total_epochs'] == len(epochs) == manifest['total_planned_epochs'] == 491
    for key in summary['mean']:
        vals = [float(r[key]) for r in tests]
        assert abs(np.mean(vals)-summary['mean'][key]) < 1e-12
        assert abs(np.std(vals,ddof=1)-summary['std'][key]) < 1e-12
    report = {'status':'PASS','folds':5,'epochs':len(epochs),'test_rows':len(all_ids),
              'source_data_and_parent_candidate_hashes':'verified','fixed_inner_selection_and_refit_epochs':'verified',
              'outer_train_scalers_and_frequencies':'verified','frequency_atol':3e-8,
              'test_predictions':'replayed','max_prediction_abs_error':max(errors),
              'test_classifications_and_ranks':'identical','independent_refit_dev_set':False,
              'claim_scope':manifest['claim_scope']}
    (root/'verification.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__ == '__main__': main()
