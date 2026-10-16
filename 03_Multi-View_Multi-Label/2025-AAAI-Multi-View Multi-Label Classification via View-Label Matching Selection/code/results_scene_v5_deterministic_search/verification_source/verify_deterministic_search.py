"""重放 v5 内部选模实验，核对候选、早停、训练信息边界和最终测试。"""
import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch


def read_csv(path):
    with path.open(encoding='utf-8-sig') as f: return list(csv.DictReader(f))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('output_dir')
    cli = p.parse_args()
    root = Path(cli.output_dir).resolve()
    manifest = json.loads((root/'manifest.json').read_text(encoding='utf-8'))
    assert manifest['status'] == 'complete'
    assert manifest['candidate_grid'] == ['sgd', 'adam']
    for name, h in manifest['source_sha256'].items():
        assert hashlib.sha256((root/'source'/name).read_bytes()).hexdigest() == h
    sys.path.insert(0, str(root/'source'))
    from runtime_config import configure_determinism
    configure_determinism()
    from strict_training import (VAMS, make_splits, fit_scalers, transform, predict, metrics)
    import demo
    args = SimpleNamespace(**manifest['args']); demo.args=args
    path = Path(args.data_root)/args.dataset
    assert hashlib.sha256(path.read_bytes()).hexdigest() == manifest['data_sha256']
    raw, y = demo.data_loader(str(path)); raw=[v.float() for v in raw]
    y=torch.as_tensor(y).float()
    if y.shape[0] != len(raw[0]): y=y.T
    expected = make_splits(len(y), args.k_fold, args.seed, args.dev_fraction)
    epochs=read_csv(root/'epoch_metrics.csv'); trials=read_csv(root/'candidate_metrics.csv')
    tests=read_csv(root/'fold_test_metrics.csv')
    assert len(trials)==10 and len(tests)==5
    all_ids=[]; errors=[]
    for fold,(tr,dv,te) in enumerate(expected,1):
        scalers=fit_scalers(raw,tr)
        tv=transform(raw,tr,scalers,'cuda'); vv=transform(raw,dv,scalers,'cuda')
        fold_trials=[t for t in trials if int(t['fold'])==fold]
        assert [t['candidate'] for t in fold_trials]==manifest['candidate_grid']
        assert fold_trials[0]['initial_state_sha256']==fold_trials[1]['initial_state_sha256']
        winner=max(fold_trials,key=lambda t: float(t['dev_LRAP']))
        selected=tests[fold-1]
        assert int(selected['fold'])==fold and selected['selected_candidate']==winner['candidate']
        for trial in fold_trials:
            candidate=trial['candidate']
            curve=[r for r in epochs if int(r['fold'])==fold and r['candidate']==candidate]
            assert [int(r['epoch']) for r in curve]==list(range(1,int(trial['epochs_run'])+1))
            assert 80<=len(curve)<=120
            best=max(curve,key=lambda r: float(r['dev_LRAP']))
            assert int(best['epoch'])==int(trial['selected_epoch'])
            assert float(best['dev_LRAP'])==float(trial['dev_LRAP'])
            for row in curve:
                assert abs(float(row['lr'])-args.lr*.933**(int(row['epoch'])-1))<1e-12
            if len(curve)<120: assert len(curve)-int(best['epoch'])>=20
            cp=torch.load(root/'candidate_checkpoints'/f'fold_{fold}_{candidate}.pth',map_location='cpu',weights_only=False)
            assert cp['protocol']==manifest['protocol'] and cp['candidate']==candidate
            assert cp['epoch']==int(best['epoch']) and cp['dev_LRAP']==float(best['dev_LRAP'])
            for actual,wanted in zip((cp['split'][k] for k in ('train','dev','test')),(tr,dv,te)):
                np.testing.assert_array_equal(actual,wanted)
            assert not(set(tr)&set(dv) or set(tr)&set(te) or set(dv)&set(te))
            # CUDA 均值的 FP32 倒数乘法与 CPU 除法可能差一个 ULP。
            # 用训练标签的 FP64 精确计数作独立参照，只容许约两 ULP。
            torch.testing.assert_close(cp['frequency'],y[tr].double().mean(0).float(),atol=3e-8,rtol=0)
            for wanted,actual in zip(scalers,cp['scalers']):
                for a,b in zip(wanted,actual): torch.testing.assert_close(a,b,atol=0,rtol=0)
            model=VAMS(cp['input_dims'],cp['labels'],args).to('cuda'); model.load_state_dict(cp['state_dict'])
            ds=predict(model,tv,vv,args)
            with np.load(root/f'fold_{fold}_{candidate}_dev_predictions.npz') as saved:
                np.testing.assert_array_equal(dv,saved['sample_ids']); np.testing.assert_array_equal(y[dv],saved['labels'])
                actual=ds.cpu().numpy()
                np.testing.assert_allclose(actual,saved['predictions'],atol=5e-6,rtol=1e-6)
                np.testing.assert_array_equal(actual>=args.threshold,saved['predictions']>=args.threshold)
                errors.append(float(np.abs(actual-saved['predictions']).max()))
            assert abs(metrics(y[dv],ds,args.threshold)['LRAP']-cp['dev_LRAP'])<1e-7
            if candidate==winner['candidate']:
                chosen=root/'checkpoints'/f'fold_{fold}.pth'
                assert hashlib.sha256(chosen.read_bytes()).hexdigest()==hashlib.sha256((root/'candidate_checkpoints'/f'fold_{fold}_{candidate}.pth').read_bytes()).hexdigest()
                ts=predict(model,tv,transform(raw,te,scalers,'cuda'),args)
                with np.load(root/f'fold_{fold}_predictions.npz') as saved:
                    np.testing.assert_array_equal(te,saved['sample_ids']); np.testing.assert_array_equal(y[te],saved['labels'])
                    actual=ts.cpu().numpy(); want=saved['predictions']
                    np.testing.assert_allclose(actual,want,atol=5e-6,rtol=1e-6)
                    np.testing.assert_array_equal(actual>=args.threshold,want>=args.threshold)
                    np.testing.assert_array_equal(np.argsort(actual,axis=1),np.argsort(want,axis=1))
                    errors.append(float(np.abs(actual-want).max()))
                for k,v in metrics(y[te],ts,args.threshold).items(): assert abs(v-float(selected[k]))<1e-7
                all_ids.extend(te.tolist())
            del model
    assert sorted(all_ids)==list(range(len(y)))
    summary=json.loads((root/'summary.json').read_text())
    assert summary['total_epochs']==len(epochs)
    for k in summary['mean']:
        vals=[float(t[k]) for t in tests]
        assert abs(np.mean(vals)-summary['mean'][k])<1e-12
        assert abs(np.std(vals,ddof=1)-summary['std'][k])<1e-12
    report={'status':'PASS','source_and_data_hashes':'verified','folds':5,'candidates':10,
            'epochs':len(epochs),'test_rows':len(all_ids), 'early_stop_and_inner_selection':'verified',
            'train_only_scalers_and_frequencies':'verified','dev_and_test_predictions':'replayed',
            'frequency_reference':'FP64 mean of inner training labels', 'frequency_atol':3e-8,
            'max_prediction_abs_error':max(errors),'test_classifications_and_ranks':'identical',
            'preflight_cross_process_three_epoch_repeatability':'see formula_alignment_v5/training_repeatability.json'}
    (root/'verification.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__': main()
