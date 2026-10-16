"""汇总固定协议的五个 v4 单折运行，保留各折来源，不重复训练第一折。"""
import csv
import hashlib
import json
import shutil
from pathlib import Path
import numpy as np

def read_csv(path):
    with path.open(encoding='utf-8-sig') as f: return list(csv.DictReader(f))

def write_csv(path,rows):
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def main():
    runs=[Path(f'results_paper_aligned_v4_5fold_fold{i}_80epochs') for i in range(1,6)]
    root=Path('results_paper_aligned_v4_5fold_80epochs')
    if root.exists(): raise FileExistsError('汇总目录已存在，拒绝覆盖')
    manifests=[json.loads((r/'manifest.json').read_text()) for r in runs]
    baseline=manifests[0]
    splits=json.loads((runs[0]/'splits.json').read_text())
    rows,epochs,provenance=[],[],[]
    seen=[]
    for fold,(r,m) in enumerate(zip(runs,manifests),1):
        assert m['status']=='complete' and m['selected_folds']==[fold]
        assert m['protocol']==baseline['protocol']=='vams_paper_alignment_train_dev_test_v4'
        assert m['source_sha256']==baseline['source_sha256']
        assert m['data_sha256']==baseline['data_sha256']
        assert json.loads((r/'splits.json').read_text())==splits
        for name,value in baseline['args'].items():
            if name not in ['output_dir','fold_index']: assert m['args'][name]==value,(fold,name)
        for name,h in m['source_sha256'].items():
            assert hashlib.sha256((r/'source'/name).read_bytes()).hexdigest()==h,(fold,name)
        er=read_csv(r/'epoch_metrics.csv');fr=read_csv(r/'fold_test_metrics.csv')
        assert len(er)==80 and len(fr)==1
        assert [int(x['epoch']) for x in er]==list(range(1,81))
        assert all(int(x['fold'])==fold for x in er) and int(fr[0]['fold'])==fold
        with np.load(r/f'fold_{fold}_predictions.npz') as prediction:
            np.testing.assert_array_equal(prediction['sample_ids'],splits[fold-1]['test'])
            assert np.isfinite(prediction['predictions']).all()
            seen.extend(prediction['sample_ids'].tolist())
        rows.extend(fr);epochs.extend(er)
        s=json.loads((r/'summary.json').read_text());assert s['status']=='complete'
        provenance.append({'fold':fold,'source_run':r.name,'training_seconds':s['seconds'],
            'checkpoint_sha256':hashlib.sha256((r/'checkpoints'/f'fold_{fold}.pth').read_bytes()).hexdigest(),
            'predictions_sha256':hashlib.sha256((r/f'fold_{fold}_predictions.npz').read_bytes()).hexdigest()})
    assert sorted(seen)==list(range(baseline['samples'])) and len(set(seen))==593
    root.mkdir();(root/'checkpoints').mkdir();shutil.copytree(runs[0]/'source',root/'source')
    shutil.copy2(runs[0]/'splits.json',root/'splits.json')
    for fold,r in enumerate(runs,1):
        shutil.copy2(r/'checkpoints'/f'fold_{fold}.pth',root/'checkpoints'/f'fold_{fold}.pth')
        shutil.copy2(r/f'fold_{fold}_predictions.npz',root/f'fold_{fold}_predictions.npz')
    write_csv(root/'epoch_metrics.csv',epochs);write_csv(root/'fold_test_metrics.csv',rows)
    manifest=dict(baseline);manifest['args']=dict(baseline['args'],fold_index=None,output_dir='./'+root.name)
    manifest['selected_folds']=[1,2,3,4,5];manifest['assembly']='fold_1_reused_plus_fixed_folds_2_to_5'
    manifest['source_runs']=provenance
    (root/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    metrics=['LRAP','Hamming','Ranking','Coverage','Micro_F1','Macro_F1']
    summary={'status':'complete','protocol':baseline['protocol'],'folds':5,'outer_folds':5,
        'selected_folds':[1,2,3,4,5],'epochs':80,'scope':'all_folds',
        'seconds':sum(p['training_seconds'] for p in provenance),
        'new_folds_seconds':sum(p['training_seconds'] for p in provenance[1:]),
        'mean':{k:float(np.mean([float(x[k]) for x in rows])) for k in metrics},
        'std':{k:float(np.std([float(x[k]) for x in rows],ddof=1)) for k in metrics},
        'std_definition':'sample_std_ddof_1','unique_test_rows':len(seen)}
    (root/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
