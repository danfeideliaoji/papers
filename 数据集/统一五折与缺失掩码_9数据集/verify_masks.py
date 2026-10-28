"""Read back every delivered split/mask and verify the agreed protocol."""
import json
import sys
from pathlib import Path

import numpy as np

root = Path(sys.argv[1])
sys.path.insert(0, str(root.resolve()))
from protocol_io import load_partition, apply_label_mask
from prepare_masks import file_hash, nested_group_masks, seed_words

protocol = json.loads((root / '协议.json').read_text(encoding='utf-8'))
assert protocol['view_rates'] == [0.0,0.1,0.3,0.5]
assert protocol['label_rates'] == [0.0,0.1,0.3,0.5,0.7]
assert len(list(root.glob('*/fold_*.npz'))) == 45
assert len(list(root.glob('*/masks/fold_*/view_mask_*.npz'))) == 180
assert len(list(root.glob('*/masks/fold_*/label_mask_*.npz'))) == 225
counts = {'datasets':0,'folds':0,'view_masks':0,'label_masks':0,'replay_files':0}
max_error = {'view':0.0,'label_outer_train':0.0,'label_inner_train':0.0}
all_unknown_main = {}
for source in protocol['source_records']:
    name = source['dataset']; d = root / name
    n, labels, views = source['samples'], source['labels'], len(source['view_dimensions'])
    assert file_hash(source['source_path']) == source['source_sha256']
    groups = np.load(d/'sample_group_ids.npy',allow_pickle=False)
    meta = json.loads((d/'metadata.json').read_text(encoding='utf-8'))
    counts['datasets'] += 1
    all_test=[]; main_unknown=[]
    for fold in range(5):
        with np.load(d/f'fold_{fold}.npz',allow_pickle=False) as saved:
            ids={k:saved[k] for k in saved.files}
        assert file_hash(d/f'fold_{fold}.npz') == meta['files'][fold]['sha256']
        train,val,test,outer=[ids[k] for k in ['train_idx','val_idx','test_idx','outer_train_idx']]
        assert np.array_equal(np.sort(np.r_[train,val]),outer)
        assert np.array_equal(np.sort(np.r_[train,val,test]),np.arange(n))
        for a,b in [(train,val),(train,test),(val,test)]:
            assert not np.intersect1d(a,b).size
            assert not np.intersect1d(groups[a],groups[b]).size
        all_test.append(test)
        directory=d/'masks'/f'fold_{fold}'
        record=json.loads((directory/'metadata.json').read_text(encoding='utf-8'))
        assert record['source_label_values_used'] is False
        roles={'train64':train,'val16':val,'outer_train80':outer,'test20':test}
        for kind,columns,key in [('view',views,'visible_mask'),('label',labels,'known_mask')]:
            previous=None
            for entry in record[f'{kind}_masks']:
                p=directory/entry['file'];rate=entry['rate']
                assert file_hash(p) == entry['sha256']
                with np.load(p,allow_pickle=False) as saved:
                    mask=saved[key]
                assert mask.shape==(n,columns) and mask.dtype==np.uint8
                assert np.isin(mask,[0,1]).all()
                if previous is not None:
                    assert np.all(mask<=previous), 'non-nested missingness'
                previous=mask
                # Equality within exact-feature groups for all original rows.
                order=np.argsort(groups,kind='stable'); equal=np.diff(groups[order])==0
                assert np.all(mask[order[1:]][equal]==mask[order[:-1]][equal])
                if kind=='view':
                    assert np.all(mask.sum(axis=1)>=1)
                    if views==2 and rate==0.5: assert np.all(mask.sum(axis=1)==1)
                    if rate==0: assert np.all(mask==1)
                    active=['train64','val16','outer_train80','test20']
                else:
                    assert np.all(mask[test]==0)
                    if rate==0: assert np.all(mask[outer]==1)
                    active=['train64','outer_train80']
                    if rate==0.5: main_unknown.append(int(np.count_nonzero(mask[outer].sum(axis=1)==0)))
                for role in active:
                    index=roles[role]; actual=float(1-mask[index].mean())
                    stats=entry['partitions'][role]
                    assert abs(stats['actual_missing_rate']-actual)<1e-12
                    assert stats['all_unknown_or_missing_rows']==int(np.count_nonzero(mask[index].sum(axis=1)==0))
                    assert stats['visible_per_column']==mask[index].sum(axis=0).astype(int).tolist()
                    metric='view' if kind=='view' else ('label_outer_train' if role=='outer_train80' else 'label_inner_train')
                    max_error[metric]=max(max_error[metric],abs(actual-rate))
                    if kind=='view' or role=='outer_train80':
                        if kind=='view' and role=='outer_train80':
                            # Views were sampled separately in inner train/val;
                            # the combined rounding bound is their sum.
                            max_weight=sum(np.unique(groups[roles[r]],return_counts=True)[1].max() for r in ['train64','val16'])
                        else:
                            max_weight=np.unique(groups[index],return_counts=True)[1].max()
                        assert abs(actual-rate)<=max_weight/(2*len(index)*columns)+1e-10, (name,fold,kind,rate,role,actual)
                counts[f'{kind}_masks']+=1
                # Full deterministic mask replay for fold0 of each dataset.
                if fold==0:
                    if kind=='label':
                        _,inverse,weights=np.unique(groups[outer],return_inverse=True,return_counts=True)
                        generated=dict(nested_group_masks(weights,columns,protocol['label_rates'],seed_words(name,fold,'label:outer_train80')))[rate]
                        assert np.array_equal(mask[outer],generated[inverse])
                    else:
                        for role in ['train64','val16','test20']:
                            index=roles[role]
                            _,inverse,weights=np.unique(groups[index],return_inverse=True,return_counts=True)
                            generated=dict(nested_group_masks(weights,columns,protocol['view_rates'],seed_words(name,fold,f'view:{role}'),True))[rate]
                            assert np.array_equal(mask[index],generated[inverse])
                    counts['replay_files']+=1
        small=load_partition(root,name,fold,'train',True)
        large=load_partition(root,name,fold,'train',False)
        positions=np.searchsorted(large['sample_ids'],small['sample_ids'])
        assert np.array_equal(large['sample_ids'][positions],small['sample_ids'])
        for key in ['view_mask','label_known_mask']:
            assert np.array_equal(large[key][positions],small[key])
        for role in ['val','test']:
            part=load_partition(root,name,fold,role,True)
            assert np.all(part['label_known_mask']==1)
        try: load_partition(root,name,fold,'val',False)
        except ValueError: pass
        else: raise AssertionError('Validation access allowed in no-validation API')
        # Unknown labels are -1; flipping hidden truth leaves observed labels unchanged.
        mask=small['label_known_mask']
        y=np.zeros(mask.shape,dtype=np.uint8)
        observed=apply_label_mask(y,mask)
        altered=y.copy();altered[mask==0]=1
        assert np.array_equal(observed,apply_label_mask(altered,mask))
        assert np.all(observed[mask==0]==-1) and np.all(observed[mask==1]==0)
        counts['folds']+=1
    assert np.array_equal(np.sort(np.concatenate(all_test)),np.arange(n))
    all_unknown_main[name]=main_unknown
result={'status':'PASS','checked':counts,'max_absolute_missing_rate_error':max_error,
        'main_50pct_label_mask_all_unknown_outer_train_rows_per_fold':all_unknown_main,
        'checks':['all splits unchanged by SHA256','each sample test exactly once','indices and groups disjoint',
                  '405 masks shape/dtype/binary/hash readback','nested missingness for every rate',
                  'at least one observed view per sample','same masks within duplicate groups',
                  'recorded actual rates and counts','outer80 label-mask consistency with inner64',
                  'validation/test full-label evaluation API','unknown labels are -1 and invariant to hidden truth',
                  'deterministic full fold0 mask replay on all 9 datasets','all 9 raw sources retain original SHA256'],
        'training_or_metrics_evaluation_run':False}
(root/'核验结果.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(result,ensure_ascii=False))
