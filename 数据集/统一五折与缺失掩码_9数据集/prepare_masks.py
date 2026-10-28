"""Create nested, label-independent missingness masks for the existing nine-dataset folds."""
import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np

SEED = 20261026
VIEW_RATES = [0.0, 0.1, 0.3, 0.5]
LABEL_RATES = [0.0, 0.1, 0.3, 0.5, 0.7]
PROTOCOL = 'nested_group_consistent_missingness_v1'


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def seed_words(dataset, fold, component):
    token = hashlib.sha256(f'{dataset}|{fold}|{component}'.encode()).digest()
    return [SEED] + np.frombuffer(token[:16], dtype='<u4').astype(int).tolist()


def nested_group_masks(weights, columns, rates, words, protect_one=False):
    """Uniform random ordering of eligible group/column slots, weighted by rows.

    One view is uniformly protected per group when protect_one=True. Each rate
    chooses the nearest cumulative row-weighted prefix. No label values are read.
    A complete group-slot is hidden at once, preserving duplicate-group identity.
    """
    rng = np.random.default_rng(words)
    group_count = len(weights)
    candidates = np.arange(group_count * columns, dtype=np.int64)
    if protect_one:
        protected = rng.integers(columns, size=group_count)
        candidates = candidates[candidates % columns != protected[candidates // columns]]
    rng.shuffle(candidates)
    cumulative = np.cumsum(weights[candidates // columns], dtype=np.int64)
    total_slots = int(weights.sum()) * columns
    for rate in rates:
        if protect_one and rate > 1 - 1 / columns + 1e-12:
            raise ValueError('Missingness exceeds the one-visible-view constraint')
        target = rate * total_slots
        position = int(np.searchsorted(cumulative, target))
        choices = sorted({min(position, len(candidates)), min(position + 1, len(candidates))})
        cut = min(choices, key=lambda k: (abs((int(cumulative[k - 1]) if k else 0) - target), k))
        mask = np.ones((group_count, columns), dtype=np.uint8)
        mask.flat[candidates[:cut]] = 0
        yield rate, mask


def mask_stats(mask):
    return {'rows': len(mask), 'actual_missing_rate': float(1 - mask.mean()),
            'all_unknown_or_missing_rows': int(np.count_nonzero(mask.sum(axis=1) == 0)),
            'min_visible_per_row': int(mask.sum(axis=1).min()),
            'visible_per_column': mask.sum(axis=0).astype(int).tolist()}


def save_mask(path, key, array):
    np.savez_compressed(path, **{key: array})
    return {'file': path.name, 'sha256': file_hash(path), 'shape': list(array.shape)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--splits', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Use a new output directory; existing artifacts are preserved')
    directories = sorted(p.parent for p in args.splits.glob('*/metadata.json'))
    assert len(directories) == 9
    args.output.mkdir(parents=True)
    rows, source_records, fold_records = [], [], []
    for source_dir in directories:
        meta = json.loads((source_dir / 'metadata.json').read_text(encoding='utf-8'))
        dataset, n, labels = meta['dataset'], meta['samples'], meta['labels']
        views = len(meta['view_dimensions'])
        source_path = Path(meta['source_path'])
        if file_hash(source_path) != meta['source_sha256']:
            raise ValueError(f'Source changed: {dataset}')
        source_records.append({k: meta[k] for k in ['dataset','source_path','source_sha256','samples','labels','view_dimensions']})
        destination = args.output / dataset
        destination.mkdir()
        group_file = source_dir / 'sample_group_ids.npy'
        groups = np.load(group_file, allow_pickle=False)
        assert groups.shape == (n,)
        shutil.copyfile(group_file, destination / group_file.name)
        group_hash = file_hash(group_file)
        copied_folds = []
        for fold in range(5):
            original = source_dir / f'fold_{fold}.npz'
            expected = meta['files'][fold]['sha256']
            assert file_hash(original) == expected
            shutil.copyfile(original, destination / original.name)
            assert file_hash(destination / original.name) == expected
            copied_folds.append({'file': original.name, 'sha256': expected})
            with np.load(original, allow_pickle=False) as saved:
                ids = {k: saved[k] for k in saved.files}
            assert np.array_equal(np.sort(np.concatenate([ids['train_idx'],ids['val_idx'],ids['test_idx']])), np.arange(n))
            out = destination / 'masks' / f'fold_{fold}'
            out.mkdir(parents=True)
            roles = {'train64': ids['train_idx'], 'val16': ids['val_idx'],
                     'outer_train80': ids['outer_train_idx'], 'test20': ids['test_idx']}
            view_arrays = {r: np.zeros((n, views), dtype=np.uint8) for r in VIEW_RATES}
            rng_records = {}
            for role in ['train64','val16','test20']:
                index = roles[role]
                unique, inverse, weights = np.unique(groups[index], return_inverse=True, return_counts=True)
                words = seed_words(dataset, fold, f'view:{role}')
                rng_records[f'view:{role}'] = words
                for rate, mask in nested_group_masks(weights, views, VIEW_RATES, words, protect_one=True):
                    view_arrays[rate][index] = mask[inverse]
            record = {'dataset': dataset, 'fold': fold, 'split_sha256': expected,
                      'group_file_sha256': group_hash, 'view_masks': [], 'label_masks': []}
            for rate, array in view_arrays.items():
                assert np.all(array.sum(axis=1) >= 1)
                entry = save_mask(out / f'view_mask_{rate:.1f}.npz', 'visible_mask', array)
                entry.update({'rate': rate, 'partitions': {k: mask_stats(array[v]) for k,v in roles.items()}})
                record['view_masks'].append(entry)
                for role, index in roles.items():
                    rows.append({'dataset':dataset,'fold':fold,'kind':'view','nominal_missing_rate':rate,
                                 'partition':role,**{k:v for k,v in mask_stats(array[index]).items() if k!='visible_per_column'}})
            outer_ids = ids['outer_train_idx']
            unique, inverse, weights = np.unique(groups[outer_ids], return_inverse=True, return_counts=True)
            words = seed_words(dataset, fold, 'label:outer_train80')
            rng_records['label:outer_train80'] = words
            for rate, mask in nested_group_masks(weights, labels, LABEL_RATES, words):
                array = np.zeros((n, labels), dtype=np.uint8)
                array[outer_ids] = mask[inverse]
                assert np.all(array[ids['test_idx']] == 0)
                entry = save_mask(out / f'label_mask_{rate:.1f}.npz', 'known_mask', array)
                entry.update({'rate':rate, 'partitions': {k:mask_stats(array[roles[k]]) for k in ['train64','outer_train80']}})
                record['label_masks'].append(entry)
                for role in ['train64','outer_train80']:
                    rows.append({'dataset':dataset,'fold':fold,'kind':'label','nominal_missing_rate':rate,
                                 'partition':role,**{k:v for k,v in mask_stats(array[roles[role]]).items() if k!='visible_per_column'}})
            record.update({'rng': 'numpy.default_rng/PCG64', 'rng_seed_words':rng_records,
                           'source_label_values_used':False,
                           'evaluation_labels':'Validation/test use full labels in evaluator; stored training known_mask is NOT an evaluation mask.'})
            (out / 'metadata.json').write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
            fold_records.append(record)
        meta['files'] = copied_folds
        meta['missing_masks'] = {'protocol':PROTOCOL,'path':'masks/fold_<0..4>/','view_rates':VIEW_RATES,'label_rates':LABEL_RATES}
        meta['training_protocol'] = {
            'needs_validation':'Train on train_idx (~64%), select on val_idx (~16%), directly evaluate on test_idx (~20%).',
            'no_validation':'Train on outer_train_idx (~80%), directly evaluate on the same test_idx (~20%).',
            'outer_train_idx':'Outer training pool; refitting a model selected on val_idx is optional, not required.'}
        for item in meta['folds']:
            item['outer_train'] = item.pop('final_refit_train')
        (destination / 'metadata.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
        assert file_hash(source_path) == meta['source_sha256']
        print(json.dumps({'dataset':dataset,'folds':5,'view_masks':20,'label_masks':25},ensure_ascii=False),flush=True)
    with (args.output / '实际缺失率.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)
    settings = {'main':{'view_missing_rate':0.5,'label_missing_rate':0.5},
                'view_sensitivity':[{'view_missing_rate':r,'label_missing_rate':0.5} for r in [0.1,0.3,0.5]],
                'label_sensitivity':[{'view_missing_rate':0.5,'label_missing_rate':r} for r in [0.1,0.3,0.5,0.7]],
                'optional_controls':[{'view_missing_rate':0.0,'label_missing_rate':0.0},
                                     {'view_missing_rate':0.0,'label_missing_rate':0.5},
                                     {'view_missing_rate':0.5,'label_missing_rate':0.0}],
                'execution':'Masks only; no training or evaluation was launched. Main setting is prepared for all 9 datasets; sensitivity need not run on all datasets.'}
    manifest={'protocol':PROTOCOL,'seed':SEED,'view_rates':VIEW_RATES,'label_rates':LABEL_RATES,
        'view_rate_denominator':'sample rows x views within each internal partition',
        'label_rate_denominator':'outer_train_idx sample rows x labels; internal train64 rate is also recorded',
        'value_semantics':'1=observed/known, 0=hidden; arrays indexed by original zero-based sample row',
        'group_policy':'Exact-feature duplicate rows share identical view/label masks within each fold.',
        'view_sampling':'One uniformly protected view per group, then uniform random ordering of other group/view slots. Row-weighted nearest prefixes select missing slots. This is constrained random missingness, not independent Bernoulli MCAR.',
        'label_sampling':'Uniform random ordering of group/label slots; row-weighted nearest prefixes hide positive/negative positions without reading their values. No known-positive repair and no forced known label per sample.',
        'nesting':'Within each fold, hidden positions at a lower rate remain hidden at every higher rate.',
        'validation_and_test_labels':'Full benchmark labels are evaluation-only. For no-validation methods, inner-val sample rows enter outer training ONLY with the stored training label mask.',
        'source_records':source_records,'fold_count':len(fold_records),'view_mask_count':len(fold_records)*len(VIEW_RATES),
        'label_mask_count':len(fold_records)*len(LABEL_RATES),'numpy_version':np.__version__,
        'settings':settings}
    (args.output/'协议.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    shutil.copyfile(args.splits/'sources.json',args.output/'sources.json')
    shutil.copyfile(Path(__file__),args.output/'prepare_masks.py')
    shutil.copyfile(Path(__file__).with_name('protocol_io.py'),args.output/'protocol_io.py')
    for name in ['verify_masks.py','说明.md']:
        companion=Path(__file__).with_name(name)
        if companion.exists(): shutil.copyfile(companion,args.output/name)
    print('MASK_GENERATION_COMPLETE',flush=True)


if __name__=='__main__':
    main()
