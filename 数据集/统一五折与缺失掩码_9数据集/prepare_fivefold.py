"""Prepare index-only multilabel folds; no preprocessing/model fitting or missing masks."""
import argparse
import csv
import hashlib
import json
import platform
import shutil
from pathlib import Path

import numpy as np
import scipy
import scipy.io as sio
from scipy import sparse

PROTOCOL = 'multilabel_grouped_iterative_holdout_v1'
SEED = 20261026


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def load_source(path):
    data = sio.loadmat(path)
    label_key = next(k for k in ['label', 'target', 'targets'] if k in data)
    raw_y = np.asarray(data[label_key])
    view_key = next(k for k in ['X', 'view', 'data'] if k in data and data[k].dtype == object)
    raw_views = list(data[view_key].flat)
    first = raw_views[0]
    possibilities = [n for n in raw_y.shape if n in first.shape]
    if len(possibilities) != 1:
        raise ValueError(f'Ambiguous sample axis: {raw_y.shape}, {first.shape}')
    n = possibilities[0]
    if raw_y.shape[0] != n:
        raw_y = raw_y.T
    values = np.unique(raw_y)
    if not np.isfinite(raw_y).all() or not set(values.tolist()).issubset({-1, 0, 1}):
        raise ValueError(f'Unexpected full-label coding: {values}')
    y = (raw_y > 0).astype(np.int32)
    views = []
    for value in raw_views:
        value = value.toarray() if sparse.issparse(value) else np.asarray(value)
        if value.shape[0] != n:
            value = value.T
        assert value.shape[0] == n and value.ndim == 2
        if not np.isfinite(value).all():
            raise ValueError('Non-finite raw features require explicit handling')
        views.append(np.ascontiguousarray(value))
    return y, views, {'label_key': label_key, 'view_key': view_key,
                     'raw_label_values': values.tolist()}


def exact_feature_groups(views):
    n = len(views[0])
    digests = [hashlib.sha256() for _ in range(n)]
    for view in views:
        for i, row in enumerate(view):
            # Exact raw bytes within this dataset, without fitting a transform.
            digests[i].update(row.tobytes())
    group_of = np.empty(n, dtype=np.int64)
    dictionary = {}
    for i, digest in enumerate(digests):
        token = digest.digest()
        if token not in dictionary:
            dictionary[token] = len(dictionary)
        group_of[i] = dictionary[token]
    # Verify equality for each grouped row, independently of its hash.
    first = {}
    for i, group in enumerate(group_of):
        if group in first:
            assert all(np.array_equal(v[i], v[first[group]]) for v in views)
        else:
            first[group] = i
    return group_of


def iterative_assign(group_y, weights, fractions, seed):
    """Rarest-label-first, sample-weighted, group-constrained iterative stratification.

    This is a custom documented splitter, not an invocation of iterstrat. Labels
    balance positive sample counts; groups stay intact. Largest-remainder sample
    budgets are exact for singleton groups and approximate for duplicate groups.
    """
    rng = np.random.default_rng(seed)
    fractions = np.asarray(fractions, dtype=float)
    assert np.isclose(fractions.sum(), 1)
    wanted = fractions * weights.sum()
    budgets = np.floor(wanted).astype(np.int64)
    extra = int(weights.sum() - budgets.sum())
    order = np.argsort(-(wanted - budgets), kind='stable')
    budgets[order[:extra]] += 1
    remaining_budget = budgets.copy()
    remaining_labels = group_y.sum(axis=0).astype(np.int64)
    label_need = fractions[:, None] * remaining_labels[None, :]
    unassigned = np.ones(len(weights), dtype=bool)
    assignment = np.full(len(weights), -1, dtype=np.int8)

    def choose(group, label=None):
        feasible = np.flatnonzero(remaining_budget >= weights[group])
        candidates = feasible if len(feasible) else np.arange(len(fractions))
        if label is not None:
            need = label_need[candidates, label]
            candidates = candidates[np.isclose(need, need.max(), rtol=0, atol=1e-10)]
        capacity = remaining_budget[candidates]
        candidates = candidates[capacity == capacity.max()]
        fold = int(rng.choice(candidates))
        assignment[group] = fold
        unassigned[group] = False
        remaining_budget[fold] -= weights[group]
        label_need[fold] -= group_y[group]
        remaining_labels[:] -= group_y[group]

    while np.any(remaining_labels > 0):
        positive = np.flatnonzero(remaining_labels > 0)
        rare = positive[remaining_labels[positive] == remaining_labels[positive].min()]
        label = int(rng.choice(rare))
        candidates = np.flatnonzero(unassigned & (group_y[:, label] > 0))
        rng.shuffle(candidates)
        candidates = candidates[np.argsort(-weights[candidates], kind='stable')]
        for group in candidates:
            choose(group, label)
    remainder = np.flatnonzero(unassigned)
    rng.shuffle(remainder)
    remainder = remainder[np.argsort(-weights[remainder], kind='stable')]
    for group in remainder:
        choose(group)
    assert np.all(assignment >= 0)
    return assignment, budgets


def verify_fold(arrays, group_of, n):
    train, val, test, outer = [arrays[k] for k in ['train_idx', 'val_idx', 'test_idx', 'outer_train_idx']]
    for ids in [train, val, test, outer]:
        assert ids.dtype == np.int64 and np.array_equal(np.unique(ids), ids)
        assert len(ids) and ids.min() >= 0 and ids.max() < n
    assert not np.intersect1d(train, val).size
    assert not np.intersect1d(train, test).size
    assert not np.intersect1d(val, test).size
    assert np.array_equal(np.sort(np.concatenate([train, val])), outer)
    assert np.array_equal(np.sort(np.concatenate([outer, test])), np.arange(n))
    for a, b in [(train, val), (train, test), (val, test)]:
        assert not np.intersect1d(group_of[a], group_of[b]).size


def prepare_one(item, output):
    name, path = item['dataset'], Path(item['path'])
    digest = sha256_file(path)
    y, views, coding = load_source(path)
    n, labels = y.shape
    group_of = exact_feature_groups(views)
    n_groups = int(group_of.max() + 1)
    weights = np.bincount(group_of)
    group_y = np.zeros((n_groups, labels), dtype=np.int64)
    np.add.at(group_y, group_of, y)
    outer, budgets = iterative_assign(group_y, weights, [0.2] * 5, SEED)
    # Re-run only the deterministic partitioner to verify seed reproducibility.
    replay, _ = iterative_assign(group_y, weights, [0.2] * 5, SEED)
    assert np.array_equal(outer, replay)
    group_has_label = (group_y > 0).sum(axis=0)
    d = output / name
    d.mkdir()
    np.save(d / 'sample_group_ids.npy', group_of, allow_pickle=False)
    rows, supports, fold_files = [], {}, []
    all_test = []
    for fold in range(5):
        fit_groups = np.flatnonzero(outer != fold)
        inner, inner_budgets = iterative_assign(group_y[fit_groups], weights[fit_groups], [0.8, 0.2], SEED + fold + 1)
        inner_replay, _ = iterative_assign(group_y[fit_groups], weights[fit_groups], [0.8, 0.2], SEED + fold + 1)
        assert np.array_equal(inner, inner_replay)
        role = np.full(n_groups, 2, dtype=np.int8)
        role[fit_groups] = inner
        arrays = {key: np.flatnonzero(condition).astype(np.int64) for key, condition in {
            'train_idx': role[group_of] == 0,
            'val_idx': role[group_of] == 1,
            'test_idx': outer[group_of] == fold,
            'outer_train_idx': outer[group_of] != fold,
        }.items()}
        verify_fold(arrays, group_of, n)
        all_test.append(arrays['test_idx'])
        f = d / f'fold_{fold}.npz'
        np.savez_compressed(f, **arrays)
        # Verify serialized indices rather than trusting in-memory state alone.
        with np.load(f, allow_pickle=False) as saved:
            restored = {k: saved[k] for k in saved.files}
            verify_fold(restored, group_of, n)
            assert all(np.array_equal(restored[k], arrays[k]) for k in arrays)
        fold_files.append({'file': f.name, 'sha256': sha256_file(f)})
        supports[str(fold)] = {}
        row = {'dataset': name, 'fold': fold, 'samples': n, 'views': len(views),
               'labels': labels, 'groups': n_groups,
               'train': len(arrays['train_idx']), 'validation': len(arrays['val_idx']),
               'test': len(arrays['test_idx']), 'final_refit_train': len(arrays['outer_train_idx'])}
        for key, ids in arrays.items():
            count = y[ids].sum(axis=0)
            supports[str(fold)][key] = {'positive_samples_per_label': count.tolist(),
                                     'labels_without_positive': np.flatnonzero(count == 0).tolist(),
                                     'empty_label_rows': int((y[ids].sum(axis=1) == 0).sum())}
        rows.append(row)
    assert np.array_equal(np.sort(np.concatenate(all_test)), np.arange(n))
    assert sha256_file(path) == digest
    conflicts = 0
    for group in np.flatnonzero(weights > 1):
        values = y[group_of == group]
        conflicts += int(np.any(values != values[0]))
    metadata = {'dataset': name, 'source_path': str(path), 'source_sha256': digest,
        'source_bytes': path.stat().st_size, 'protocol': PROTOCOL, 'seed': SEED,
        'outer_folds': 5, 'outer_target_test_fraction': 0.2,
        'inner_validation_fraction_of_outer_train': 0.2,
        'inner_seeds': [SEED + i + 1 for i in range(5)],
        'index_base': 0, 'index_reference': 'original MAT sample rows, before any shuffle or filtering',
        'samples': n, 'labels': labels, 'view_dimensions': [int(v.shape[1]) for v in views],
        'exact_feature_groups': n_groups, 'duplicate_extra_rows': n - n_groups,
        'largest_group': int(weights.max()), 'duplicate_groups_with_label_disagreement': conflicts,
        'group_identity_limit': 'Exact equality over all stored raw feature views only; no claim of original image identity or near-duplicate detection.',
        'labels_in_fewer_than_five_groups': np.flatnonzero(group_has_label < 5).tolist(),
        'outer_target_fold_sample_counts': budgets.tolist(), 'folds': rows,
        'label_supports': supports, 'files': fold_files,
        'missing_masks': 'NOT_GENERATED; do not inherit old masks without mapping source rows and checking new fitting boundaries',
        'feature_preprocessing': 'NOT_FITTED; train-only fit for selection, outer-train-only refit for final evaluation',
        'full_label_access': 'Preparation-only stratification and label-support summaries; no model training or evaluation.',
        'source_coding': coding,
        'versions': {'python': platform.python_version(), 'numpy': np.__version__, 'scipy': scipy.__version__}}
    (d / 'metadata.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'dataset': name, 'n': n, 'groups': n_groups, 'duplicates': n-n_groups,
                      'fold_counts': [[r[k] for k in ['train','validation','test','final_refit_train']] for r in rows]}, ensure_ascii=False), flush=True)
    return metadata, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sources', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    sources = json.loads(args.sources.read_text(encoding='utf-8'))
    assert len(sources) == 9 and len({s['dataset'] for s in sources}) == 9
    if args.output.exists():
        raise FileExistsError('Output must be a new directory; existing splits are never replaced')
    args.output.mkdir(parents=True)
    all_meta, all_rows = [], []
    for item in sources:
        meta, rows = prepare_one(item, args.output)
        all_meta.append(meta)
        all_rows.extend(rows)
    with (args.output / '划分数量.csv').open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0]))
        w.writeheader()
        w.writerows(all_rows)
    shutil.copyfile(args.sources, args.output / 'sources.json')
    shutil.copyfile(Path(__file__), args.output / 'prepare_fivefold.py')
    summary = {'protocol': PROTOCOL, 'seed': SEED, 'datasets': len(all_meta), 'folds': len(all_rows),
               'verification': {'indices_in_range_and_unique': True, 'partitions_disjoint_and_cover_source': True,
                                'each_sample_test_once': True, 'exact_feature_groups_never_cross_boundaries': True,
                                'deterministic_replay': True, 'serialized_readback': True, 'source_hash_unchanged': True},
               'source_versions': [{k: m[k] for k in ['dataset','source_path','source_sha256','samples','labels','view_dimensions','exact_feature_groups']} for m in all_meta],
               'status': 'INDEX_PREPARATION_COMPLETE; MASKS_AND_TRAINING_NOT_RUN'}
    (args.output / '核验结果.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    lines = ['# 九个数据集的统一五折索引', '',
        '已生成 9 个数据集 × 5 折，共 45 套索引。原始特征与标签不复制，来源路径及 SHA256 保存在各数据集 metadata.json。原数据和旧实验文件未修改，本次没有删除其他数据集。', '',
        '## 划分规则', '',
        '- 固定 seed=20261026。采用自定义、按样本数加权的稀有标签优先迭代分层，并将全部原始视图特征精确相同的行作为不可拆分组。它不是 iterstrat 包的标准行级实现；规则见随附脚本。',
        '- 外层五折，每个样本恰好测试一次。每折约 80% 为外层训练、20% 测试；外层训练再留约 20% 验证，即总体约 64/16/20。分组约束与标签稀有程度会影响精确数量和标签平衡，实数见表。',
        '- 内部选模：只用 train_idx 拟合预处理、模型和供体库，用 val_idx 选参数及训练轮数。最终评估：固定选择后，在 outer_train_idx 上从头重建预处理、供体库及模型，使用 test_idx 一次评估。',
        '- 全标签仅用于数据准备阶段的分层和标签支持统计。缺失掩码尚未生成；选定视图/标签缺失率后再按原样本行号统一制作，同折同设置的所有方法共用。旧模型、旧掩码与旧归一化/SVD/GP缓存不能直接当作新折训练产物。',
        '- 精确特征分组不能识别所有同源图片或近重复。原始上游特征提取/码本拟合范围也没有因重新划分而改变。此次划分不清除历史数据曝光。', '',
        '## 各数据集规模与实际数量', '', '| 数据集 | 样本 | 视图 | 标签 | 五折测试数量 | 第0折：内部训练/验证/最终重训 |', '|---|---:|---:|---:|---|---|']
    for meta in all_meta:
        f0 = meta['folds'][0]
        lines.append('| '+meta['dataset']+' | '+str(meta['samples'])+' | '+str(len(meta['view_dimensions']))+' | '+str(meta['labels'])+' | '+ '/'.join(str(r['test']) for r in meta['folds'])+' | '+f"{f0['train']}/{f0['validation']}/{f0['final_refit_train']}"+' |')
    lines += ['', 'IAPRTC12 当前采用 4,999 样本、5 视图的本地子集；MIRFlickr 和 ESPGame 分别采用完整 25,000 和 20,770 样本六视图版。Scene 固定为 2,407 样本、2 视图、6 标签版。它们与仓库中其他同名版本不能交换索引。', '',
        '## 文件与使用', '',
        '每个数据集目录包含 fold_0.npz 至 fold_4.npz、sample_group_ids.npy 和 metadata.json。NPZ 只含四个 int64 数组：', '',
        '- train_idx：内部训练行。', '- val_idx：内部验证行。', '- test_idx：外层测试行。',
        '- outer_train_idx：内部训练与验证的并集，用于选好设置后的最终重新训练。', '',
        '所有索引均从 0 开始，指向原始 MAT 中样本轴的行号（若原矩阵转置存储，先按 metadata 指示整理成样本×特征/标签）。不要对原始行重新洗牌、删行后仍使用这些索引。', '',
        '```python', 'import numpy as np', 'with np.load("Corel5k/fold_0.npz", allow_pickle=False) as split:', '    train_ids = split["train_idx"]', '    val_ids = split["val_idx"]', '    test_ids = split["test_idx"]', '    final_train_ids = split["outer_train_idx"]', '```', '',
        '生成器需要 NumPy、SciPy。复现到新目录：', '',
        '```text', 'python prepare_fivefold.py --sources sources.json --output 一个尚不存在的新目录', '```', '',
        '45 折均通过范围、唯一性、分区覆盖、测试恰好一次、组间隔离、同种子重放及落盘读回核验；原始来源文件运行前后 SHA256 一致。核验结果见 核验结果.json，标签稀有性及逐折支持见各 metadata.json。','']
    (args.output / '说明.md').write_text('\n'.join(lines), encoding='utf-8')
    print('COMPLETE', flush=True)


if __name__ == '__main__':
    main()
