"""Read fixed folds/masks. This module does not load raw features or labels."""
from pathlib import Path
import numpy as np


def load_partition(root, dataset, fold, partition, needs_validation=True,
                   view_missing_rate=0.5, label_missing_rate=0.5):
    directory = Path(root) / dataset
    with np.load(directory / f'fold_{fold}.npz', allow_pickle=False) as saved:
        if partition == 'train':
            ids = saved['train_idx' if needs_validation else 'outer_train_idx']
        elif partition == 'val' and needs_validation:
            ids = saved['val_idx']
        elif partition == 'test':
            ids = saved['test_idx']
        else:
            raise ValueError('partition must be train/test, or val when needs_validation=True')
    masks = directory / 'masks' / f'fold_{fold}'
    with np.load(masks / f'view_mask_{view_missing_rate:.1f}.npz', allow_pickle=False) as saved:
        view_mask = saved['visible_mask'][ids]
    with np.load(masks / f'label_mask_{label_missing_rate:.1f}.npz', allow_pickle=False) as saved:
        known_mask = saved['known_mask'][ids] if partition == 'train' else np.ones((len(ids), saved['known_mask'].shape[1]), dtype=np.uint8)
    return {'sample_ids': ids, 'view_mask': view_mask, 'label_known_mask': known_mask}


def apply_label_mask(binary_labels, known_mask):
    labels = np.asarray(binary_labels)
    mask = np.asarray(known_mask)
    if labels.shape != mask.shape or not np.isin(labels, [0, 1]).all() or not np.isin(mask, [0, 1]).all():
        raise ValueError('Expected equally shaped binary labels and binary known mask')
    return np.where(mask == 1, labels, -1).astype(np.int8)
