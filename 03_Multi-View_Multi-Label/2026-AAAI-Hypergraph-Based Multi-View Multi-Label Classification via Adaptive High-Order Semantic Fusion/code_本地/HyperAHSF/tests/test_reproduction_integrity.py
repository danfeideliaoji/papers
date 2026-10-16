from types import SimpleNamespace
import json

import pytest
import numpy as np
import torch
import torch.nn.functional as F

from model import HGMVMLC_Model
from utils import PrecomputedContrastiveLoss
from corel5k_cv import five_fold_indices, scale_fold, independent_ap_micro
from metrics import AveragePrecision, MicroF1
import scipy.io
from corel5k_cv import run_corel5k


def toy_model():
    data = SimpleNamespace(features=[torch.randn(8, 4), torch.randn(8, 4)],
                           labels=torch.tensor([[1, 0], [1, 0], [0, 1], [0, 1]] * 2))
    params = dict(k=2, thr=0.5, test_k=2, test_thr=99, num_views=2, num_class=2,
                  num_layers=2, inputs_sizes=[4, 4], EDC_feature_matrix=[[4], [4]],
                  specific_dim_matrix=[[4], [4]], hid_dim=4)
    return HGMVMLC_Model(data, params, torch.device('cpu'))


def reference_loss(x, labels, weight):
    xn = F.normalize(x, dim=1)
    terms = []
    for row in labels.unique(dim=0):
        mask = (labels == row).all(dim=1)
        if mask.sum() < 2:
            continue
        proto = F.normalize(xn[mask].mean(dim=0, keepdim=True), dim=1)
        sim = (xn @ proto.T).squeeze(1)
        negative = sim[~mask].mean() if (~mask).any() else x.sum() * 0
        terms.append(negative - sim[mask].mean())
    return weight * torch.stack(terms).mean() if terms else x.sum() * 0


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_vectorized_loss_and_gradients_match_original_loop(dtype):
    torch.manual_seed(4)
    labels = torch.tensor([[1, 0], [1, 0], [0, 1], [0, 1], [1, 1]])
    x = torch.randn(5, 7, dtype=dtype, requires_grad=True)
    got = PrecomputedContrastiveLoss(labels, 0.1)(x)
    ref = reference_loss(x, labels, 0.1)
    torch.testing.assert_close(got, ref)
    grad = torch.autograd.grad(got, x, retain_graph=True)[0]
    ref_grad = torch.autograd.grad(ref, x)[0]
    torch.testing.assert_close(grad, ref_grad, atol=1e-7, rtol=1e-5)


def test_identical_labels_have_finite_loss_and_gradient():
    labels = torch.ones(5, 2, dtype=torch.long)
    x = torch.randn(5, 4, requires_grad=True)
    loss = PrecomputedContrastiveLoss(labels, 0.1)(x)
    assert torch.isfinite(loss)
    torch.testing.assert_close(loss, reference_loss(x, labels, 0.1))
    assert torch.isfinite(torch.autograd.grad(loss, x)[0]).all()


def test_training_one_fold_cannot_change_next_folds_activation():
    first = toy_model()
    with torch.no_grad():
        first.common_HGNN.act.weight.fill_(0.83)
    second = toy_model()
    assert second.common_HGNN.act is not first.common_HGNN.act
    torch.testing.assert_close(second.common_HGNN.act.weight, torch.tensor([0.25]))


def test_evaluation_graph_is_rebuilt_for_a_different_dataset():
    model = toy_model().eval()
    with torch.no_grad():
        model.evaluate([torch.randn(6, 4), torch.randn(6, 4)])
        logits, _, _ = model.evaluate([torch.randn(4, 4), torch.randn(4, 4)])
    assert logits.shape == (4, 2)


def test_jaccard_exact_threshold_neighbors_are_retained_and_empty_union_is_safe():
    graph_builder = SimpleNamespace(num_views=1, device=torch.device('cpu'))
    x = torch.tensor([[0., 0., 0., 0.], [.01, 0., 0., 0.],
                      [10., 0., 0., 0.], [10.01, 0., 0., 0.]])
    labels = torch.tensor([[1, 1], [1, 0], [0, 1], [0, 0]])
    graph = HGMVMLC_Model.construct_label_hyperedge_index(graph_builder, [x], labels, 1, 0.5)[0]
    # First pair has exact Jaccard=1/2 in both directions; both two-node edges must survive.
    assert (graph[1] >= len(x)).sum().item() == 4
    empty = HGMVMLC_Model.construct_label_hyperedge_index(
        graph_builder, [x], torch.zeros_like(labels), 1, 0.5)[0]
    assert empty.shape == (2, len(x))


def test_five_folds_cover_each_sample_once_without_overlap():
    splits = five_fold_indices(4999)
    assert [len(test) for _, test in splits] == [1000, 1000, 1000, 1000, 999]
    np.testing.assert_array_equal(np.sort(np.concatenate([test for _, test in splits])), np.arange(4999))
    for train, test in splits:
        assert not np.intersect1d(train, test).size
        assert len(train) + len(test) == 4999


def test_scaler_is_independent_of_held_out_values():
    view = np.array([[1., 0.], [3., 1.], [1000., -900.]])
    train, _, scalers = scale_fold([view], np.array([0, 1]), np.array([2]))
    view[2] = [-1e9, 1e9]
    train_changed, _, scalers_changed = scale_fold([view], np.array([0, 1]), np.array([2]))
    np.testing.assert_array_equal(train[0], train_changed[0])
    np.testing.assert_array_equal(scalers[0].mean_, [2., 0.5])
    np.testing.assert_array_equal(scalers[0].mean_, scalers_changed[0].mean_)


def test_independent_metrics_cover_ties_and_unlabeled_rows_without_rounding():
    labels = np.array([[1, 0, 1], [0, 0, 0], [1, 0, 0]])
    logits = np.array([[0., 0., 0.], [4., 3., 2.], [0., 1., 2.]])
    predictions = (logits > 0).astype(np.int64)
    ap, micro = independent_ap_micro(logits, labels, predictions)
    assert ap == pytest.approx(2. / 3.)
    assert micro == pytest.approx(MicroF1(predictions, labels))
    assert ap == pytest.approx(AveragePrecision(logits, labels))
    assert ap != round(ap, 4)


def test_float32_scores_do_not_reduce_precision_of_ap_counting():
    rng = np.random.RandomState(12)
    logits = rng.randn(50, 260).astype(np.float32)
    labels = np.zeros((50, 260), dtype=np.int64)
    for row in labels:
        row[rng.choice(260, 3, replace=False)] = 1
    ap, _ = independent_ap_micro(logits, labels, (logits > 0).astype(np.int64))
    assert abs(AveragePrecision(logits, labels) - ap) < 1e-12


def test_entrypoint_persists_fixed_rules_and_recomputable_results(tmp_path):
    rng = np.random.RandomState(8)
    views = np.empty((2, 1), dtype=object)
    views[:, 0] = [rng.randn(25, 4), rng.randn(25, 4)]
    labels = np.tile([[1, 0], [0, 1], [1, 1], [0, 0], [1, 0]], (5, 1))
    data = tmp_path / 'synthetic_fixture.mat'
    scipy.io.savemat(data, {'data': views, 'target': labels.T})
    params = dict(k=2, thr=0.5, test_k=2, test_thr=99, num_views=2, num_class=2,
                  num_layers=2, inputs_sizes=[4, 4], EDC_feature_matrix=[[4], [4]],
                  specific_dim_matrix=[[4], [4]], hid_dim=4, epochs=1,
                  lr=0.001, weight_decay=0.003, w_intra=0.1, confidence_bar=0.3)
    args = SimpleNamespace(folds=1, threads=2, cpu=True, device=0, seed=0,
                           split_seed=3407, output=str(tmp_path / 'run'), data=str(data))
    output = run_corel5k(args, params)
    manifest = json.loads((output / 'manifest.json').read_text(encoding='utf-8'))
    summary = json.loads((output / 'summary.json').read_text(encoding='utf-8'))
    assert manifest['status'] == 'complete'
    assert manifest['epoch_selection'] == 'fixed_final_epoch_1'
    assert summary['sample_stds'] is None
    assert not summary['full_corel5k_shape']
    assert (output / 'fold0' / 'final_checkpoint.pt').exists()
    saved = np.load(output / 'fold0' / 'predictions.npz')
    ap, micro = independent_ap_micro(saved['logits'], saved['labels'], saved['predictions'])
    assert ap == pytest.approx(summary['means']['AP'])
    assert micro == pytest.approx(summary['means']['micro'])
