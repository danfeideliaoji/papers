"""验证六项修复的行为边界，而非用正则梯度掩盖分类路径缺失。"""
import argparse
import json
from types import SimpleNamespace
from pathlib import Path
import torch
from strict_training import (VAMS, make_splits, fit_scalers, transform,
                             knn_indices, label_frequency_loss)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    args = SimpleNamespace(device=p.parse_args().device, nhidden=128, target_dim=128,
                           dropout=.4, batch_size=7, k=3)
    device = torch.device(args.device)
    torch.set_num_threads(1)
    torch.manual_seed(31)
    # 分折不丢样本，train/dev/test 无交集。
    splits = make_splits(593, 5, 2020, .2)
    assert sorted(i for _, _, te in splits for i in te) == list(range(593))
    for tr, dv, te in splits:
        assert not (set(tr) & set(dv) or set(tr) & set(te) or set(dv) & set(te))
    # 修改未参与训练的极端值，不应改变训练统计量；测试可超出 [0,1]。
    raw = [torch.tensor([[2.], [4.], [100.]])]
    scalers = fit_scalers(raw, [0, 1])
    raw[0][2] = -10000
    changed = fit_scalers(raw, [0, 1])
    for original, new in zip(scalers[0], changed[0]):
        torch.testing.assert_close(original, new)
    assert transform(raw, [2], scalers, device)[0].item() < 0
    # 重复特征时按真实身份排除自己；验证/测试邻居索引只指向训练库。
    duplicate = torch.ones(9, 2, 128)
    nn_ids = knn_indices(duplicate, duplicate, 3, True)
    assert all(i not in nn_ids[v, i].tolist() for i in range(9) for v in range(2))
    assert knn_indices(duplicate, torch.ones(4, 2, 128), 3).max() < 9
    # 独立手算式 (11)：p=.25，y=1/0，r=.5，beta=1。
    calculated = label_frequency_loss(torch.full((2, 1), .5), torch.tensor([[1.], [0.]]), torch.tensor([.25]), 1.)
    torch.testing.assert_close(calculated, torch.tensor(1.1787057), atol=1e-6, rtol=1e-6)
    # 完整模型分类损失反传；覆盖后一批、尾批、不同 k 与批大小。
    raw_views = [torch.randn(20, 64, device=device), torch.randn(20, 8, device=device)]
    model = VAMS([64, 8], 6, args).to(device)
    features = model.encode(raw_views)
    neighbors = knn_indices(features, features, args.k, True)
    scores = model.predict_batch(features[14:20], features, neighbors, 2, args)
    y = torch.randint(0, 2, scores.shape, device=device).float()
    loss = label_frequency_loss(scores, y, y.mean(0), 1.)
    optimizer = torch.optim.SGD(model.parameters(), lr=.01)
    before = [p.detach().clone() for p in model.input_projections.parameters()]
    loss.backward()  # 不添加任何正则项。
    missing = [name for name, p in model.named_parameters() if p.grad is None]
    # 末层图内边在解码前没有后续节点读取，这是边解码架构的自然终点。
    # 必须把它与用户指出的“末层节点全部断开”区分，不能用正则项掩盖。
    expected_unused = [name for name, _ in model.named_parameters()
                       if name.startswith(('graph.pro_layer2.edge_model.edge_mlp_ins.',
                                           'graph.pro_layer2.edge_model.edge_mlp_label.'))]
    assert sorted(missing) == sorted(expected_unused), missing
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    last_nodes = list(model.graph.pro_layer2.node_model.parameters())
    assert all(p.grad.abs().sum() > 0 for p in last_nodes)
    assert all(any(p.grad.abs().sum() > 0 for p in projection.parameters()) for projection in model.input_projections)
    optimizer.step()
    assert any(not torch.equal(old, new) for old, new in zip(before, model.input_projections.parameters()))
    # 标签完全图：每样本 6*5 条边、没有自环；真实标签仅用于损失。
    g, label_g = model.graph.graph_generator(features[14:20].detach().cpu(), torch.randn(6, 128),
                                             torch.zeros(2, 6, 6), args, neighbors, features.detach().cpu(), 2)
    assert label_g.edge_index.shape == (2, 6 * 6 * 5)
    assert torch.all(label_g.edge_index[0] != label_g.edge_index[1])
    report = {'status': 'PASS', 'device': str(device), 'split_no_overlap': True,
              'all_593_test_rows_preserved': True, 'train_only_preprocessing': True,
              'train_only_knn_reference': True, 'duplicate_self_exclusion': True,
              'loss_formula_hand_calculation': True, 'learned_input_projections': True,
              'complete_label_graph_without_self_loops': True,
              'last_node_parameter_tensors_with_nonzero_task_gradient': len(last_nodes),
              'terminal_intra_graph_edge_tensors_without_task_gradient': len(expected_unused),
              'unexpected_missing_task_gradients': 0}
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
