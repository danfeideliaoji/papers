# 小规模图结构诊断，不训练数据集，不覆盖保存的完整训练模型。
# 验证后续批次和尾部小批次的节点/边对齐、输出设备、梯度有限性。

"""Check sample/neighbor alignment and original-model backward propagation on CPU."""
from types import SimpleNamespace
import argparse
import torch
from GNBlock_model import _Model


# 构造可核对的人工特征与邻居索引，依次验证构图、前向和反向。
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    device = torch.device(parser.parse_args().device)
    torch.set_num_threads(1)
    torch.manual_seed(2020)
    args = SimpleNamespace(batch_size=64, k=5)
    # 使用两个 128 维视图和循环邻居，便于精确检查图节点来源。
    full_views = torch.randn(140, 2, 128)
    indices = torch.stack([
        (torch.arange(140)[:, None] + torch.arange(1, 6)) % 140
        for _ in range(2)
    ])
    label_features = torch.randn(6, 128)
    model = _Model().to(device)
    # A later, short batch exercises both the batch offset and the tail batch.
    # 取第二批的 7 个样本，同时覆盖非零 batch_num 和未满 64 的尾部批次。
    samples = full_views[64:71]
    labels = torch.randint(0, 2, (7, 2, 6)).float()
    graph, label_graph = model.graph_generator(
        samples, label_features, labels.transpose(0, 1), args,
        indices, full_views, 1)
    for sample in range(7):
        base = sample * 12
        torch.testing.assert_close(graph.x[base:base + 2], samples[sample])
        for view in range(2):
            expected = full_views[indices[view, 64 + sample], view]
            torch.testing.assert_close(graph.x[base + 2 + view * 5:base + 7 + view * 5], expected)
    counts = torch.tensor(graph.kwargs)
    cross_graph = model.generate_cross_graph(label_graph, counts)
    for sample in range(7):
        edges = cross_graph.edge_index[:, sample * 12:(sample + 1) * 12]
        assert torch.equal(edges[0].unique(), torch.tensor([sample * 12, sample * 12 + 1]))
        assert torch.equal(edges[1].unique(), torch.arange(sample * 6, (sample + 1) * 6))
    prediction, counts = model(
        samples, label_features.unsqueeze(0), labels, args,
        indices.unsqueeze(0), full_views.unsqueeze(0), 1)
    assert prediction.shape == (84, 1)
    assert prediction.device == device or (device.type == 'cuda' and prediction.device.type == 'cuda')
    assert torch.isfinite(prediction).all()
    # 只做一次诊断反向传播，确认梯度存在且有限，没有执行优化器更新。
    prediction.mean().backward()
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    assert gradients and all(torch.isfinite(g).all() for g in gradients)
    assert any(g.abs().sum() > 0 for g in gradients)
    print('PASS:', prediction.device, 'neighbor alignment, cross-sample edges, tail batch, forward and backward')


if __name__ == '__main__':
    main()
