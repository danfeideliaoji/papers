"""软超图构造和加权双向聚合的数值行为测试。"""

import copy
import unittest

import torch

from experiments.candidate1.soft_graph import (
    SoftHypergraphBuilder,
    aggregate_edge_to_node,
    aggregate_node_to_edge,
)


class SoftHypergraphPublicBehaviorTests(unittest.TestCase):
    """保护软图有限性、归一化、边界稳定性、等价性与确定性。"""

    def setUp(self):
        """创建固定随机双视图特征、教师嵌入和非极端概率。"""
        torch.manual_seed(7)
        self.features = [torch.randn(7, 4), torch.randn(7, 3)]
        self.embeddings = [torch.randn(7, 5), torch.randn(7, 6)]
        self.probabilities = [torch.rand(7, 3) * 0.8 + 0.1 for _ in range(2)]

    def test_build_returns_finite_normalized_sparse_graphs_without_labels(self):
        """无真实标签构图应返回有限非负权重，且每条超边成员权重和为 1。"""
        builder = SoftHypergraphBuilder(k=3)
        bundle = builder.build(
            self.features,
            self.embeddings,
            self.probabilities,
            torch.ones(2),
        )

        self.assertEqual(len(bundle.view_graphs), 2)
        self.assertEqual(bundle.reliability.shape, (7, 2))
        self.assertTrue(torch.isfinite(bundle.reliability).all())
        self.assertTrue((bundle.reliability > 0).all())
        for graph in (*bundle.view_graphs, bundle.common_graph):
            self.assertEqual(graph.edge_index.shape[1], graph.incidence_weight.numel())
            self.assertTrue(torch.isfinite(graph.incidence_weight).all())
            self.assertTrue((graph.incidence_weight >= 0).all())
            sums = torch.zeros(graph.num_edges).scatter_add_(
                0, graph.edge_index[1], graph.incidence_weight
            )
            self.assertTrue(torch.allclose(sums, torch.ones_like(sums), atol=1e-6))

    def test_builder_handles_single_view_and_extreme_probabilities_without_nan(self):
        """单视图、k 大于邻居数及 0/1 极端概率均不应产生 NaN/Inf。"""
        builder = SoftHypergraphBuilder(k=20)
        probabilities = [
            torch.tensor(
                [[0.0, 1.0], [1.0, 0.0], [0.5, 0.5], [1e-12, 1 - 1e-12]],
                dtype=torch.float32,
            )
        ]
        bundle = builder.build(
            [torch.randn(4, 2)],
            [torch.randn(4, 3)],
            probabilities,
            torch.tensor([1.0]),
        )
        self.assertTrue(torch.isfinite(bundle.incidence_weight).all())
        self.assertTrue(torch.isfinite(bundle.reliability).all())
        self.assertEqual(bundle.view_graphs[0].num_edges, 4)

    def test_all_one_incidence_matches_unweighted_row_normalized_propagation(self):
        """成员权重全为 1 时，加权聚合应退化为普通行平均传播。"""
        x = torch.tensor([[1.0, 2.0], [3.0, 4.0], [9.0, 6.0]])
        edge_index = torch.tensor([[0, 1, 1, 2], [0, 0, 1, 1]])
        ones = torch.ones(edge_index.shape[1])

        edge_features = aggregate_node_to_edge(x, edge_index, ones, num_edges=2)
        expected_edges = torch.stack(((x[0] + x[1]) / 2, (x[1] + x[2]) / 2))
        self.assertTrue(torch.allclose(edge_features, expected_edges))

        node_features = aggregate_edge_to_node(
            edge_features, edge_index, ones, num_nodes=3
        )
        expected_nodes = torch.stack(
            (expected_edges[0], (expected_edges[0] + expected_edges[1]) / 2, expected_edges[1])
        )
        self.assertTrue(torch.allclose(node_features, expected_nodes))

    def test_build_is_deterministic_for_same_inputs(self):
        """相同输入不依赖随机状态，必须产生完全相同的图索引和权重。"""
        builder = SoftHypergraphBuilder(k=3)
        first = builder.build(
            self.features, self.embeddings, self.probabilities, torch.ones(2)
        )
        second = builder.build(
            copy.deepcopy(self.features),
            copy.deepcopy(self.embeddings),
            copy.deepcopy(self.probabilities),
            torch.ones(2),
        )
        self.assertTrue(torch.equal(first.hyperedge_index, second.hyperedge_index))
        self.assertTrue(torch.allclose(first.incidence_weight, second.incidence_weight))


if __name__ == "__main__":
    unittest.main()
