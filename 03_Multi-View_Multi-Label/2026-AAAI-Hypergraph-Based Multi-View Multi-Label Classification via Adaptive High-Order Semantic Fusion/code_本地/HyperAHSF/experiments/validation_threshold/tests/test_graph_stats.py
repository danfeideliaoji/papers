from __future__ import annotations

from types import SimpleNamespace

import torch

from model import HGMVMLC_Model


def _build_local_hypergraph(threshold: float) -> torch.Tensor:
    holder = SimpleNamespace(num_views=1, device=torch.device("cpu"))
    features = [
        torch.tensor(
            [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]],
            dtype=torch.float32,
        )
    ]
    graphs = HGMVMLC_Model.construct_test_hyperedge_index(
        holder, features, k=2, thr=threshold
    )
    return graphs[0]


def test_author_99_has_only_one_self_loop_per_node() -> None:
    from experiments.validation_threshold.graph_stats import (
        summarize_local_hypergraph,
    )

    graph = _build_local_hypergraph(99.0)
    stats = summarize_local_hypergraph(graph, num_nodes=3)

    assert stats["self_loop_hyperedges"] == 3
    assert stats["non_self_loop_hyperedges"] == 0
    assert stats["memberships"] == 3


def test_minus_one_keeps_every_knn_candidate() -> None:
    from experiments.validation_threshold.graph_stats import (
        summarize_local_hypergraph,
    )

    graph = _build_local_hypergraph(-1.0)
    stats = summarize_local_hypergraph(graph, num_nodes=3)

    assert stats["non_self_loop_hyperedges"] == 3
    assert stats["non_self_loop_memberships"] == 9
    assert stats["max_hyperedge_size"] == 3


def test_graph_summary_reports_density_and_mean_size() -> None:
    from experiments.validation_threshold.graph_stats import (
        summarize_local_hypergraph,
    )

    stats = summarize_local_hypergraph(_build_local_hypergraph(-1.0), num_nodes=3)

    assert stats["hyperedges"] == 6
    assert stats["non_self_loop_ratio"] == 0.5
    assert stats["mean_hyperedge_size"] == 2.0
