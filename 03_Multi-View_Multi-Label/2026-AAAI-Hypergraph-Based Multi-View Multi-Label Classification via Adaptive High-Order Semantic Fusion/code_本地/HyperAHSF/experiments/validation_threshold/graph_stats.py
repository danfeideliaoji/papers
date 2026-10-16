"""验证局部超图的结构统计。"""

from __future__ import annotations

import torch


def summarize_local_hypergraph(
    hyperedge_index: torch.Tensor,
    num_nodes: int,
) -> dict[str, int | float]:
    """统计作者构图约定下的自环、动态超边和成员关系数量。"""

    edge_ids = hyperedge_index[1]
    num_edges = int(edge_ids.max().item()) + 1 if edge_ids.numel() else 0
    self_loop_edges = min(int(num_nodes), num_edges)
    sizes = torch.bincount(edge_ids, minlength=num_edges)
    dynamic_sizes = sizes[self_loop_edges:]
    return {
        "hyperedges": num_edges,
        "self_loop_hyperedges": self_loop_edges,
        "non_self_loop_hyperedges": num_edges - self_loop_edges,
        "memberships": int(hyperedge_index.shape[1]),
        "non_self_loop_memberships": int(dynamic_sizes.sum().item()),
        "non_self_loop_ratio": (
            (num_edges - self_loop_edges) / num_edges if num_edges else 0.0
        ),
        "mean_hyperedge_size": (
            float(sizes.to(torch.float32).mean().item()) if sizes.numel() else 0.0
        ),
        "max_hyperedge_size": int(sizes.max().item()) if sizes.numel() else 0,
    }
