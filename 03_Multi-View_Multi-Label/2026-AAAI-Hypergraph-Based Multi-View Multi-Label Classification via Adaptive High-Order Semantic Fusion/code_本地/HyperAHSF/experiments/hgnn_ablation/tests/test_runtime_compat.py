from __future__ import annotations

import subprocess
import sys

import pytest
import torch


def test_model_imports_without_optional_torch_scatter_package() -> None:
    completed = subprocess.run(
        [sys.executable, "-c", "import model"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr


def test_message_passing_preserves_original_mean_aggregation() -> None:
    try:
        from model import HG_messagepassing
    except ModuleNotFoundError as exc:
        pytest.fail(f"model cannot be imported in the supported runtime: {exc}")

    layer = HG_messagepassing(
        in_dim=4,
        hid_dim=4,
        out_dim=4,
        hyperedge_weight=None,
        act=torch.nn.Identity(),
    )
    with torch.no_grad():
        layer.lin_n2e.weight.copy_(torch.eye(4))
        layer.bias_n2e.zero_()

    x = torch.tensor(
        [[1.0, 3.0, 5.0, 7.0], [3.0, 5.0, 7.0, 9.0], [5.0, 7.0, 9.0, 11.0]]
    )
    hyperedge_index = torch.tensor([[0, 1, 1, 2], [0, 0, 1, 1]])

    nodes, edges = layer(x, hyperedge_index, num_nodes=3, num_edges=2)

    expected_edges = torch.tensor([[2.0, 4.0, 6.0, 8.0], [4.0, 6.0, 8.0, 10.0]])
    expected_nodes = torch.tensor(
        [[2.0, 4.0, 6.0, 8.0], [3.0, 5.0, 7.0, 9.0], [4.0, 6.0, 8.0, 10.0]]
    )
    torch.testing.assert_close(edges, expected_edges, rtol=0, atol=0)
    torch.testing.assert_close(nodes, expected_nodes, rtol=0, atol=0)
