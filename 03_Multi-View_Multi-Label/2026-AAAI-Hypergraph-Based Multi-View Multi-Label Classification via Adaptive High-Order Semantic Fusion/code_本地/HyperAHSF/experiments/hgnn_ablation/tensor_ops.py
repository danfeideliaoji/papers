"""用原生 PyTorch 提供本项目需要的分组求和与均值。"""

from __future__ import annotations

from typing import Optional

import torch


def scatter_add(
    source: torch.Tensor,
    index: torch.Tensor,
    dim: int = 0,
    dim_size: Optional[int] = None,
) -> torch.Tensor:
    """按一维 ``index`` 在第 0 维求和，等价于项目原用法。"""

    if dim != 0:
        raise ValueError("HyperAHSF 的 scatter_add 仅支持 dim=0。")
    if index.ndim != 1 or index.numel() != source.shape[0]:
        raise ValueError("index 必须是一维且长度等于 source 的第 0 维。")
    size = int(dim_size) if dim_size is not None else (
        int(index.max().item()) + 1 if index.numel() else 0
    )
    output = source.new_zeros((size, *source.shape[1:]))
    if index.numel():
        output.index_add_(0, index, source)
    return output


def scatter_mean(
    source: torch.Tensor,
    index: torch.Tensor,
    dim: int = 0,
    dim_size: Optional[int] = None,
) -> torch.Tensor:
    """按一维 ``index`` 在第 0 维求均值，空分组保持全零。"""

    summed = scatter_add(source, index, dim=dim, dim_size=dim_size)
    counts = scatter_add(
        torch.ones(index.shape[0], dtype=source.dtype, device=source.device),
        index,
        dim=0,
        dim_size=summed.shape[0],
    ).clamp_min_(1)
    return summed / counts.reshape((-1,) + (1,) * (source.ndim - 1))
