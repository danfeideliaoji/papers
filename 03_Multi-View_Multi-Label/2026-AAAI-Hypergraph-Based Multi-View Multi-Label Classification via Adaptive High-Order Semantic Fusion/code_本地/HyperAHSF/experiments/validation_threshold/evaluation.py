"""按阈值隔离验证图缓存，并保护共享模型参数。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch


@dataclass(frozen=True)
class ThresholdEvaluation:
    """一次阈值评估产生的预测与实际使用的超图。"""

    logits: torch.Tensor
    local_hyperedges: tuple[torch.Tensor, ...]
    unified_hyperedge_index: torch.Tensor


def evaluate_threshold(
    model: torch.nn.Module,
    features: Sequence[torch.Tensor],
    threshold: float,
) -> ThresholdEvaluation:
    """清空缓存后按指定阈值评估，并拒绝任何参数变化。"""

    parameter_ids = tuple(id(parameter) for parameter in model.parameters())
    parameter_values = tuple(
        parameter.detach().clone() for parameter in model.parameters()
    )

    model.test_hyperedge_index = None
    model.eval_hyperedge_index = None
    model.test_thr = float(threshold)
    model.eval()
    with torch.no_grad():
        logits, _, _ = model.evaluate(features)

    if parameter_ids != tuple(id(parameter) for parameter in model.parameters()):
        raise RuntimeError("阈值评估替换了模型参数对象。")
    for before, after in zip(parameter_values, model.parameters()):
        if not torch.equal(before, after.detach()):
            raise RuntimeError("阈值评估修改了模型参数值。")

    if model.test_hyperedge_index is None or model.eval_hyperedge_index is None:
        raise RuntimeError("模型评估结束后没有生成完整验证超图。")
    return ThresholdEvaluation(
        logits=logits.detach(),
        local_hyperedges=tuple(model.test_hyperedge_index),
        unified_hyperedge_index=model.eval_hyperedge_index,
    )
