from __future__ import annotations

import torch


class FakeModel(torch.nn.Module):
    """只暴露真实评估器依赖的公开模型接口。"""

    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([2.0]))
        self.test_thr = 99.0
        self.test_hyperedge_index = [torch.tensor([[9], [9]])]
        self.eval_hyperedge_index = torch.tensor([[9], [9]])
        self.seen_thresholds: list[float] = []

    def evaluate(self, features: list[torch.Tensor]):
        assert self.test_hyperedge_index is None
        assert self.eval_hyperedge_index is None
        self.seen_thresholds.append(self.test_thr)
        graph = torch.tensor([[0], [0]])
        self.test_hyperedge_index = [graph]
        self.eval_hyperedge_index = graph
        return features[0] * self.weight + self.test_thr, (), torch.empty(0)


def test_evaluation_clears_both_caches_and_preserves_parameters() -> None:
    from experiments.validation_threshold.evaluation import evaluate_threshold

    model = FakeModel()
    parameter_id = id(model.weight)
    parameter_before = model.weight.detach().clone()

    evaluated = evaluate_threshold(model, [torch.ones(1)], threshold=0.9)

    assert model.seen_thresholds == [0.9]
    assert id(model.weight) == parameter_id
    torch.testing.assert_close(model.weight, parameter_before)
    torch.testing.assert_close(evaluated.logits, torch.tensor([2.9]))


def test_threshold_order_does_not_change_same_threshold_prediction() -> None:
    from experiments.validation_threshold.evaluation import evaluate_threshold

    features = [torch.ones(1)]
    first_model = FakeModel()
    first = {
        threshold: evaluate_threshold(first_model, features, threshold).logits.clone()
        for threshold in (0.9, -1.0)
    }
    second_model = FakeModel()
    second = {
        threshold: evaluate_threshold(second_model, features, threshold).logits.clone()
        for threshold in (-1.0, 0.9)
    }

    torch.testing.assert_close(first[0.9], second[0.9])
    torch.testing.assert_close(first[-1.0], second[-1.0])
