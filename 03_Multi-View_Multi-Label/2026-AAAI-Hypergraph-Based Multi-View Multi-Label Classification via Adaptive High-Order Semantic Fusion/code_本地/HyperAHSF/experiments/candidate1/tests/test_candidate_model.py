"""Candidate 1 模型、EMA 和温度校准的公开行为测试。"""

import copy
import unittest

import torch
import torch.nn.functional as F

from experiments.candidate1.candidate_model import (
    Candidate1Model,
    fit_temperatures,
    update_ema,
)
from experiments.candidate1.soft_graph import SoftHypergraphBuilder


PARAMS = {
    # 使用很小但维度不同的两个视图，覆盖多视图映射和拼接逻辑。
    "num_views": 2,
    "inputs_sizes": [4, 3],
    "num_class": 3,
    "EDC_feature_matrix": [[5], [6]],
    "specific_dim_matrix": [[8], [8]],
    "num_layers": 2,
    "hid_dim": 8,
}


class CandidateModelPublicBehaviorTests(unittest.TestCase):
    """保护模型输出契约、梯度隔离、教师更新和校准边界。"""

    def setUp(self):
        """为每个测试创建可复现的双视图特征和二值多标签。"""
        torch.manual_seed(11)
        self.features = [torch.randn(9, 4), torch.randn(9, 3)]
        self.labels = torch.randint(0, 2, (9, 3)).float()

    def test_forward_accepts_graph_bundle_and_returns_expected_shapes(self):
        """模型应接受完整图包，并返回与 N/C/H 对齐的结构化结果。"""
        model = Candidate1Model(PARAMS)
        embeddings, view_logits = model.encode_for_graph(self.features)
        probabilities = [torch.sigmoid(logits.detach()) for logits in view_logits]
        graph = SoftHypergraphBuilder(k=3).build(
            self.features, [z.detach() for z in embeddings], probabilities, torch.ones(2)
        )
        output = model(self.features, graph)
        self.assertEqual(output.main_logits.shape, (9, 3))
        self.assertEqual(len(output.view_logits), 2)
        self.assertEqual(output.contrastive_representation.shape, (9, 8))

    def test_view_head_loss_does_not_backpropagate_into_encoders(self):
        """detach 开启时，视图辅助损失只能更新预测头，不能更新编码器。"""
        model = Candidate1Model(PARAMS)
        _, view_logits = model.encode_for_graph(self.features, detach_for_view_heads=True)
        loss = sum(F.binary_cross_entropy_with_logits(logits, self.labels) for logits in view_logits)
        loss.backward()
        self.assertTrue(any(p.grad is not None for p in model.view_predictors.parameters()))
        self.assertTrue(all(p.grad is None for p in model.encoders.parameters()))

    def test_ema_has_no_gradients_and_moves_toward_student(self):
        """EMA 更新后教师应更接近学生，同时继续保持全部参数冻结。"""
        student = Candidate1Model(PARAMS)
        teacher = copy.deepcopy(student)
        with torch.no_grad():
            for parameter in student.parameters():
                parameter.add_(1.0)
        before = next(teacher.parameters()).detach().clone()
        update_ema(teacher, student, decay=0.5)
        after = next(teacher.parameters()).detach().clone()
        target = next(student.parameters()).detach()
        self.assertTrue(torch.norm(target - after) < torch.norm(target - before))
        self.assertTrue(all(not p.requires_grad for p in teacher.parameters()))

    def test_temperature_fit_is_bounded_and_does_not_increase_bce(self):
        """网格温度必须落在边界内，且训练 BCE 不应高于 T=1。"""
        logits = [torch.randn(30, 3) * 3.0, torch.randn(30, 3) * 2.0]
        labels = torch.randint(0, 2, (30, 3)).float()
        before = sum(F.binary_cross_entropy_with_logits(x, labels) for x in logits)
        temperatures = fit_temperatures(logits, labels)
        after = sum(
            F.binary_cross_entropy_with_logits(x / temperatures[v], labels)
            for v, x in enumerate(logits)
        )
        self.assertTrue(((temperatures >= 0.5) & (temperatures <= 5.0)).all())
        self.assertLessEqual(float(after), float(before) + 1e-5)


if __name__ == "__main__":
    unittest.main()
