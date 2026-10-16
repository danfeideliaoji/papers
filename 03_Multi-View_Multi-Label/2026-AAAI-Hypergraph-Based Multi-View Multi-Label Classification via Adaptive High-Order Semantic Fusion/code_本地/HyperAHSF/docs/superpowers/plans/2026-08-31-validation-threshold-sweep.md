# Validation Threshold Sweep Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在不改变作者原始 `test_thr: 99`、训练过程、模型权重和数据划分的前提下，用同一训练权重诊断五个验证余弦阈值对 Emotions 超图结构与八项验证指标的影响。

**Architecture:** 新建独立的 `experiments/validation_threshold` 包。运行器每折只训练一个 baseline 模型；每个 epoch 更新完权重后，依次清空两级验证图缓存、设置一个阈值、在 `eval/no_grad` 下预测并记录局部图统计。配置、评估隔离、图统计、结果汇总分别放在小文件中，原始 `config.yaml` 和 `train.py` 保持不变。

**Tech Stack:** Python 3、PyTorch、PyTorch Geometric、NumPy、scikit-learn、PyYAML、matplotlib、pytest、Git。

---

## 文件结构

- Create: `experiments/validation_threshold/__init__.py` — 实验包入口与公共常量导出。
- Create: `experiments/validation_threshold/configuration.py` — 五个固定阈值及运行配置校验。
- Create: `experiments/validation_threshold/evaluation.py` — 缓存清理、阈值评估、参数不变性检查。
- Create: `experiments/validation_threshold/graph_stats.py` — 局部超图自环/动态边/成员数统计。
- Create: `experiments/validation_threshold/reporting.py` — CSV、聚合、曲线和中文诊断报告。
- Create: `experiments/validation_threshold/runner.py` — 单模型训练与五阈值循环、CLI。
- Create: `experiments/validation_threshold/tests/test_configuration.py` — 阈值与运行配置测试。
- Create: `experiments/validation_threshold/tests/test_evaluation.py` — 缓存、参数、顺序隔离测试。
- Create: `experiments/validation_threshold/tests/test_graph_stats.py` — 99/-1 边界和统计测试。
- Create: `experiments/validation_threshold/tests/test_reporting.py` — 指标列与五折聚合测试。
- Create: `experiments/validation_threshold/tests/test_runner.py` — 一折一 epoch 真实集成测试。
- Create after experiments: `research_outputs/validation_threshold_sweep/*` — 冒烟与正式实验产物；只暂存小型 CSV/JSON/SVG/Markdown。

### Task 1: 固定阈值配置与 CLI 数据模型

**Files:**
- Create: `experiments/validation_threshold/__init__.py`
- Create: `experiments/validation_threshold/configuration.py`
- Create: `experiments/validation_threshold/tests/__init__.py`
- Create: `experiments/validation_threshold/tests/test_configuration.py`

- [ ] **Step 1: 写阈值列表和配置校验的失败测试**

```python
from pathlib import Path
import pytest

from experiments.validation_threshold.configuration import (
    DEFAULT_THRESHOLDS,
    SweepConfig,
)


def test_default_thresholds_preserve_author_value_and_order() -> None:
    assert [(item.name, item.value) for item in DEFAULT_THRESHOLDS] == [
        ("author_99", 99.0),
        ("cos_099", 0.99),
        ("cos_095", 0.95),
        ("cos_090", 0.90),
        ("knn_all", -1.0),
    ]


def test_sweep_defaults_to_five_folds_and_125_epochs(tmp_path: Path) -> None:
    config = SweepConfig(project_root=tmp_path, output_dir=tmp_path / "out")
    assert config.folds == (0, 1, 2, 3, 4)
    assert config.epochs == 125


def test_sweep_rejects_duplicate_names_and_values(tmp_path: Path) -> None:
    threshold_type = type(DEFAULT_THRESHOLDS[0])
    with pytest.raises(ValueError, match="唯一"):
        SweepConfig(
            project_root=tmp_path,
            output_dir=tmp_path / "out",
            thresholds=(threshold_type("a", 0.9), threshold_type("a", 0.8)),
        )
```

- [ ] **Step 2: 运行测试并确认因模块不存在而失败**

Run: `python -m pytest experiments/validation_threshold/tests/test_configuration.py -v`

Expected: FAIL，提示 `experiments.validation_threshold` 不存在。

- [ ] **Step 3: 实现不可变阈值和运行配置**

```python
@dataclass(frozen=True)
class ThresholdVariant:
    name: str
    value: float


DEFAULT_THRESHOLDS = (
    ThresholdVariant("author_99", 99.0),
    ThresholdVariant("cos_099", 0.99),
    ThresholdVariant("cos_095", 0.95),
    ThresholdVariant("cos_090", 0.90),
    ThresholdVariant("knn_all", -1.0),
)


@dataclass(frozen=True)
class SweepConfig:
    project_root: Path
    output_dir: Path
    folds: tuple[int, ...] = (0, 1, 2, 3, 4)
    epochs: int = 125
    seed: int = 0
    device: str = "cuda:0"
    thresholds: tuple[ThresholdVariant, ...] = DEFAULT_THRESHOLDS

    def __post_init__(self) -> None:
        if not self.folds or any(fold not in range(5) for fold in self.folds):
            raise ValueError("fold 必须是 0..4 中至少一个编号。")
        if self.epochs <= 0:
            raise ValueError("epochs 必须为正整数。")
        names = [item.name for item in self.thresholds]
        values = [item.value for item in self.thresholds]
        if len(names) != len(set(names)) or len(values) != len(set(values)):
            raise ValueError("阈值名称和值必须唯一。")
```

- [ ] **Step 4: 运行配置测试并确认通过**

Run: `python -m pytest experiments/validation_threshold/tests/test_configuration.py -v`

Expected: PASS。

- [ ] **Step 5: 提交配置骨架**

```powershell
git add experiments/validation_threshold/__init__.py experiments/validation_threshold/configuration.py experiments/validation_threshold/tests/__init__.py experiments/validation_threshold/tests/test_configuration.py
git commit -m "test: define validation threshold sweep matrix"
```

### Task 2: 图结构统计与阈值边界证明

**Files:**
- Create: `experiments/validation_threshold/graph_stats.py`
- Create: `experiments/validation_threshold/tests/test_graph_stats.py`

- [ ] **Step 1: 写 99、-1.0 和统计字段的失败测试**

```python
import torch

from experiments.validation_threshold.graph_stats import summarize_local_hypergraph
from model import HGMVMLC_Model


def _builder(threshold: float) -> torch.Tensor:
    holder = object.__new__(HGMVMLC_Model)
    holder.num_views = 1
    holder.device = torch.device("cpu")
    features = [torch.tensor([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]])]
    return holder.construct_test_hyperedge_index(features, k=2, thr=threshold)[0]


def test_author_99_has_only_one_self_loop_per_node() -> None:
    graph = _builder(99.0)
    stats = summarize_local_hypergraph(graph, num_nodes=3)
    assert stats["self_loop_hyperedges"] == 3
    assert stats["non_self_loop_hyperedges"] == 0
    assert stats["memberships"] == 3


def test_minus_one_keeps_every_knn_candidate() -> None:
    graph = _builder(-1.0)
    stats = summarize_local_hypergraph(graph, num_nodes=3)
    assert stats["non_self_loop_hyperedges"] == 3
    assert stats["non_self_loop_memberships"] == 9
    assert stats["max_hyperedge_size"] == 3
```

- [ ] **Step 2: 运行测试并确认统计函数缺失**

Run: `python -m pytest experiments/validation_threshold/tests/test_graph_stats.py -v`

Expected: FAIL，提示不能导入 `summarize_local_hypergraph`。

- [ ] **Step 3: 实现不依赖模型内部状态的统计函数**

```python
def summarize_local_hypergraph(index: torch.Tensor, num_nodes: int) -> dict[str, int | float]:
    edge_ids = index[1]
    edge_count = int(edge_ids.max().item()) + 1 if edge_ids.numel() else 0
    sizes = torch.bincount(edge_ids, minlength=edge_count)
    self_loop_count = min(num_nodes, edge_count)
    dynamic_sizes = sizes[num_nodes:]
    dynamic_count = int(dynamic_sizes.numel())
    dynamic_memberships = int(dynamic_sizes.sum().item())
    memberships = int(index.shape[1])
    return {
        "hyperedges": edge_count,
        "self_loop_hyperedges": self_loop_count,
        "non_self_loop_hyperedges": dynamic_count,
        "memberships": memberships,
        "non_self_loop_memberships": dynamic_memberships,
        "non_self_loop_ratio": dynamic_count / edge_count if edge_count else 0.0,
        "mean_hyperedge_size": float(sizes.float().mean().item()) if edge_count else 0.0,
        "max_hyperedge_size": int(sizes.max().item()) if edge_count else 0,
    }
```

- [ ] **Step 4: 运行图统计测试并确认通过**

Run: `python -m pytest experiments/validation_threshold/tests/test_graph_stats.py -v`

Expected: PASS，并明确证明 `99` 无动态超边、`-1.0` 保留每个锚点的全部 KNN 候选。

- [ ] **Step 5: 提交图统计**

```powershell
git add experiments/validation_threshold/graph_stats.py experiments/validation_threshold/tests/test_graph_stats.py
git commit -m "test: prove validation threshold graph boundaries"
```

### Task 3: 隔离式阈值评估

**Files:**
- Create: `experiments/validation_threshold/evaluation.py`
- Create: `experiments/validation_threshold/tests/test_evaluation.py`

- [ ] **Step 1: 写缓存清理、参数不变和顺序不变测试**

```python
import torch

from experiments.validation_threshold.evaluation import evaluate_threshold


class FakeModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([2.0]))
        self.test_thr = 99.0
        self.test_hyperedge_index = [torch.tensor([[9], [9]])]
        self.eval_hyperedge_index = torch.tensor([[9], [9]])
        self.seen = []

    def evaluate(self, features):
        assert self.test_hyperedge_index is None
        assert self.eval_hyperedge_index is None
        self.seen.append(self.test_thr)
        graph = torch.tensor([[0], [0]])
        self.test_hyperedge_index = [graph]
        self.eval_hyperedge_index = graph
        return features[0] * self.weight + self.test_thr, (), torch.empty(0)


def test_evaluation_clears_both_caches_and_preserves_parameters() -> None:
    model = FakeModel()
    before = model.weight.detach().clone()
    output = evaluate_threshold(model, [torch.ones(1)], threshold=0.9)
    assert model.seen == [0.9]
    torch.testing.assert_close(model.weight, before)
    assert output.logits.item() == 2.9


def test_threshold_order_does_not_change_same_threshold_prediction() -> None:
    features = [torch.ones(1)]
    first = FakeModel()
    a = {t: evaluate_threshold(first, features, t).logits.clone() for t in (0.9, -1.0)}
    second = FakeModel()
    b = {t: evaluate_threshold(second, features, t).logits.clone() for t in (-1.0, 0.9)}
    torch.testing.assert_close(a[0.9], b[0.9])
    torch.testing.assert_close(a[-1.0], b[-1.0])
```

- [ ] **Step 2: 运行测试并确认评估函数缺失**

Run: `python -m pytest experiments/validation_threshold/tests/test_evaluation.py -v`

Expected: FAIL，提示不能导入 `evaluate_threshold`。

- [ ] **Step 3: 实现缓存清理、状态切换和参数快照校验**

```python
@dataclass(frozen=True)
class ThresholdEvaluation:
    logits: torch.Tensor
    local_hyperedges: tuple[torch.Tensor, ...]
    unified_hyperedge_index: torch.Tensor


def evaluate_threshold(model, features, threshold: float) -> ThresholdEvaluation:
    parameter_ids = tuple(id(parameter) for parameter in model.parameters())
    parameter_values = tuple(parameter.detach().clone() for parameter in model.parameters())
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
    return ThresholdEvaluation(
        logits=logits.detach(),
        local_hyperedges=tuple(model.test_hyperedge_index),
        unified_hyperedge_index=model.eval_hyperedge_index,
    )
```

- [ ] **Step 4: 运行隔离测试并确认通过**

Run: `python -m pytest experiments/validation_threshold/tests/test_evaluation.py -v`

Expected: PASS；同一 `FakeModel` 的参数对象和值均未改变。

- [ ] **Step 5: 提交隔离式评估**

```powershell
git add experiments/validation_threshold/evaluation.py experiments/validation_threshold/tests/test_evaluation.py
git commit -m "feat: isolate cached validation graphs by threshold"
```

### Task 4: 指标行、聚合、曲线和中文报告

**Files:**
- Create: `experiments/validation_threshold/reporting.py`
- Create: `experiments/validation_threshold/tests/test_reporting.py`

- [ ] **Step 1: 写完整列、五阈值聚合和逐折胜负测试**

```python
from experiments.validation_threshold.reporting import metric_row, summarize_thresholds


METRICS = {"HM": 0.2, "AP": 0.7, "OE": 0.3, "RL": 0.2,
           "subset": 0.1, "coverage": 1.5, "macro": 0.55, "micro": 0.6}


def test_metric_row_contains_identity_threshold_and_eight_metrics() -> None:
    row = metric_row("cos_090", 0, 3, 0.9, 1.0, METRICS, 0.01)
    assert [row[key] for key in ("variant", "fold", "epoch", "threshold")] == [
        "cos_090", 0, 3, 0.9
    ]
    assert len([key for key in row if key in {
        "Hamming Loss", "Average Precision", "One Error", "Ranking Loss",
        "Subset Accuracy", "Coverage", "Macro-F1", "Micro-F1"
    }]) == 8


def test_summary_uses_ap_best_epoch_per_fold_and_reports_mean_std_wins() -> None:
    rows = [
        metric_row("author_99", 0, 0, 99.0, 1.0, {**METRICS, "AP": 0.70}, 0.01),
        metric_row("cos_090", 0, 0, 0.9, 1.0, {**METRICS, "AP": 0.75}, 0.01),
        metric_row("author_99", 1, 0, 99.0, 1.0, {**METRICS, "AP": 0.80}, 0.01),
        metric_row("cos_090", 1, 0, 0.9, 1.0, {**METRICS, "AP": 0.79}, 0.01),
    ]
    summary = {row["variant"]: row for row in summarize_thresholds(rows)}
    assert summary["cos_090"]["AP_wins_vs_author_99"] == 1
    assert summary["cos_090"]["AP_losses_vs_author_99"] == 1
    assert "Average Precision_std" in summary["author_99"]
```

- [ ] **Step 2: 运行测试并确认报表模块缺失**

Run: `python -m pytest experiments/validation_threshold/tests/test_reporting.py -v`

Expected: FAIL，提示不能导入新报表 API。

- [ ] **Step 3: 实现记录、聚合与输出函数**

实现时复用 `experiments.hgnn_ablation.reporting.METRIC_NAME_MAP`、`write_csv`、`write_json`，并明确：

```python
def metric_row(variant, fold, epoch, threshold, loss, metrics, epoch_seconds):
    row = {"variant": variant, "fold": int(fold), "epoch": int(epoch),
           "threshold": float(threshold), "loss": float(loss),
           "epoch_seconds": float(epoch_seconds)}
    row.update({long: float(metrics[short]) for short, long in METRIC_NAME_MAP.items()})
    return row
```

`summarize_thresholds()` 先对每个 `(variant, fold)` 选择 AP 最大的 epoch，再计算五折均值、样本标准差（仅一折时为 0）、相对 `author_99` 的 AP 胜/平/负；`plot_learning_curves()` 每个指标一张子图、每个阈值一条五折均值线；`write_report()` 展示五阈值的 AP/Micro/Macro/HL 均值±标准差、胜负、图密度和计时，并写清“事后诊断不能替代正式结果”。

- [ ] **Step 4: 运行报表测试并确认通过**

Run: `python -m pytest experiments/validation_threshold/tests/test_reporting.py -v`

Expected: PASS。

- [ ] **Step 5: 提交报表功能**

```powershell
git add experiments/validation_threshold/reporting.py experiments/validation_threshold/tests/test_reporting.py
git commit -m "feat: report validation threshold comparisons"
```

### Task 5: 单模型训练、五阈值验证运行器

**Files:**
- Create: `experiments/validation_threshold/runner.py`
- Create: `experiments/validation_threshold/tests/test_runner.py`

- [ ] **Step 1: 写一折一 epoch 真实集成失败测试**

```python
import csv
import json
from pathlib import Path

from experiments.validation_threshold.configuration import SweepConfig
from experiments.validation_threshold.runner import run_sweep


def test_one_epoch_real_sweep_writes_all_threshold_rows(tmp_path: Path) -> None:
    project_root = Path(__file__).resolve().parents[3]
    output = tmp_path / "sweep"
    result = run_sweep(SweepConfig(
        project_root=project_root,
        output_dir=output,
        folds=(0,),
        epochs=1,
        device="cpu",
    ))
    with (output / "metrics.csv").open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 5
    assert {row["variant"] for row in rows} == {
        "author_99", "cos_099", "cos_095", "cos_090", "knn_all"
    }
    assert {row["threshold"] for row in rows} == {"99.0", "0.99", "0.95", "0.9", "-1.0"}
    assert result["trained_models"] == 1
    assert (output / "graph_statistics.csv").exists()
    assert (output / "aggregate_results.csv").exists()
    with (output / "config.json").open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    assert payload["source_test_thr"] == 99
```

- [ ] **Step 2: 运行集成测试并确认运行器缺失**

Run: `python -m pytest experiments/validation_threshold/tests/test_runner.py -v`

Expected: FAIL，提示不能导入 `run_sweep`。

- [ ] **Step 3: 实现每折单模型训练和每 epoch 五阈值评估**

运行器必须遵循以下骨架：

```python
for fold in config.folds:
    fix_seed(config.seed)
    train_dataset = MyDataset_k_fold(..., "train", scalers=None)
    val_dataset = MyDataset_k_fold(..., "val", scalers=train_dataset.scalers)
    model = HGMVMLC_Model(dataset=train_dataset, params=params, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=params["lr"],
                                  weight_decay=params["weight_decay"])
    contrastive = PrecomputedContrastiveLoss(train_dataset.labels, params["w_intra"])
    for epoch in range(config.epochs):
        loss = _train_epoch(train_dataset, model, optimizer, contrastive, config.seed)
        shared_parameter_ids = tuple(id(p) for p in model.parameters())
        for threshold in config.thresholds:
            evaluated = evaluate_threshold(model, val_dataset.features, threshold.value)
            assert shared_parameter_ids == tuple(id(p) for p in model.parameters())
            metrics = dict(all_metrics(
                evaluated.logits.cpu().numpy(),
                (torch.sigmoid(evaluated.logits) > params["confidence_bar"]).long().cpu().numpy(),
                val_dataset.labels.cpu().numpy(),
            ))
            metrics_rows.append(metric_row(...))
            for view, graph in enumerate(evaluated.local_hyperedges):
                graph_rows.append({"variant": threshold.name, "fold": fold,
                                   "epoch": epoch, "threshold": threshold.value,
                                   "view": view, **summarize_local_hypergraph(
                                       graph, val_dataset.features[view].shape[0])})
```

在创建输出前调用 `prepare_output_dir(config.output_dir)` 拒绝覆盖；CLI 支持 `--folds`、`--epochs`、`--device`、`--output-dir`。`config.json` 保存源配置 `source_test_thr=99`、阈值顺序、环境和完整 params。正式产物固定为 `metrics.csv`、`graph_statistics.csv`、`aggregate_results.csv`、`config.json`、`learning_curves.svg`、`验证阈值诊断报告.md`。

- [ ] **Step 4: 运行真实一 epoch 集成测试**

Run: `python -m pytest experiments/validation_threshold/tests/test_runner.py -v`

Expected: PASS；只训练一个 fold 模型，但输出五行阈值指标。

- [ ] **Step 5: 运行新包和旧消融包全部测试**

Run: `python -m pytest experiments/validation_threshold/tests experiments/hgnn_ablation/tests -v`

Expected: 全部 PASS；旧 baseline 行为保持不变。

- [ ] **Step 6: 运行语法检查**

Run: `python -m compileall -q experiments/validation_threshold experiments/hgnn_ablation model.py metrics.py`

Expected: 退出码 0，无输出。

- [ ] **Step 7: 提交运行器**

```powershell
git add experiments/validation_threshold/runner.py experiments/validation_threshold/tests/test_runner.py
git commit -m "feat: run shared-weight validation threshold sweep"
```

### Task 6: fold0 × 20 epoch 冒烟实验

**Files:**
- Create: `research_outputs/validation_threshold_sweep_smoke/*`
- Modify if a verified defect is found: corresponding `experiments/validation_threshold/*.py` and test file

- [ ] **Step 1: 确认 CUDA、数据集和 Git 工作区边界**

Run: `python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"`

Expected: CUDA 可用时打印 `True` 与 GPU 型号；否则明确改用 CPU。随后运行 `git status --short`，确认 `Multi_View_dataset/` 与既有 `research_outputs/` 未被暂存。

- [ ] **Step 2: 运行 fold0 的 20 epoch**

Run: `python -m experiments.validation_threshold.runner --folds 0 --epochs 20 --device cuda:0 --output-dir research_outputs/validation_threshold_sweep_smoke`

Expected: 退出码 0，`metrics.csv` 恰有 100 行（1 折 × 20 epoch × 5 阈值），`graph_statistics.csv` 恰有 200 行（再乘 2 视图），无 NaN/Inf。

- [ ] **Step 3: 检查冒烟门禁**

Run: `python -c "import csv,math,pathlib; p=pathlib.Path('research_outputs/validation_threshold_sweep_smoke'); m=list(csv.DictReader((p/'metrics.csv').open(encoding='utf-8-sig'))); g=list(csv.DictReader((p/'graph_statistics.csv').open(encoding='utf-8-sig'))); assert len(m)==100 and len(g)==200; assert all(math.isfinite(float(r[k])) for r in m for k in ('Average Precision','Micro-F1','Macro-F1','Hamming Loss')); assert all(int(r['non_self_loop_hyperedges'])==0 for r in g if r['variant']=='author_99'); print('smoke gate passed')"`

Expected: `smoke gate passed`。

- [ ] **Step 4: 若门禁暴露缺陷，先补失败测试，再最小修复并重跑 Task 5 全部测试**

只修复可复现的实现错误；不得因指标高低调整阈值、训练参数或数据划分。修复后删除的只能是本任务新建且已确认路径为 `research_outputs/validation_threshold_sweep_smoke` 的失败产物，再以相同命令重跑。

- [ ] **Step 5: 提交冒烟摘要，不提交模型或数据**

```powershell
git add research_outputs/validation_threshold_sweep_smoke/config.json research_outputs/validation_threshold_sweep_smoke/aggregate_results.csv research_outputs/validation_threshold_sweep_smoke/验证阈值诊断报告.md
git commit -m "experiment: validate threshold sweep smoke run"
```

### Task 7: 五折 × 125 epoch 正式实验与最终报告

**Files:**
- Create: `research_outputs/validation_threshold_sweep/metrics.csv`
- Create: `research_outputs/validation_threshold_sweep/graph_statistics.csv`
- Create: `research_outputs/validation_threshold_sweep/aggregate_results.csv`
- Create: `research_outputs/validation_threshold_sweep/config.json`
- Create: `research_outputs/validation_threshold_sweep/learning_curves.svg`
- Create: `research_outputs/validation_threshold_sweep/验证阈值诊断报告.md`

- [ ] **Step 1: 运行正式实验**

Run: `python -m experiments.validation_threshold.runner --folds 0 1 2 3 4 --epochs 125 --device cuda:0 --output-dir research_outputs/validation_threshold_sweep`

Expected: 退出码 0，`metrics.csv` 有 3125 行，`graph_statistics.csv` 有 6250 行。

- [ ] **Step 2: 验证完整性、有限值、阈值覆盖和 99 图结构**

Run: `python -c "import csv,math,pathlib; p=pathlib.Path('research_outputs/validation_threshold_sweep'); m=list(csv.DictReader((p/'metrics.csv').open(encoding='utf-8-sig'))); g=list(csv.DictReader((p/'graph_statistics.csv').open(encoding='utf-8-sig'))); assert len(m)==3125 and len(g)==6250; assert {r['variant'] for r in m}=={'author_99','cos_099','cos_095','cos_090','knn_all'}; assert {(int(r['fold']),int(r['epoch'])) for r in m}=={(f,e) for f in range(5) for e in range(125)}; assert all(math.isfinite(float(r[k])) for r in m for k in ('Average Precision','Micro-F1','Macro-F1','Hamming Loss')); assert all(int(r['non_self_loop_hyperedges'])==0 for r in g if r['variant']=='author_99'); print('full data gate passed')"`

Expected: `full data gate passed`。

- [ ] **Step 3: 运行最终回归验证**

Run: `python -m pytest experiments/validation_threshold/tests experiments/hgnn_ablation/tests -v`

Expected: 全部 PASS。

Run: `python -m compileall -q experiments/validation_threshold experiments/hgnn_ablation model.py metrics.py`

Expected: 退出码 0，无输出。

- [ ] **Step 4: 审阅报告结论边界**

确认报告同时展示五个阈值、五折标准差、逐折 AP 胜平负、图密度和运行时间；不得把最优验证阈值称作正式修复或测试集结论。若两个视图的动态边比例明显不同，报告必须说明固定阈值的跨视图密度不可比。

- [ ] **Step 5: 精确暂存小型结果并提交**

```powershell
git add experiments/validation_threshold docs/superpowers/plans/2026-08-31-validation-threshold-sweep.md research_outputs/validation_threshold_sweep/metrics.csv research_outputs/validation_threshold_sweep/graph_statistics.csv research_outputs/validation_threshold_sweep/aggregate_results.csv research_outputs/validation_threshold_sweep/config.json research_outputs/validation_threshold_sweep/learning_curves.svg research_outputs/validation_threshold_sweep/验证阈值诊断报告.md
git status --short
git commit -m "experiment: compare validation cosine thresholds"
```

Expected: `Multi_View_dataset/`、其他既有 `research_outputs/`、checkpoint、缓存和大文件都没有进入提交。

## 自检结果

- 设计中的五个阈值、同权重、缓存隔离、参数不变、顺序不变、八项指标、图统计、先冒烟后五折、报告边界和 Git 边界均有对应任务。
- 每个代码步骤都给出了实际接口、命令与预期结果，没有未定义占位实现。
- 公共名称保持一致：`ThresholdVariant`、`SweepConfig`、`evaluate_threshold`、`ThresholdEvaluation`、`summarize_local_hypergraph`、`metric_row`、`summarize_thresholds`、`run_sweep`。
