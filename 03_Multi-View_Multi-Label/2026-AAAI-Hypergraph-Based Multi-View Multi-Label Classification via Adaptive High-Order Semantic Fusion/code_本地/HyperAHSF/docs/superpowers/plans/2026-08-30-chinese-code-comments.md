# 科研项目中文代码注释 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 为仓库内全部 Python 源码补充准确、详细的中文注释和文档字符串，同时保证算法、接口、控制流和运行结果不变。

**Architecture:** 按“主训练流程、候选实验、测试”三个边界分批注释。每个模块说明职责，每个类和函数说明参数、返回值、张量形状与副作用，关键算法块解释数据流；不为显而易见的赋值机械添加注释，不借机修复或重构现有实现。

**Tech Stack:** Python、PyTorch、PyTorch Geometric、NumPy、scikit-learn、unittest。

---

### Task 1: 主训练流程中文注释

**Files:**
- Modify: `data_preprocessor.py`
- Modify: `model.py`
- Modify: `train.py`
- Modify: `metrics.py`
- Modify: `utils.py`

- [x] **Step 1: 补充模块级职责说明**

  为五个文件增加中文模块文档字符串，说明它们在数据加载、超图构造、模型传播、训练与评估中的位置。

- [x] **Step 2: 补充公开结构说明**

  为所有类和函数增加中文文档字符串，至少覆盖参数、返回值、关键形状与状态修改。

- [x] **Step 3: 注释核心算法数据流**

  在标准化、K 折划分、标签引导构图、测试构图、节点到超边、超边到节点、公共/特异分支、对比损失和指标计算处增加中文行内注释。

- [x] **Step 4: 验证主流程语法**

  Run: `python -m py_compile data_preprocessor.py model.py train.py metrics.py utils.py`

  Expected: 退出码为 0，且不输出 SyntaxError。

### Task 2: Candidate 1 实验代码中文注释

**Files:**
- Modify: `experiments/candidate1/candidate_model.py`
- Modify: `experiments/candidate1/soft_graph.py`
- Modify: `experiments/candidate1/diagnose_graphs.py`
- Modify: `experiments/candidate1/run_experiments.py`
- Modify: `experiments/candidate1/summarize_results.py`

- [x] **Step 1: 说明候选实验模块边界**

  增加模块文档字符串，明确候选模型、软超图、图诊断、实验运行和结果汇总的职责。

- [x] **Step 2: 注释数据结构与核心算法**

  为 dataclass、模型类、软图构造器、EMA、温度校准、可靠性/冲突权重、诊断统计和报告生成函数补充中文说明。

- [x] **Step 3: 注释命令行入口与产物**

  说明参数来源、输出目录、CSV/JSON/SVG/Markdown 产物和各实验变体的含义。

- [x] **Step 4: 验证实验代码语法**

  Run: `python -m py_compile experiments/candidate1/*.py`

  Expected: 退出码为 0，且不输出 SyntaxError。

### Task 3: 测试代码中文注释

**Files:**
- Modify: `experiments/candidate1/tests/test_candidate_model.py`
- Modify: `experiments/candidate1/tests/test_diagnostics.py`
- Modify: `experiments/candidate1/tests/test_runner.py`
- Modify: `experiments/candidate1/tests/test_soft_graph.py`
- Modify: `experiments/candidate1/tests/test_summary.py`

- [x] **Step 1: 说明每组测试的保护目标**

  为测试模块、测试类和测试方法补充中文文档字符串，写清输入场景、断言对象和防止的回归。

- [x] **Step 2: 验证测试文件语法**

  Run: `python -m py_compile experiments/candidate1/tests/*.py`

  Expected: 退出码为 0，且不输出 SyntaxError。

### Task 4: 全量行为与覆盖核查

**Files:**
- Verify: all Python files listed above

- [x] **Step 1: 运行全量语法编译**

  Run: `python -m compileall -q .`

  Expected: 退出码为 0。

- [x] **Step 2: 运行 Candidate 1 单元测试**

  Run: `E:\project\cuda\.venv\Scripts\python.exe -m unittest discover -s experiments\candidate1\tests -v`

  Expected: 所有可运行测试通过；若环境缺少项目依赖，准确记录缺失项而不掩盖。

  Result: 命令已执行。诊断与汇总的 3 个测试通过；其余测试模块在导入时
  分别因缺少 `torch_geometric`、`torch_scatter` 和 `yaml` 报错，未进入测试体。

- [x] **Step 3: 检查注释覆盖和意外代码修改**

  检查每个模块是否具有中文模块说明、每个类/函数是否具有文档字符串，并确认修改仅包含注释、文档字符串与必要的空行。
