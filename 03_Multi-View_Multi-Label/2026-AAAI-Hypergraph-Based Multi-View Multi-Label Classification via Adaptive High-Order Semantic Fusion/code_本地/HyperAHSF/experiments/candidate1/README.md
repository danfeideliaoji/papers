# Candidate 1 独立实验

本目录实现“校准后验驱动的软超图关联”，不修改原始 HyperAHSF 模型、训练入口或配置。

## 入口

- `diagnose_graphs.py`：五折原始构图审计，输出 CSV、JSON、SVG 和诊断报告。
- `run_experiments.py`：运行 Candidate 1、消融、feature-only 和真标签 oracle。
- `summarize_results.py`：聚合原始基线与全部新实验，生成最终研究判断。
- `tests/`：软图、加权传播、EMA、温度校准、诊断和训练冒烟测试。

## 已执行配置

- 数据：Emotions 原五折划分；
- 训练：5 折 × 125 epoch，随机种子 0，CPU；
- warm-up：10 epoch；图刷新周期：5 epoch；K=6；
- 变体：`candidate1`、`feature_only`、`no_semantic`、`no_reliability`、`no_conflict`、`oracle`。

全部结果位于 `research_outputs/candidate1/`。`oracle` 使用验证真标签，只能作为诊断上界。

## 复现命令

```powershell
& '..\..\..\.venv\Scripts\python.exe' -m unittest discover -s experiments\candidate1\tests -v
& '..\..\..\.venv\Scripts\python.exe' -m experiments.candidate1.diagnose_graphs
& '..\..\..\.venv\Scripts\python.exe' -m experiments.candidate1.run_experiments --variants candidate1 feature_only no_semantic no_reliability no_conflict oracle
& '..\..\..\.venv\Scripts\python.exe' -m experiments.candidate1.summarize_results
```
