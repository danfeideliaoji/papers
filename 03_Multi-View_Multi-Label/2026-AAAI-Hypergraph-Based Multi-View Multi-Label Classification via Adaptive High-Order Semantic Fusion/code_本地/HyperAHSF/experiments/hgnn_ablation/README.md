# HyperAHSF 独立 HGNN 消融实验

本目录提供统一的 Emotions 五折实验运行器、指标保存、计时、显存统计、参数统计和曲线生成。公共分支 `experiment/hgnn-ablation-base` 只允许运行 `baseline`；三个单机制变体必须在各自独立分支运行。

## 运行环境

本机实验复用已有 CUDA PyTorch 解释器，并把缺失的普通 Python 库放在被 Git 忽略的 `.deps/`：

```powershell
$env:PYTHONPATH = "$PWD\.deps"
$python = 'E:\project\cuda\.venv\Scripts\python.exe'
& $python -c "import torch, torch_geometric, sklearn, yaml, matplotlib; print(torch.__version__, torch.cuda.is_available())"
```

不得把 `.deps/`、数据集、checkpoint、缓存或日志加入 Git。

## 公共基线

完整五折：

```powershell
$env:PYTHONPATH = "$PWD\.deps"
& 'E:\project\cuda\.venv\Scripts\python.exe' -m experiments.hgnn_ablation.runner --variant baseline --folds 0 1 2 3 4 --epochs 125 --seed 0 --device cuda:0
```

CPU 单折检查：

```powershell
$env:PYTHONPATH = "$PWD\.deps"
& 'E:\project\cuda\.venv\Scripts\python.exe' -m experiments.hgnn_ablation.runner --variant baseline --folds 0 --epochs 1 --seed 0 --device cpu --output-root research_outputs/hgnn_ablation_cpu_check
```

## 测试

```powershell
$env:PYTHONPATH = "$PWD\.deps"
& 'E:\project\cuda\.venv\Scripts\python.exe' -m pytest experiments/candidate1/tests experiments/hgnn_ablation/tests -q
```

运行器拒绝覆盖任何非空输出目录。若一次正式运行已经存在，先检查其完整性并换用新的输出根目录；不得删除或覆盖旧结果。

## 固定输出

每个正式变体目录包含：

- `config.json`
- `metrics.csv`
- `fold_summary.csv`
- `timing.json`
- `parameter_count.json`
- `learning_curves.svg`
- `单机制实验报告.md`

`metrics.csv` 保留项目原有八项指标，并使用明确列名：Hamming Loss、Average Precision、One Error、Ranking Loss、Subset Accuracy、Coverage、Macro-F1、Micro-F1。
