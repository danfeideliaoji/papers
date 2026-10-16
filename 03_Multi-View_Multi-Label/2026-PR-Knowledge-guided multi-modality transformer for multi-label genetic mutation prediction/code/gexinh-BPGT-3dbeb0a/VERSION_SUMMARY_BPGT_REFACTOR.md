# BPGT Refactor Version Summary

## Version Goal

本版本目标是把裁剪后的 BPGT 代码整理成可运行、命名与论文一致的实验版本。核心对齐论文中的 Biological-knowledge-enhanced PathGenomic multi-label Transformer (BPGT)，并明确三块模块：

- Visual Extractor (VE): 从 WSI patch features 中提取视觉特征。
- Gene Encoder (GE): 通过 Gene Graph (GG) 和 Knowledge Association Module (KAM) 建模 gene priors。
- Label Decoder (LD): 融合 visual features 和 gene priors，输出 multi-label mutation prediction。

## Differences From Previous Version

- 新建独立目录 `BPGT_refactor`，保留原始 `BPGT` checkout 不变。
- 修复 `model/model_exp.py` 的语法错误、缺失 `Geneformer` 导入，以及旧实验名与论文命名不一致的问题。
- 新增 `BPGTModel`，保留旧入口 `model_markets = BPGTModel` 以兼容现有训练代码。
- 将旧的 `geneformer`/`graphormer` 别名收束到正式 `bpgt` graph module。
- 从完整库 `/data/gexinh/omics` 补回必要依赖：`base_model.py`、`ml_decoder.py`、`graphormer.py`、`file_utils.py`、`eval_utils.py`、`utils_dino.py`、`wsi_core/`、`data/graph/`、`data/list/`、`data/pathways/`。
- 补充干净的 `model/loss.py`，替代完整库中损坏/二进制状态的同名文件。
- 将 comparative multi-label loss 整理为 `ComparativeMultiLabelLoss`，保留 `comparative_multi_label_loss` 和 `TwoWayLoss` 兼容名。
- 修复 `[3]_main.py` 中缺失的 `modify_path` 生成逻辑，自动添加 `non_mutation` 标签。
- 修复 adjacency 与 label embedding 的节点数对齐，为 `non_mutation` 添加额外图节点。
- 修复 graph/list/split 路径不一致问题，兼容 `splits/tcga_100` 和 `tcga_100` 两种布局。
- 修复 `exp_script.py` 中默认 `graph_module` 非法、`args.traverse` 未定义、`inter_region` 未定义等入口问题。
- 修复旧版 `TransformerDecoderLayerOptimal` 与新版 PyTorch `nn.TransformerDecoder` 的兼容问题，改为本地 `TransformerDecoderWithAttention` wrapper，稳定返回 hidden states 和 cross-attention。
- 将 `h5py`、`kmeans_pytorch`、`nystrom_attention` 等分支依赖改成按需检查，避免未使用分支在 import 阶段阻塞主实验。
- 去除 `exp_script.py` 的调试 PID 输出，保持实验脚本和 `--help` 输出干净。
- 按论文 Table 3 的实验设置补齐 visual extractor backbone 入口：`genehe-ve`、`deephis-ve`、`att-ve`、`trans-ve`、`kat`、`hipt`。
- 接入 KAT backbone：选择 `model=kat` 时自动使用 H5 feature/coords，并在模型中根据 patch coordinates 构造 kernel relative distance 和 mask。
- 补回 HIPT backbone 的 feature extraction 代码：`[0][3]_extract_feature_hipt.py` 和 `model/HIPT_4K/*.py`，但不纳入大 checkpoint/notebook/cache 文件。
- 将 `exp_script.py --traverse` 调整为论文式 ablation grid：6 个 visual extractor backbones × 5 个 graph modules。
- 重写 README 为论文开源仓库格式，并去除 `News`、`Notes` 等临时段落。
- 清理冗余项：`__pycache__`、`.pyc`、`model/model.rar`、嵌套重复目录 `model/model/`。

## Expected Experimental Results

- 预期 BPGT 主线配置 `graph_module=bpgt, model=trans_mil, word_embedd_model=biobert` 能正确构建模型，并使用论文中的 gene priors + visual features + label decoder 流程训练。
- 相比上个裁剪版本，本版本应首先提升代码可运行性：核心模块 import/compile 不再因为缺失文件、旧名、语法错误、路径错误而失败。
- 由于恢复了知识图谱、GeneCard embedding 和 comparative multi-label loss，预期实验行为更接近论文实现，尤其是在 multi-label mutation prediction、rare mutation 和 co-mutation pattern 建模上更稳定。
- 真实性能仍需在完整 patch feature 目录存在后运行训练验证；当前仓库示例数据包含 CSV/splits/knowledge files，但不包含完整 `.pt` patch feature bags。

## Validation

- 使用 `/data/gexinh/miniconda3/envs/trident/bin/python` 完成 `py_compile`，共检查 46 个 Python 文件。
- 完成 `BPGTModel` 最小前向验证：`agg_method=att`、`agg_method=trans_mil`、`agg_method=abmil` 均输出 `(1, num_classes)` logits。
- 完成论文 backbone 最小前向验证：`genehe-ve`、`deephis-ve`、`att-ve`、`trans-ve`、`kat`、`hipt` 均输出 `(1, num_classes)` logits。
- 完成 `ComparativeMultiLabelLoss` smoke test，loss 为有限值。
- 完成 `[3]_main.py --help`、`exp_script.py --help` 和核心入口 import 检查。
- 尚未运行完整训练；原因是当前仓库缺少完整 WSI `.pt` patch feature bags。
