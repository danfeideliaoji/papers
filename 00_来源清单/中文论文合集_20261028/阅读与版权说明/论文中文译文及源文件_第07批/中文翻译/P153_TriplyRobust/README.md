# P153 半监督三重稳健归纳迁移学习

已生成：`P153_TriplyRobust_中文全文译文.pdf`（23页）。本目录仅用于 dot 云端交付，未发布至外部网站或 GitHub，原仓库论文未改动。

## 版本与许可

- 原文：Tianxi Cai、Mengyan Li、Molei Liu，*Semi-supervised Triply Robust Inductive Transfer Learning*。
- 精确版本：[arXiv:2209.04977v2](https://arxiv.org/abs/2209.04977v2)，2024-10-22，共35页。
- 官方 PDF 与仓库 PDF 字节一致，SHA-256：`513c0b5e5c0da8b7ce398b90b53cad6585c450df31aa966d80e960282332407a`。
- 官方版本页明确链接 [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)。保存了版本页 HTML、官方 PDF、TeX 归档与逐项权利核查。
- 唯一原图为系数估计柱状图，已目检，无第三方照片、权利排除或单独署名。
- 中文译文为非官方翻译及重新排版，采用 CC BY 4.0；未暗示作者认可。

## 内容覆盖

完整翻译指定35页文件：摘要、六节正文、算法1、9个编号公式、3个定理、5条备注、4个表格、唯一图、致谢、披露声明及43条参考文献。所有原图表、9个编号公式、3个定理的主要误差界及3组模拟模型公式以20张清晰原文裁图保留。行内公式与其余未编号表达式重新排版，书目保留英文检索信息并增加中文题名。

**重要范围边界：** 本文反复引用外部《补充材料》S.1至S.11，但该35页 PDF 与官方 arXiv TeX 归档均不包含这些材料。因此，原文中不存在的补充假设、完整证明和补充实验不在本译文中，未声称已译。正文原有的交叉引用全部保留。`coverage.json` 列出原页与译页对应关系。

保留并明确注明了原文可见的不一致：模拟协方差的 `I_p` 维数记号；第4节负迁移阈值中 `n_T` 与第3节 `n_T^{-1}` 的差异；图注“STIFLE”拼写及图内虚线图例“SUPTrans”与图注“SUPSource”的差异。

## 文件与复现

- `translation.zh.md`：完整中文 Markdown。
- `references.zh.md`：43条原文书目信息及中文题名。
- `build.json`：来源哈希与20项原图、表、公式裁剪坐标。
- `assets/`：原文裁图，按3倍比例生成。
- `source/`：精确官方 PDF、版本页 HTML、TeX 包、解包文件和文本提取。
- `build_report.json`、`qa.json`、`coverage.json`：构建与质检记录。
- `qa/pages/`、`qa/pair*.png`：最终23页逐页渲染及双页检查图。
- `SHA256SUMS`：最终产物和辅助文件的完整性校验。

在 dot 云端运行：

```bash
PYTHONPATH=/tmp/cloud-pdf-deps python /workspace/shared/papers-cloud/中文翻译/scripts/build_cloud.py /workspace/shared/papers-cloud/中文翻译/P153_TriplyRobust/build.json
PYTHONPATH=/tmp/cloud-pdf-deps python /workspace/shared/papers-cloud/中文翻译/P153_TriplyRobust/render_qa.py
```

依赖：PyMuPDF、Pillow、pandoc、ziamath、WeasyPrint、Noto CJK 字体。最终23页均已检查，未发现裁切、文字遮挡、缺字、缺图或表格溢出。独立复核已逐页检查35页原文及23页译文，并验证20张裁图与原文新裁图逐像素一致。复核仅补回表3标题末句；重建后第15页另行目检，其余22页与独立目检版本逐像素一致。
