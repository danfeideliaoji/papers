# P139 中文全文译文

**已完成：40 页中文 PDF，全部正文、附录、32 条双语参考文献、11 幅图、16 张表和原公式。**

- 最终 PDF：`P139_ActiveLearningPitfalls_中文全文译文.pdf`
- 完整可编辑译文：`translation.zh.md`
- 原版裁切图像：`assets/`（29 个，表 2 分为两部分并重复原表头）
- 构建清单：`build.json`
- 许可核验：`rights_review.local.json`、`LICENSE.md`
- 逐页／逐裁切 QA：`qa.json`、`QA.md`、`asset_inventory.json`；页面图像见 `qa/`
- 校验和：`SHA256SUMS`

原文为 Dustin Eisenhardt、Yunhee Jeong、Florian Buettner 的 *Mind the Gap: A Framework for Assessing Pitfalls in Multimodal Active Learning*，arXiv:2603.29677v1（41 页）。官方 PDF 与仓库原件逐字节一致，许可为 CC BY-SA 4.0。本中文改编依相同许可提供。原件未改动；无桌面操作，无 GitHub 或公开上传。

本译文不是摘要或节译。图表及公式保留原版字体、数据和符号，附完整中文说明；参考文献保留全部原始书目信息并加中文题名。原文的 AULC 公式、幂集术语、训练网格、预算规则及表间差异都保留并明确注记，不静默修正。

## 重新构建

在仓库根目录执行：

```bash
PYTHONPATH=/tmp/cloud-pdf-deps python3 中文翻译/scripts/build_cloud.py 中文翻译/P139_ActiveLearningPitfalls/build.json
python3 中文翻译/P139_ActiveLearningPitfalls/render_qa.py
```

需 Pandoc、PyMuPDF、Pillow、Noto CJK、Noto Serif、WeasyPrint 和 ziamath。每次内容／排版更改后必须重新检查所有受影响页面及裁切，不能仅以脚本通过替代人工视觉检查。`finalize.py` 中的视觉结论只适用于本次已逐页审阅的最终版本，重建后不可未经检查直接复用。
