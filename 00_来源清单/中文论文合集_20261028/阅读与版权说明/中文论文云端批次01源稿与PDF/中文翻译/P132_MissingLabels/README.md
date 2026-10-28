# P132 中文全文译文

交付：`P132_MissingLabels_中文全文译文.pdf`（40页）。含原论文9页正文全部内容及官方19页补充材料 A–E；不是摘要。长公式、图表及原文参考文献以精确裁切保留，配中文翻译/释义；全部15表、3图、3算法均覆盖。原文疑似错误均以译注明示。

## 文件

- `translation.zh.md`：整合全文，可编辑。
- `main.zh.md`、`supplement.zh.md`：分段源稿；若改动需重新整合。
- `build.json`：可复现裁切/构建清单，45个精确裁切。
- `source/p132-supp.pdf`：从官方页面下载的补充原文。
- `rights_and_source_review.md`：许可、来源与原文问题审核。
- `qa_report.json`、`supplement-qa.json`、`build_report.json`：完整性及渲染核验。

## 重建

在仓库根目录运行：

```bash
PYTHONPATH=/tmp/cloud-pdf-deps python 中文翻译/scripts/build_cloud.py 中文翻译/P132_MissingLabels/build.json
```

依赖为 pandoc、PyMuPDF、Pillow、Noto CJK、ziamath 与 WeasyPrint，详见通用构建脚本。`/tmp/cloud-pdf-deps` 为本次云端安装位置，重建环境需自行提供同等依赖。不要依赖 LaTeX `\tag` 自动显示编号；本稿以显式数学括号保留编号。

最终 PDF SHA-256：`337220952c36a79d55e7c4d92a93611ab6a861e4bde3e9f3bfb1e471553d787e`。

本工作完全在 dot 云端完成；原论文、其他目录与根状态文件未修改；未推送或创建 PR。非官方 AI 译文，作者/出版方未审定。
