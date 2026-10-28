# P105 LIMO 中文全文译文

原文：Xi-Zhu Wu and Zhi-Hua Zhou. *A Unified View of Multi-Label Performance Measures*. ICML 2017, PMLR 70:3780–3788.

- 官方论文页：https://proceedings.mlr.press/v70/wu17a.html
- 原文正文：仓库 `02_Multi-Label/A Unified View of Multi-Label Performance Measures/paper.pdf`，9 页。
- 官方补充材料：https://proceedings.mlr.press/v70/wu17a/wu17a-supp.pdf ，5 页，保存在 `source/p105-supp.pdf`。
- 出版许可：CC BY 4.0，依据 https://proceedings.mlr.press/pmlr-license-agreement.html 。原图均为本文数学示意、实验结果或公式表格；逐图检查未见单独的第三方权利声明。

本文件夹的 PDF 为非官方中文全文译文，包含正文六节、所有定理及证明、算法、四幅原图、两张正文表、致谢、22 条完整英文参考文献，以及补充材料 A/B 全文和完整实验表。表 B.3 按数据集和指标分块，重复必要表头，保留全部数据。

正文与补充材料中的公式、概率推导和聚合定义若存在不一致，均用“译注”标明，不冒充作者勘误。图表内英文标识和参考文献保留原文，中文图注及解释在正文中提供。原文所有文件均保持不变。

## 文件

- `translation.zh.md`：可编辑的完整中文稿。
- `P105_LIMO_中文全文译文.pdf`：供阅读的中文排版版。
- `build.json`：原文哈希及各图表裁剪位置。
- `assets/`：根据原文生成的图表和复杂公式。
- `build_report.json`、`qa_report.json`、`review.md`：构建与质量检查记录。

## 重建

在仓库根目录，根据 `中文翻译/scripts/cloud-pdf-requirements.txt` 安装依赖，保证 Pandoc 和 Noto CJK 字体可用，然后运行：

```sh
python 中文翻译/scripts/build_cloud.py 中文翻译/P105_LIMO/build.json
```

本次翻译与构建在 dot 云端工作区完成，不需要用户电脑运行。数学公式由 MathML 转成矢量 SVG 后嵌入；公式可清晰缩放，但不保证作为普通文本复制。中文正文可选择和搜索。
