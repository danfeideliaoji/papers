# P131 中文全文译文

- 原论文：Riccardo D’Alberto、Meri Raggi，Integrating rather than collecting: statistical matching in the data flood era，Statistical Papers 65 (2024):2135–2163。
- 原文 DOI：https://doi.org/10.1007/s00362-023-01468-3
- 最终交付：`P131_StatisticalMatching_中文全文译文.pdf`（26页）。
- 可编辑译稿：`translation.zh.md`；裁图与构建清单：`build.json`、`assets/`。
- 完整范围：摘要，正文第1–6节，表1（中文重排），图1及中文图注，所有数学推导和编号公式(1)–(7)，脚注1–8，作者贡献、经费及开放获取说明，附录A.1–A.3全部应用和软件条目，95条参考文献、出版者声明。本文为综述，无单列证明或实验补充材料。
- 书目保留原文作者、题名、出版信息及链接，便于准确检索；不是参考文献摘要。
- 权利：CC BY 4.0。来源PDF第23页明确允许改编；图1无第三方除外声明。详见`license.txt`。
- 源文件 SHA-256：`86de35310b65a73fc4b45feee52ae2f9c77adf4efd9878ec5db88067113763d4`，原文未修改。
- 非官方译文；中文翻译与重新排版属于改编，不代表原作者或出版者审定、认可。

## 复现

在论文库根目录执行：

```bash
PYTHONPATH=/tmp/cloud-pdf-deps python 中文翻译/scripts/build_cloud.py 中文翻译/P131_StatisticalMatching/build.json
```

最终稿与清单即构建源；无需运行一次性预处理脚本。

## QA

全26页渲染在`qa/page-*.png`，5张逐页联系表在`qa/sheet-*.png`。已检查全页布局，并对公式密集页4、7、12、13、14及全部9个裁图作大尺寸复核。式(4)第一轮下括号边缘不足，已扩大底部范围重建；最终全部裁图墨迹边缘至少留2pt。字体按原文核对：变量x/y/z和参数向量粗体，矩阵B粗斜体，A/B/C数据集标签正体；未按语义全局推定字体。公式(1)–(7)直接保留原始排版。详情见`qa_report.json`与`review.md`。
