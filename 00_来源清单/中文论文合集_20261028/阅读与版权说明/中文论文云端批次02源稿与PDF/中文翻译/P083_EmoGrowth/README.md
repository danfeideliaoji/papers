# P083 EmoGrowth 中文全文译本

主交付：`P083_EmoGrowth_中文全文译文.pdf`。
可编辑全文：`translation.zh.md`。
构建清单：`build.json`；高分辨率原图/表裁片：`assets/`。
来源、许可及第三方图像限制：`source-license.md`；正式 PDF 与官网快照：`source/`。
质量记录：`qa_report.json`、`review.md`、`qa/`；构建及输出哈希：`build_report.json`。

## 范围

完整覆盖原 18 页：摘要、正文第 1-5 节、致谢、影响声明、42 条参考文献（保留英文书目并译题名）、附录 A-D、算法 1、全部 18 个编号公式、图 1-9、表 1-11。表 6 的 27+28 项定义全部译为中文；数值结果表保留源图并提供中文表头与说明。原文没有独立定理/证明。

为避免图内标签失真，全部图及数值表保留原始内容，图注及名称释义译为中文。17 个 3 倍栅格裁片覆盖 9 图和 9 个图像表（表 4/6 为中文重排表，图 8 与表 7 合并一个裁片）。数学公式经 MathML 渲染为矢量 SVG，保留矩阵/向量粗体与标量差异。译注标出原文方法数量、任务索引、附录引用、Video28 标签、情感维度统计、ROI 计数及原图 ALSL 等不一致之处，未改实验数值。

## 云端重建

在 `/workspace/shared/papers-cloud` 中运行：

```bash
PYTHONPATH=/tmp/cloud-pdf-deps python 中文翻译/scripts/build_cloud.py 中文翻译/P083_EmoGrowth/build.json
```

清单 source 相对于仓库根目录，输出相对于本目录；脚本先检查源 PDF SHA-256。无需用户电脑或外部翻译接口。共享构建脚本未被修改。
