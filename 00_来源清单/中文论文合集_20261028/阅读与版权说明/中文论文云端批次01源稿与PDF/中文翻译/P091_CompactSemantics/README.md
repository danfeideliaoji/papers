# P091 · COME 中文全文译文

- 成品：`P091_COME_中文全文译文.pdf`（22 页）
- 可编辑译文：`translation.zh.md`
- 构建清单：`build.json`；31 个原始裁切资产（9 图、4 表、18 公式）位于 `assets/`
- 原文：`../../03_Multi-View_Multi-Label/2025-ICML-Learning Compact Semantic Information for Incomplete Multi-View Missing Multi-Label/paper.pdf`（14 页）
- 运行环境：仅 dot cloud；未使用用户桌面、local Codex、cloud_threads、外部翻译服务或 API 密钥。未进行 GitHub 写入。

## 来源与许可

Wen, Jie; Liu, Yadong; Tang, Zhanyan; He, Yuting; Chen, Yulong; Li, Mu; Liu, Chengliang. *Learning Compact Semantic Information for Incomplete Multi-View Missing Multi-Label Classification*. ICML 2025, PMLR 267:66467–66480.

- 官方文章：https://proceedings.mlr.press/v267/wen25c.html
- 官方 PDF：https://raw.githubusercontent.com/mlresearch/v267/main/assets/wen25c/wen25c.pdf
- PMLR 出版协议：https://proceedings.mlr.press/pmlr-license-agreement.html
- CC BY 4.0：https://creativecommons.org/licenses/by/4.0/
- 原文 SHA-256：`d7b68fadc1a73fbfe679a3800285eafd7b617d409d359a38fe6e55b2b1293f6c`
- `rights_review.csv` 的 P091 行已记录官方版本标题/SHA 匹配；本次重新计算源文件 SHA 一致，且访问官方文章页与许可协议核验。
- 图 1–9 为模型实验曲线/柱状图/雷达图、方法框架图和参数分析；逐图及图注检查未发现第三方署名、排除 CC BY 的信用行或许可例外。表 1–4 与算法 1 亦无独立权利声明。

本件为非官方 AI 辅助中文译文，改动为翻译及版式调整；保留作者、出版信息、原文和许可链接，不暗示作者认可。所有编号公式原图保留，参考文献保留英文，图中英文标签配中文解释。未把摘要替代原文段落。

## 覆盖

摘要、正文第 1–5 节及全部子节、假设 2.1、命题 2.2、定义 2.3、算法 1（12 步）、致谢、影响声明、参考文献 55 条、附录 A–F；编号公式 1–18、图 1–9、表 1–4。附录 A 的所有等式推导完整保留，原文无其他独立证明章节。

原文中实验“所有指标最优”与 ESPGame 1-OE 数值差异、雷达图“1-AUC”标注、图 7 图注遗漏第三数据集、附录 F 的五/四/两个数据集表述，均保留并明确加译注；未擅自校正公式（12）的符号。

## 重建与 QA

在仓库根目录运行：

```bash
PYTHONPATH=/tmp/cloud-pdf-deps python 中文翻译/scripts/build_cloud.py 中文翻译/P091_CompactSemantics/build.json
PYTHONPATH=/tmp/cloud-pdf-deps python 中文翻译/P091_CompactSemantics/render_qa.py
```

所有 22 页均已渲染至 `qa/page-*.png` 并通过全页联系表检查；公式资产额外逐式检查。修正了式（12）裁切中的邻行残片与末页孤立图注。最终页面无越界块、无重叠/缺字/截断，图表与公式可读。详见 `qa_report.json`、`review.md` 和 `build_report.json`。
