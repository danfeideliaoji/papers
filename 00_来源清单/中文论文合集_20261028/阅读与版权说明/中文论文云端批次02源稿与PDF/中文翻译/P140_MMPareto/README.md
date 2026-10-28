# P140 · MMPareto 中文全文译文

成品：**P140_MMPareto_中文全文译文.pdf**（23 页）。

完整翻译原始 14 页论文：摘要、正文 1–5、致谢、影响声明、47 条参考文献（题名中译并保留原名及完整著录）、附录 A–F、全部证明、式 (1)–(17)、算法 1、6 幅图及全部子图、9 张表。不是概要或节选。

- `translation.zh.md`：可编辑中文全文。
- `build.json`：源文件哈希及 12 个图表裁切清单。
- `assets/`：原图 1–6、原表 1–5/7 的高分辨率裁切；数据不重绘。表 6/8/9 全部中文重排。所有图表均有中文图注/表头/标签释义。
- `LICENSE.md`：官方许可、完整署名及逐图信用行审查。
- `official_source_verification.txt`：官方 PDF 下载后 SHA-256 核对结果。
- `QA.md`、`build_report.json`、`qa/`：覆盖、校对、逐页视觉证据与构建报告。
- `SHA256SUMS.txt`：交付文件校验和。

## 在 dot 云端重建

从 `/workspace/shared/papers-cloud` 运行：

```bash
PYTHONPATH=/tmp/cloud-pdf-deps python 中文翻译/scripts/build_cloud.py 中文翻译/P140_MMPareto/build.json
```

依赖：pandoc、PyMuPDF、Noto CJK、WeasyPrint、ziamath。构建器会重新生成图像与 PDF 并将报告的人工 QA 状态重置为 pending；如重建，需重新查看所有渲染页后更新 QA。未改动共享构建器。

## 阅读说明

本文是非官方中译改编，原作者 Yake Wei、Di Hu 保留其权利。按 PMLR 官方出版协议 CC BY 4.0 署名提供。图内原始英文字和数值保留，中文解释放在相邻图注；参考文献出版物名、作者名及算法专名保留以便检索。原文中的情绪枚举错误、参数符号差异和理论论证中的假设缺口均有明确译注，未静默修正。公式为矢量图像，部分 PDF 文本抽取工具不会读出公式；完整 LaTeX 公式在 Markdown 中。

全程在 dot 云端完成，无用户桌面调用、无外部翻译服务、无 API 密钥、无 GitHub 写入；未修改其他论文目录或根 CSV。
