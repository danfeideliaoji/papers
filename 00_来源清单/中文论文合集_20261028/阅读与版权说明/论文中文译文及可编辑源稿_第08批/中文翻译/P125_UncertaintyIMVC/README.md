# P125 中文全文译文

已完成《Exploring and Exploiting Uncertainty for Incomplete Multi-View Classification》的中文全文 PDF，共 16 页，依据 arXiv:2304.05165v1 的 10 页原文。

## 文件

- `P125_UncertaintyIMVC_中文全文译文.pdf`：正式中文全文 PDF。
- `translation.zh.md`：可编辑译文；保留引用编号，附参考文献英文原题与中文译题。
- `build.json`：原始 PDF 哈希与所有裁图的源页、坐标和排版宽度。
- `assets/`：4 幅原图、表 1 的 5 个数据集分组、9 个编号公式，共 18 个无重绘裁图。
- `coverage.json`：原文页与译文页的内容对应清单。
- `qa_report.json`、`review.md`、`qa/`：逐页、逐裁图审阅和文本/字形检查记录。
- `SHA256SUMS`：正式文件、来源、构建配置、资产及检查记录的 SHA-256。

## 覆盖范围

作者、单位、同等贡献与通讯作者脚注；摘要；正文第 1–5 节及全部小节；定义 3.1、3.2；公式 1–9；算法 1；图 1–4；表 1 全部 5 个数据集、8 个方法和 6 个缺失率列；致谢；参考文献 [1]–[55]。此版本无附录。

## 来源和授权

- 版本页：https://arxiv.org/abs/2304.05165v1
- PDF：https://arxiv.org/pdf/2304.05165v1
- 原作许可：https://creativecommons.org/licenses/by/4.0/
- 原作者：Mengyao Xie、Zongbo Han、Changqing Zhang、Yichen Bai、Qinghua Hu。
- 已依据父任务核验的官方 arXiv CC BY 4.0 信息、精确源文件哈希和全部 4 图目视授权检查制作。`source-verification.json` 的 `decision` 为 `licensed_exact_source_figures_visually_cleared`；其早期 `status: rights_pending` 是遗留字段，以上述明确审查结论为准。
- 原始 PDF 和仓库原件均保持不变。未进行 GitHub 提交、公开发布或其他外部上传。
- 本文件是 ChatGPT 制作的非官方中文翻译，改变内容为语言转换、图表注释中文化和单栏重排；不暗示作者认可。许可与改动说明已写入 PDF 首屏。

## 可复现构建

在云端工作目录运行：

```bash
PYTHONPATH=/tmp/cloud-pdf-deps python /workspace/shared/papers-cloud/中文翻译/scripts/build_cloud.py /workspace/shared/papers-cloud/中文翻译/P125_UncertaintyIMVC/build.json
```

需 PyMuPDF、Pandoc、WeasyPrint、ziamath、Noto CJK 和 Noto Serif。公式 1–9 与原图原表使用原文裁图，以避免源 PDF 隐藏 LaTeX 文本层干扰；这些图片中的文字不可直接按正文方式搜索，完整内容仍可查看，其他正文与参考文献可抽取。

## 保留的原文异常

表 1 的 YaleB / TCCA / η=0.3 单元格印为 `0.90.77 ± 0.01`，保留原图并加译注，不推测改数。图 3、4 图注列出 η=0，但图中横轴从 0.1 开始；仍保留原注与原图。书目中原有的重复项 [41]/[47]、拼写和部分条目未给出的会议信息均保留，不以外部推测替换。
