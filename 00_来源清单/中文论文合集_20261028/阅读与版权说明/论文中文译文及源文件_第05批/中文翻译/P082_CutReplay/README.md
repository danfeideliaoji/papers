# P082：裁出并回放（CUTER）

- `P082_CutReplay_中文全文译文.pdf`：27页非官方中文全文译文。
- `translation.zh.md`：完整可编辑译文，涵盖摘要、正文1–4、算法1、影响声明、致谢、72条参考文献、附录A–F（含全部证明、脚注、图表说明）。
- `main.zh.md`、`references.zh.md`、`appendix.zh.md`：分工工作稿；最终排版以 `translation.zh.md` 为准。
- `build.json`、`assets/`：42个公式/图表精确裁图与可重现构建参数。
- `source/wang25bg.pdf`：从官方 PMLR 下载的19页源文件；原仓库 PDF 未被覆盖。
- `source/publication.html`、`source/license.html`、`LICENSE.md`：来源与 CC BY 4.0 证据及署名。
- `source/version_comparison.json`、`source/version.diff`：版本核对。官方版本仅在提取文本中移除了 arXiv 标记并将作者 Shao-yuan 改为 Shao-Yuan；其余提取文本相同。
- `qa_report.json`、`review/`、`hashes.json`：质量检查、逐页渲染及 SHA-256。

构建（仓库根目录）：

```sh
PYTHONPATH=/tmp/cloud-pdf-deps python 中文翻译/scripts/build_cloud.py 中文翻译/P082_CutReplay/build.json
```

原文 https://proceedings.mlr.press/v267/wang25bg.html 。本译文不是作者审定版本。保留原文符号字体而非依语义改成粗体向量：x/y/p/D/Y/L/R 等为斜体，C/M 的具体花体依原文；全部展示公式使用官方原版裁图，含原有编号，不使用 `\tag`。表格采用全数值原图并提供中文表头及题注对应，图内英文采用中文逐项释义，避免重绘损失数据。
