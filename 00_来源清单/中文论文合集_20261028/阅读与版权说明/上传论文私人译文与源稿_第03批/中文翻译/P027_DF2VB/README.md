# P027 DF2-VB 私人中文全文译本

已完成：10 页用户上传原文对应 20 页中文 PDF。保留 14 个编号公式、4 图、3 表、23 步算法、两条定理陈述、致谢和 39 条参考文献。图表及编号公式以 3 倍分辨率原页裁切保存，表 1 全部 210 个数值条目均经边界覆盖检查。参考文献维持原始英文书目信息；图表英文标签有中文释义。

## 文件

- `translation.zh.pdf`：最终中文 PDF。
- `translation.zh.md`：可编辑的完整中文译稿。
- `assets/`：28 项原始裁切资产，包含原件 CVF 页首声明。
- `build.json`：源文件哈希、全部裁切坐标及排版尺寸。
- `build_cloud.py`、`cloud-pdf-requirements.txt`：本目录可复现构建器及额外依赖清单；构建器为已核读的云端排版工具副本，不是用户上传档案内的代码。
- `coverage.json`、`coverage.tsv`：逐节与源页对应的覆盖清单。
- `qa/`：20 页最终渲染 PNG、拼图、裁切检查拼图、边界与字体检查、表 1 数值覆盖记录。
- `QA.md`：人工视觉核对和保留原文疑点的说明。
- `build_report.json`：最终构建与复核状态。
- `SHA256SUMS`：本目录交付文件的 SHA-256 清单（不包含自身）。
- `source/user-upload.pdf`、`source/user-upload-provenance.json`：用户提供的原始 PDF 与来源证据。

## 范围与权利

本成果为用户提供材料的私人翻译，不是公开发布。保留原作者及出版方的声明和标识，不新增或推定公开再分发许可，不声明 CC 许可。仅依据提供的 10 页 PDF；未包含附录或两条定理的证明，原件亦未见明确的补充材料文件链接。不补写未提供内容。

原文 Friedman 检验原假设的措辞、消融叙述的推论方向、表 3 的 FDW/FDF 字样及式（11）均保持原样，并在译者校核说明中区分于作者正文。实验数字及源件粗体标记按原样保留。独立复核修正了 57 个行内/中文算法表达式的源字体忠实性问题，详见 `qa/inline_fidelity_fixes.json`，未改动原始公式、数据或研究命题。

## 云端重建

在当前目录结构下执行：

```bash
PYTHONPATH=/tmp/cloud-pdf-deps python /workspace/shared/private-papers/中文翻译/P027_DF2VB/build_cloud.py /workspace/shared/private-papers/中文翻译/P027_DF2VB/build.json
```

依赖：Python、PyMuPDF、Pillow、Pandoc、Noto Serif / Sans（含 CJK SC）、WeasyPrint、ziamath。额外 Python 依赖见清单。译稿中的局部 CSS 优先采用 Noto Serif，确保 Demšar 的 š 等拉丁变音字符正确显示。无需桌面应用、网络下载或执行论文/档案代码。

重建会更新 PDF 与构建报告。之后应重新渲染和复核所有页面，并更新 QA 状态及哈希清单；旧有人工检查不能自动适用于后续编辑版本。
