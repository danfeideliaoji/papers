# P137 中文全文翻译交付

主文件：`P137_MaskConformal_中文全文译文.pdf`（46页）；权威可编辑文本：`translation.zh.md`。

完整覆盖原作32页：正文1–5节、致谢、56项参考文献（保留英文书目身份并附中文题名）、核查清单、附录A–E全部章节与证明、算法1–5、公式1–18、图1–16、表1–7、全部脚注、100维系数向量。非摘要或缩写证明版本。

## 保真与许可

官方PMLR文件与用户原件SHA完全一致；CC BY4.0证据、署名与图表例外核查见 `LICENSE.md`、`rights_figures.md`、`source/official.html`、`source/pmlr-license-agreement.html`。未发现图表第三方例外。`source/official.pdf`是同哈希官方完整原件。

45个原始裁剪以216dpi生成，数学转写依据源字体区分普通斜体、花体、黑板粗体；未因向量含义擅加粗。复杂推导保留完整原图，中文解释逐段保留。数学中的“such that”表示“满足”，“if”表示“若”；实验图英文轴标/图例在邻近中文说明中解释。原文疑点集中标注，不隐性修正。

## 文件

- `build.json`：来源哈希、全部裁剪页码/坐标与输出设置。
- `assets/`：45个最终公式/图表资产。
- `source/font_evidence.json`：逐跨度字体证据。
- `QA.md`、`qa_report.json`：完整性、渲染、校验记录。
- `qa/`：最终46页渲染和6张联系表。
- `SHA256SUMS`：交付文件校验和。
- `part*.zh.md`、`mid_manifest.json`、`part*_assets.json`：工作分片，非最终权威版本；部分修复仅见合并后的 `translation.zh.md`。

## 云端重建

在 `/workspace/shared/papers-cloud` 根目录执行：

```sh
PYTHONPATH=/tmp/cloud-pdf-deps python 中文翻译/scripts/build_cloud.py 中文翻译/P137_MaskConformal/build.json
python 中文翻译/P137_MaskConformal/qa_render.py
```

使用共享既有构建器（未修改）、pandoc、ziamath、WeasyPrint、Noto CJK。源文件校验失败时构建会拒绝继续。全程仅云端工具；无桌面启动、外部翻译器、API密钥或GitHub写入。
