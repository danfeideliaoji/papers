# RCSC 中文全文样例

- 原作：Yalan Qin, Nan Pu, Guorui Feng, Nicu Sebe. *Robust Consensus Anchor Learning for Efficient Multi-view Subspace Clustering*. ICML 2025, PMLR 267:50327-50337.
- 官方页面：https://proceedings.mlr.press/v267/qin25e.html
- 官方 PDF：https://raw.githubusercontent.com/mlresearch/v267/main/assets/qin25e/qin25e.pdf
- 原文与本译文许可：[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)。[PMLR 出版协议](https://proceedings.mlr.press/pmlr-license-agreement.html) 第 2、3 条说明许可与署名要求。
- 本译文非官方，未获原作者审阅或背书；由 Codex 辅助翻译、校勘及重新排版。
- PDF：[RCSC_中文全文译文.pdf](RCSC_中文全文译文.pdf)。可编辑中文源稿：[translation.zh.md](translation.zh.md)。

## 范围与保真

覆盖原文全部正文、摘要、定理 1-3 与证明、引理 1、算法 1、全部 34 个编号公式、图 1-3、表 1-3、致谢、影响声明和全部参考文献。原文无另附的附录或补充材料。本次完整范围指仓库中已核实的 11 页正式出版 PDF，不扩展到未提供的其他版本。

公式、图表原样保留，正文、图注、算法及说明译为中文；图内英文标签另附中文对照，书目按英文原样保留，便于核查。原文已有的记号或图轴疑点在“译者校勘记录”单列，不擅自修正文献。无正文删节，无不可读区域。

官方下载版与仓库 `paper.pdf` 逐字节一致。SHA-256：`923aabbbcdd7605a235c49cc0bd607731cc5c9ccd5e581d99478bed6369b4d9d`。所有新文件位于 `中文翻译/`，原文未修改。

## 图片审核

| 项目 | 原文页 | 内容与处理 | 权利检查 |
|---|---:|---|---|
| 图 1 | 7 | 参数与锚点数敏感性曲线；保留原图，加中文图注 | 未见单列的第三方版权或许可例外，随文章 CC BY 4.0 |
| 图 2 | 7 | 噪声鲁棒性曲线；保留原图，加中文图注 | 同上 |
| 图 3 | 7 | 运行时间柱状图；保留原图，加中文图注 | 同上 |
| 表 1-3 | 7-8 | 实验数值、误差项与强调原样保留 | 文章 CC BY 4.0；不复制数据集本身 |

数据集名称和相关文献是引用，不表示拥有其另行再分发权。原文正文、图注、页脚与文末已逐项检查，未见所用图表的例外声明。

## 本地复现

需要 Python 3.12+、`reportlab`、`pypdf`、`Pillow`、`pdfplumber` 和 Poppler `pdftoppm`。字体默认使用 Windows 微软雅黑与 Cambria（嵌入 PDF 子集；仓库不分发字体文件）。可通过参数使用本机已获许可、具有相应字符覆盖的 TrueType 字体。

```powershell
python ../scripts/build_rcsc.py --poppler pdftoppm --font C:/Windows/Fonts/msyh.ttc --math-font C:/Windows/Fonts/cambria.ttc
python ../scripts/verify_rcsc.py
```

从本目录执行；脚本读取相对仓库路径，不依赖外部翻译服务，也不访问网络。中间渲染文件在 `中文翻译/tmp/`，已由 `.gitignore` 排除。`assets.json` 记录每个裁片的原页与坐标，公式和实验图按 240 DPI 截取，最终 PDF 的中文文本可选择和检索，参考文献保持矢量文本。

`build_report.json` 记录版本、页数与资源计数；`qa_report.json` 记录自动完整性检查；`review.md` 记录人工逐页渲染检查。重跑构建后必须重新进行视觉检查，不能沿用旧的通过标记。
