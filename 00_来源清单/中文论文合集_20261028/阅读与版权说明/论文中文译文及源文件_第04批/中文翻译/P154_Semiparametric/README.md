# P154 半参数共形预测：中文全文译本

## 来源与许可

- 原文：Ji Won Park, Kyunghyun Cho. **Semiparametric conformal prediction**. *Proceedings of the 28th International Conference on Artificial Intelligence and Statistics*, PMLR 258:3880–3888, 2025.
- 官方页面：https://proceedings.mlr.press/v258/park25c.html
- 官方 PDF：https://raw.githubusercontent.com/mlresearch/v258/main/assets/park25c/park25c.pdf
- PMLR 出版协议：https://proceedings.mlr.press/pmlr-license-agreement.html
- 许可：Creative Commons Attribution 4.0 International，https://creativecommons.org/licenses/by/4.0/ 。协议第2条明确向公众授予 CC BY 4.0，第3条要求引用原始出版物并链接 PMLR 原文页面。
- 本译本为中文翻译及重新排版；原文图形、部分数学公式以高分辨率裁图保留，文字、全部图注/表注、参考文献题名和检查清单均译为中文。作者和出版方未对本译本表示认可。译文沿用 CC BY 4.0，并保留原作者权利。
- 原文与重新下载的官方 PDF 均为26页，SHA-256 完全相同：`cc09249634c8b5da31d2283acd5e687cfd3783851c44c222738d2eb94391f3c9`。
- `rights_review.csv` 中 P154 记录已核对；该共享文件未改动。其未完成的逐图例外审查已在本目录完成：图1–12及图注未发现第三方版权/转载/单独许可标注。文中引用其他论文、软件、数据的文字不等同于图形版权例外。本结论仅针对所检查的官方 PDF 可见内容，见 `rights_review.local.json`。
- 原 PDF 首页仍含 AISTATS 2024、卷号待定的模板脚注。译本完整翻译此脚注，并明确区分正式的 PMLR 2025 出版信息。

## 内容与文件

- `translation.zh.md`：整合全文，含正文1–8节、致谢、73条参考文献、检查清单、补充材料9–17节。
- `main.zh.md`、`references.zh.md`、`appendix.zh.md`：可编辑分段来源。
- `P154_Semiparametric_中文全文译文.pdf`：中文全文 PDF。
- `build.json`：源文件哈希、输出名和全部原始裁图的页码/坐标/展示尺寸。
- `assets/`：源 PDF 高分辨率裁图；图形内英文标签保留，在对应中文图注中给出释义。
- `source/official.pdf`：独立下载的官方同版本文件；`layout.txt`、`raw.txt` 为原文提取文本。
- `qa/`、`qa.json`：最终页面渲染、联系表、构建与覆盖检查。
- `appendix-assets.json` / `appendix_assets.json`：附录裁图审计记录和供构建器合并的配置。
- `SHA256SUMS`：最终交付文件哈希。

## 符号与原文差异

保留原文普通斜体向量字母（不擅自改成粗体），并区分花体集合符号、黑板粗体概率/经验测度与普通斜体分布符号。编号公式显式编号，不使用 `\tag`。附录公式裁图直接保留原始字形。原文内部的若干疑似排印或论证差异均不作未声明的数学纠正；有关取整位置、集合/水平曲线、算法子集用途、证明、数据维度和缺失性方向的差异在译者说明中标出。

## 云端重建

在仓库根目录运行：

```bash
PYTHONPATH=/tmp/cloud-pdf-deps python 中文翻译/scripts/build_cloud.py 中文翻译/P154_Semiparametric/build.json
```

依赖：PyMuPDF、Pandoc、WeasyPrint、ziamath、Noto CJK。本译本仅在 dot 云端工作区制作；未使用用户电脑、外部翻译服务或 API 密钥。共享构建脚本、其他论文目录及根 CSV 未被本任务改动。
