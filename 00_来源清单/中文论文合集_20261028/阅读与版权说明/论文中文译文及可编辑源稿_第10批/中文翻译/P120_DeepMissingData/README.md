# 《含缺失数据的深度学习》中文全文译文

- 阅读文件：`P120_DeepMissingData_中文全文译文.pdf`（60 页）。
- 原作：Tianyi Ma、Tengyao Wang、Richard J. Samworth，*Deep learning with missing data*，arXiv:2504.15388v3，2026-07-01，57 页。
- [版本页面](https://arxiv.org/abs/2504.15388v3) · [官方 PDF](https://arxiv.org/pdf/2504.15388v3) · [CC BY 4.0 许可](https://creativecommons.org/licenses/by/4.0/)。
- 非官方译文；OpenAI 助手译制，2026-10-26。未经作者审定，不代表作者背书。中文翻译及新增说明也按 CC BY 4.0 提供。

## 范围与图像例外

完整翻译正文、附录 A–I、全部证明和 8 条脚注；75 条文献保留完整原书目信息，逐条附题名中文译注。保留编号公式 (1)–(40) 及其他公式。复杂陈列公式采用原文高清裁切，行内公式以矢量数学排版；部分数学因此不能作为普通文本搜索或复制。

**图 1–5、7–13 共 12 幅保留。图 6 的 MNIST 样例位图因未独立核实数据集图像许可而未收入；完整图注、正文讨论及[原图定位链接](https://arxiv.org/pdf/2504.15388v3#page=21)均保留。没有替换图像。** 原图英文标签保留，中文图注解释相应内容。原文无表格。

## 文件

- `translation.zh.md`：统一中文源稿；其余 `*.zh.md` 为分章节源稿。
- `build.json`、`*.assets.json`：重建配置与裁切位置；`assets/` 包含 135 个资产。
- `assets-manifest.json`：每项资产的来源页、裁切框、输出页和哈希。
- `coverage.md` / `coverage.json` / `coverage-pages.csv`：章节、57 个原文页及 40 个编号公式的对应清单。
- `QA.md`、`qa/automated-checks.json`、`qa/E_QA.md`、`qa/independent-review.json`：作者初检与已完成的独立全文质量核验记录。
- `source-verification.json`、`license-evidence.json`、`NOTICE.md`：来源、许可和变更记录。
- `SHA256SUMS`：交付文件校验值。

原始 PDF 只保留在内部工作缓存，未修改，**不收入本交付压缩包**；原文整页预览亦不打包。需要原作时请使用上述官方链接。

## 云端重建

在本次云端工作区、已备齐依赖和官方原文缓存的前提下：

```bash
cd /workspace/shared/papers-cloud/中文翻译/P120_DeepMissingData
python assemble.py
PYTHONPATH=/tmp/cloud-pdf-deps python /workspace/shared/papers-cloud/中文翻译/scripts/build_cloud.py build.json
```

构建器会先校验官方源 PDF 的 SHA-256，再生成裁切、MathML/SVG 和 PDF。原文预期 SHA-256：

`35899578665678567d0a4d17c57aa32f673023af1b7feb81b2ac2ed522196fcb`

本目录与压缩包均未上传 GitHub 或其他公开站点。
