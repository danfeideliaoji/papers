# P122 中文全文译文

主交付物：`P122_CorrelationCalibration_中文全文译文.pdf`（27 页）。可编辑译文：`translation.zh.md`。

原文：Tianshui Chen、Weihang Wang、Tao Pu、Jinghui Qin、Zhijing Yang、Jie Liu、Liang Lin，Dynamic Correlation Learning and Regularization for Multi-Label Confidence Calibration，arXiv:2407.06844v1，2024-07-09。版本链接：https://arxiv.org/abs/2407.06844v1 。许可：https://creativecommons.org/licenses/by/4.0/ 。此为非官方中文翻译及重新排版，原作者未参与或认可本改编；译文文字按 CC BY 4.0 提供。

## 内容与图像范围

- 全文覆盖摘要、关键词、作者单位、资助、I-VI 节、算法 1、式（1）-（17）、表 I-VI、图 1-5 全部图注、55 条参考文献和 7 段作者简介。所提供版本无附录。
- 保留 39 个经检查的原始裁切资产：17 个公式、1 个算法、7 个表格裁切、14 个图像面板裁切。
- 图 1 的 2 张照片、图 2 的输入照片及照片衍生样本热图区域、图 5 的 4 张照片，以及第 13 页的 7 张作者肖像不在交付译文内。各图的预测分数面板/架构示意图等可分离部分已保留；正文和全部图注完整翻译。未核实照片的单独使用权，不宣称保留全部图片。
- 原文不一致（包括 CSFL/CSCL、ILCAR/ILCRA、PLCAR/PLCRA、若干原式记号和一处数值/文字矛盾）保留并明确加注；不将译者修订伪装成原文。
- 编号公式和表格为高分辨率原始裁切，确保数值/排印保真；它们不是重新录入的可搜索公式/表格。正文与参考文献可提取文本，行内数学以矢量图呈现。

## 复现

在本云工作树中运行：

```sh
PYTHONPATH=/tmp/cloud-pdf-deps python build_pdf.py build.json
```

`build_pdf.py` 以共享 `../scripts/build_cloud.py` 为基础，仅将正文拉丁字体优先级设为 Noto Serif，以确保变音字符的正确提取；`prepare_build.py` 记录裁切和构建配置的生成方式。依赖：pandoc、PyMuPDF 1.26.6、Pillow 13.0.0、WeasyPrint 70.0、ziamath 0.13，系统 Noto Serif 与 Noto Serif/Sans CJK SC。构建前会校验原文 SHA-256，绝不覆盖原文。

## 配套验证

- `coverage.json`：正文和图表覆盖记录。
- `rights-and-attribution.json`：许可、改编和图片省略记录。
- `qa/QA.md`：27 页与 39 个裁切的视觉验收记录。
- `qa/verification.json`：结构、字符和哈希检查。
- `SHA256SUMS.txt`：交付文件哈希。
- `delivery-manifest.json`：允许交付的文件列表，**明确排除 `source/` 和任何源文整页渲染**。原文 PDF 仅在既有私有云缓存中保持不变；不得随译文打包，也没有 GitHub 或公开上传。

仅需要阅读时，下载主 PDF 即可；构建配套不是额外的原始全文 PDF 分享。
