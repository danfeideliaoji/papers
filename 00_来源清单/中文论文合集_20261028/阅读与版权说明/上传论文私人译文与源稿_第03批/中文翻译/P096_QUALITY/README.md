# QUALITY 中文全文译文

本目录为用户提供论文的私人中文翻译交付包，不是开放许可的再发行版本。

- `P096_QUALITY_中文全文译文.pdf`：17 页中文译本，覆盖原件 9 页全部正文、致谢、参考文献和算法。
- `translation.zh.md`：完整可编辑译稿；原图、表、公式以 `{{asset_id}}` 引用。
- `assets/`：4 图、3 表、15 个编号公式，共 22 张原件裁图，216 dpi；不改绘数据与数学记号。
- `build.json`、`build.sh`：可复现构建入口；使用同级 `../scripts/build_cloud.py` 和其依赖。
- `source/user-upload.pdf`、`source/user-upload-provenance.json`：保持不变的用户上传底本及其来源链。
- `coverage.json`：原文页码与译文页码覆盖对照。
- `qa/QA.md`、`qa/checks.json`、`qa/page-*.png`：逐页视觉复核和结构检查记录。
- `SHA256SUMS`：本目录文件的 SHA-256 清单（清单自身除外）。

原件：Quality-aware and Soft Consistency Driven Representation Fusion for Incomplete Multi-view Multi-label Classification，AAAI-26，第 23882–23890 页。源 PDF SHA-256：`6a0ce7eb13f535b79d3ab6d92fd8c2d3eac895d005594edc75c995857763a119`。

## 内容与范围

正文为完整逐段翻译，不以摘要代替正文。37 条参考文献保留原作者、年份、英文题名、出版信息、页码，并附中文题名。全部 15 个编号公式使用原件裁图以保留符号和编号；行内数学通过 MathML 转 SVG 排版；已按底本逐项核对粗体直立、斜体和花体，40 处修复明细见 `qa/notation-source-audit.json`。算法 1 的输入、初始化、输出及 13 行步骤均译出。

原文引用附录 A/B，但所提供的 9 页底本中没有附录；译本首部与相关位置明确披露，不补写缺失内容。正文“六个数据集”与其列举的五个数据集不一致，图 3 图注“九种方法”与图例十一种方法不一致；译本保留原文并作独立译注，不改动实验数字或图例。参考文献中作者排列及不完整出版信息按原文保留，例如 `W. Tsang, I.` 和 2024 年 Fast Unpaired Multi-view Clustering 未附出版信息的条目。

## 权利与隐私

Copyright © 2026, Association for the Advancement of Artificial Intelligence (www.aaai.org). All rights reserved.

本包仅用于用户授权的私人阅读翻译。上传来源不构成公开分发许可，未添加开放许可证，未更改既有公开权利状态。构建、提取与核验均在云工作区内完成；未使用桌面，未执行论文源码或用户上传归档中的代码，未将交付物发布到公共位置。

## 构建

从云工作区执行 `bash build.sh`。依赖清单见 `../scripts/cloud-pdf-requirements.txt`。字体使用已安装的 Noto Serif CJK SC / Noto Sans CJK SC；参考文献 `Müller` 整词指定 Noto Serif，保留 ü。

重建会重置 `build_report.json` 中的审核状态。必须重新渲染并检查全部输出页和裁图，不能继承本次 PDF 的视觉通过结论。
