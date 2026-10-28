# P159 TACVI 私人中文全文译本

已完成用户提供的 16 页原稿的全文翻译，输出为 28 页 A4 PDF。正文不是摘要或节译；保留 5 幅原图、5 张原表、14 个编号公式及算法 1，并提供中文题注、图中文字释义、算法译文与 54 条参考文献的中文题名。

## 文件

- `P159_TACVI_中文全文译文.pdf`：经逐页检查的最终 PDF。
- `translation.zh.md`：可编辑的完整中文译稿；`{{asset}}` 为构建时展开的原稿图像占位符。
- `assets/`：27 个核验过的原稿裁切图像；包含 5 图、5 表、14 公式、1 算法及 2 个特殊符号。
- `build.json`：原稿哈希、裁切坐标、显示宽度和输出路径。
- `build_report.json`：最终构建报告；`QA.json`、`QA.md`：质检证据与限制。
- `coverage.json`、`coverage.md`：原稿各页到译本的内容覆盖映射。
- `source/user-upload.pdf`：未修改的用户原稿；`provenance.json`：来源与权利边界。
- `SHA256SUMS`：交付文件与图像的 SHA-256 校验和。

## 来源与使用边界

原稿来自用户上传的 Pro_Residual_Exploration ZIP，父任务已确认其与仓库对应原稿一致。本子任务仅处理指定的用户提供 PDF，没有以公开网页替换它，也没有推断公共许可。原稿 arXiv 标识、预印本投稿说明、两个不同原稿日期、作者、机构和通信邮箱均保留。该翻译为私人、非官方使用，不表示公开发表或额外转载授权。原文权利归原权利人。

所提供 PDF 不含附录、独立证明或作者简介，也未见指向另行提供补充材料的声明；没有编造这些内容。译注明确区分原文与译者说明，保留原稿交叉熵符号、协方差字形、算法/正文记号、图 3 叙述等不一致。

## 排版与数学

编号公式直接来自原 PDF，以保存花体、斜体、粗体、上下标和特殊字形，尤其是第一阶段特殊编码器/分类器符号、花体 P/Q、第二阶段双线体 C。原图内的字形不被重新绘制。正文内联数学采用 MathML 转 SVG；不将原稿普通斜体 R 擅改为双线体 R。表 3、表 4 按完整模型列拆为上下两块，并重复数据集和指标列，所有数值及原有粗体/下划线保留。

中文正文和参考文献可搜索/提取；图像内的表格文字和编号公式不保证可搜索。参考文献中的 Müller 使用局部 Noto Serif 字体，并已在 PDF 第 28 页验证提取正确。原稿自身排印/数学问题未被静默修正。

## 云端重新构建

私人交付包保留以下目录关系：

```
中文翻译/
  scripts/build_cloud.py
  scripts/cloud-pdf-requirements.txt
  P159_TACVI/build.json
  P159_TACVI/translation.zh.md
  P159_TACVI/source/user-upload.pdf
  P159_TACVI/assets/...
```

在上述 `中文翻译` 的父目录运行：

```bash
PYTHONPATH=/tmp/cloud-pdf-deps python 中文翻译/scripts/build_cloud.py 中文翻译/P159_TACVI/build.json
```

构建要求：Python、PyMuPDF、Pillow、pandoc、WeasyPrint、ziamath，以及 Noto CJK／Noto Serif 字体；精确 Python 依赖见交付包的 requirements。构建器会核验输入 PDF 的 SHA-256，且不覆盖原稿。更换渲染器/字体版本后需重新逐页检查；旧 QA 不自动适用于新的 PDF。

制作过程完全在云端完成，未使用桌面，未执行上传压缩包或原项目内的代码。使用的仅为指定的私人文档构建器及此次新编写的文档处理脚本。
