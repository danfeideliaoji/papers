# P150 中文全文译文

- 成品：`P150_SemanticInvariance_中文全文译文.pdf`（22 页）。
- 可编辑稿：`translation.zh.md`；可复现构建：`build.json`。
- 工作环境：仅 dot cloud `/workspace/shared/papers-cloud`。未使用用户桌面、cloud_threads、外部翻译 API 或 GitHub 写操作。
- 源论文：Chengliang Liu, Gehui Xu, Jie Wen, Yabo Liu, Chao Huang, Yong Xu. *Partial Multi-View Multi-Label Classification via Semantic Invariance Learning and Prototype Modeling*. ICML / PMLR 235, 2024.
- 官方页面：https://proceedings.mlr.press/v235/liu24bv.html
- 官方 PDF：https://proceedings.mlr.press/v235/liu24bv/liu24bv.pdf
- 许可依据：https://proceedings.mlr.press/pmlr-license-agreement.html ，CC BY 4.0；https://creativecommons.org/licenses/by/4.0/ 。版权归原作者，中文稿为非官方翻译改编，保留署名与来源。
- 源 SHA-256：`18be622c587f3b1346801eb6c092e9aeef04acd93604191052830955f02c493d`。与 `rights_review.csv` P150 官方版本核验记录相符。

## 内容覆盖

原文 15 页全部覆盖：摘要；1–5 节及所有子节；命题 2.1、推论 3.1；算法 1 的 13 行；致谢、影响声明；40 条完整英文参考文献；附录 A–G（包括全部推导、PoE 融合/采样、数据集/基线统计、补充实验、实现细节、局限性）。正文约 7610 个汉字（含编辑说明）。

保留 25/25 个编号公式（原图裁切，编号、积分、上下标不重新排写），5/5 幅图，4/4 张表。图表标题和图例词语附中文，数值与图内原始英文保留便于核对。算法为可编辑中文步骤。非摘要替代全文。

## 图表权利审查

逐一检查原 PDF 图 1（作者方法示意）、图 2–5（作者实验绘图）、表 1–4（实验及方法/数据统计）与相邻图题、表题，未见独立第三方版权、转载来源、禁止改编或另行许可说明。图 1 无外部照片/艺术作品；图 2–5 为数据图，未发现署名例外。本结论为对原文可见署名的核查，不是对数据集原始图像使用许可的扩展授权；译文未复制数据集原始图像。

## 编辑与 QA

见 `review.md`、`qa_report.json`。原文疑似笔误均显式注记，未静默改动公式：式（12）负号缺失、算法 1 均值编码器、4.5 节 β 权重位置、比较方法数量/“全部最佳”的表述、附录 E 数据集标题和图 5b 标注。采样表示按原式保持 z⁰、lᵢ⁰。

原文页对照：1–2 摘要/引言/预备知识；3 式1–7；4 图1/式8–9；5 式10–13及分类/设置；6 表1与设置；7 图2/算法/结果；8 表2/图3–4/消融；9 结论/致谢/影响声明及参考文献起始；10–11 其余参考文献；12–13 附录A与式14–24；14 附录B–F、表3–4与式25；15 图5/附录G。

重建命令（在仓库根目录）：

```bash
PYTHONPATH=/tmp/cloud-pdf-deps python 中文翻译/scripts/build_cloud.py 中文翻译/P150_SemanticInvariance/build.json
```
