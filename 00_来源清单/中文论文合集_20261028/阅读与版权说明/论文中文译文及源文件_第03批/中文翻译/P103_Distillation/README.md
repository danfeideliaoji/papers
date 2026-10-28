# P103 蒸馏的统计学视角

完整非官方中文译文，覆盖官方主文11页及另附补充材料8页；不是摘要替代。

## 文件

- `P103_Distillation_中文全文译文.pdf`：28页全文PDF。
- `translation.zh.md`：完整可编辑译文，含摘要、正文1–6、56条参考文献、附录A.1–A.3、B.1–B.3、C.1–C.7、全部证明及脚注。
- `build.json`：重建清单，含源文件哈希和14个图表裁切。
- `assets/`：图1–13与表1；保留全部坐标、图例、数值及英文子图说明，译文提供中文对应。
- `source/`：官方主文、8页官方补充、下载授权网页，以及审阅用原文逐页文本/图片。
- `LICENSE.md`、`license_review.json`：CC BY 4.0证据、逐图审查与修改说明。
- `qa/`：最终28页渲染PNG与5张全页联系表。
- `qa_report.json`、`crop_checks.json`、`page_geometry.json`：覆盖与排版检查。
- `qa_check.py`：仅在本目录内生成检查报告与预览。

## 重建

从仓库根目录运行：

```bash
PYTHONPATH=/tmp/cloud-pdf-deps python 中文翻译/scripts/build_cloud.py 中文翻译/P103_Distillation/build.json
python 中文翻译/P103_Distillation/qa_check.py
```

依赖采用既有cloud renderer（pandoc、ziamath、WeasyPrint、Noto CJK）；未修改共享构建脚本。公式使用美元数学定界、粗体向量、原编号1–18，不使用不兼容的tag命令。

## 原文问题保留

译文明确标识引理1证明第二项的N因子与几乎处处限定、命题3单侧界转平方、概率失真函数的实值幂、式(12)负项权重与文字解释、式(14)标签索引/非负性、附录式(15)左端、偏差方差期望的随机性、标签噪声矩阵维度、push-loss归一化与极限、图7重复图例、图8均值与原交叉引用。未将这些内容默改为不同结论。

仅在云端完成。本目录之外的原文、CSV、共享脚本均未修改；没有尝试GitHub写入、外部翻译器或用户桌面操作。
