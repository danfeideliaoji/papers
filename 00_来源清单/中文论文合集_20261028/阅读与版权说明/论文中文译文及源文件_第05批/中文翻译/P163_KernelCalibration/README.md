# P163：基于核均值嵌入的神经网络可训练校准度量

完整非官方中文译文（正文和官方补充材料），只在 dot 云端工作区制作。并非 P113《On Calibration of Modern Neural Networks》。

## 交付物

- `P163_KernelCalibration_中文全文译文.pdf`：21页，已逐页渲染和视觉检查。
- `translation.zh.md`：完整中文文本；数学、图表通过相对资产引用保存。
- `assets/`：26个300%分辨率原版裁图（3图、6表、17个公式/证明数学块）。
- `build.json`：可复现构建清单，含原始输入哈希和每个裁图坐标。
- `source/official-main.pdf`：官方10页正文，与用户提供的原始PDF逐字节哈希一致。
- `source/official-supp.pdf`：官方2页补充材料；全部纳入译文。
- `evidence/`：官方页面及许可依据；`source_license.json` 记录来源、许可、图表例外审查。
- `qa/`：12张原文页面图、21张最终输出页面图、完整裁图边距记录和 QA 报告。
- `SHA256SUMS.txt`：交付目录内各文件的 SHA-256（清单自身除外）。

## 内容覆盖

正文摘要、1–7节、定理1–3及证明/证明指向、式(1)–(12)、图1–3、表1–4、脚注、致谢、46条参考文献；补充材料1–4节、定理2和定理3的全部证明、补充式(1)、补充表1–2及2条参考文献。原版图表的全部英文标签均在相邻中文图注/表注给出对应译名；数值与公式从原PDF直接裁取，避免重排失真。

## 许可与原文问题

作者：Aviral Kumar、Sunita Sarawagi、Ujjwal Jain。原文发表于 ICML 2018，PMLR 80:2805–2814。出版链接：<https://proceedings.mlr.press/v80/kumar18a.html>。PMLR 官方出版协议规定 CC BY 4.0；版权仍属作者。中文翻译及重新排版已明确标注为改编；不暗示作者/出版方认可。逐页检查未发现图1–3或表1–4、补充表的独立第三方版权署名或排除声明；它们为本文实验曲线、数值表。数据集和参考代码仍归各自权利人，本包不分发数据集或代码。

原文式(8)正号、式(9)变量文字与分母不一致、补充证明的原有表述均保留，并作译者说明；没有静默改写原作者论证。

## 复现

从 `papers-cloud` 根目录执行：

```bash
PYTHONPATH=/tmp/cloud-pdf-deps python 中文翻译/scripts/build_cloud.py 中文翻译/P163_KernelCalibration/build.json
python 中文翻译/P163_KernelCalibration/verify.py
```

共享脚本未改动。运行 `make_manifest.py` 会重新生成初始裁图坐标；如确需重建坐标，随后运行 `expand_crops.py`，保证每个完整字形至少2pt边距。直接复现交付版本应使用已审核的 `build.json`。公式编号保留原印刷数字，不使用 `\tag`。

## QA

- 主文输入 SHA-256：`58aa61b016426e255b14ca7160a2ca701d87ee9562d313847f05fe1fe7f5d39a`
- 官方补充材料 SHA-256：`cda706c170a6f579d8fbadc3dc7f2e92e23c00e78a7837ae556673c9a95bc13d`
- 最终PDF SHA-256：`9f9e7e03356dac2ff049ce284860fe35609027037a14f4c72f39809341a94f3c`
- Markdown SHA-256：`155f052b2aac87ab56a8c66cbc67e2b734464829b10d102db9f5a628a29b15d4`
- 所有含可提取字形的裁图最小完整字形边距≥2.49pt；栅格图2、图3像素检查最小外缘余白分别≥4.66pt、≥3pt。
- 21页最终输出全部审核，无页边界越界、无缺失资产、无未使用资产、无 `\tag` 命令；图表与中文说明不跨页分离。
