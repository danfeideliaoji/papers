# P136 鞅一致的自监督学习 中文译文

主交付文件：`translation.zh.pdf`。可编辑完整译文：`translation.zh.md`；正文、书目、附录的分文件分别为 `main.zh.md`、`references.zh.md`、`appendix.zh.md`。

- 原文：Moritz Gögl, Hanwen Xing, Christopher Yau, *Martingale-Consistent Self-Supervised Learning*, arXiv:2605.11846v1，2026-05-12，22 页预印本。
- 原文及许可证据：https://arxiv.org/abs/2605.11846v1
- 原始 PDF：https://arxiv.org/pdf/2605.11846v1
- 许可：https://creativecommons.org/licenses/by/4.0/
- 核验来源：`source-verification.json`（继承根任务已完成的官方许可、版本同一性及图像权利审查）。原 PDF 与仓库版本的 SHA-256 完全相同：`ece0e4d192c1c8a962bcc2b0ad0623db19fcea32ce6d7f6949162546dd8795a0`。

## 范围及重要省略说明

正文 1–6、摘要、致谢、全部 28 条参考文献、附录 A–F（包括证明、数据协议、超参数和全部 32 个编号公式）均已译出。公式和符号用 LaTeX 转为矢量数学图形；PDF 文字提取不包含这部分数学字形，精确可编辑数学源码保留在 Markdown 中。参考文献保留原文题名与检索信息并附中文题名。

图 1–3、5–7 完整保留，图 3 另有三个局部放大裁片；四张表保留原始数值与强调标记，图注、表注及英文术语键均译出。**图 4 只保留 B 面板；A 面板含第三方 CIFAR 示例照片，其再分发权限未确立，因此省略，未制作替代照片。**完整图注与涉及 A 面板的文字仍译出，并链接原始 PDF 第 9 页。不能描述为“全部图片完整保留”。

图内英文和原表单元格以原始裁片形式保留，中文对照见相邻图注/表注。由于重排和图表放大，译文分页不同于原文。译文非官方、由 AI 辅助制作，未获作者审定；研究结果及原文自身的假设、局限和表述均按原文保留，不把翻译当作独立复现或验证。

## 构建与核验

仅在云端工作区制作，没有 GitHub 提交、公开发布或桌面写入。原始 PDF 未改动。共享 `../scripts/build_cloud.py` 未改动。

```bash
python assemble.py
PYTHONPATH=/tmp/cloud-pdf-deps python ../scripts/build_cloud.py build.json
python qa/check.py
```

`assemble.py` 合并译文并为带变音符号的人名局部指定 Noto Serif 字体；`build.json` 记录源文件校验值和全部裁切坐标。`assets/` 包含 14 个原图/原表裁片；`manifest.json` 记录交付文件、许可、覆盖范围和校验值；`review.md` 与 `qa/` 记录逐页视觉审查、字形检查、源裁片像素对照和构建证据；`SHA256SUMS` 便于复核交付文件。

原文页图仅用于内部审阅，审阅后未纳入交付；中文图像资产不包含图 4A。原始 PDF 作为未改动的来源文件保留。
