# P114 CGGM — 非官方中文全文译文

交付：`P114_CGGM_中文全文译文.pdf`；可编辑全文：`translation.zh.md`。

底本为明确按 CC BY 4.0 发布的 arXiv:2411.01409v1，不是仓库内未核实许可的会议版本。详细证据及图形例外检查见 `source-license.md`。原文件仅下载到本目录，未覆盖任何原论文。

## 完整性

摘要、正文1–5节、致谢、28条参考文献、附录A–C；算法1；公式1–15；图1–7；表1–8。该版本无独立定理/证明章节。图表原始数值保留，所有图注及表头词汇提供中文说明；参考文献完整原始条目与题名中文译文并列。

## 云端复现

在仓库根目录运行（共享构建器未改动）：

```bash
PYTHONPATH=/tmp/cloud-pdf-deps python 中文翻译/scripts/build_cloud.py 中文翻译/P114_CGGM/build.json
python 中文翻译/P114_CGGM/render_qa.py
```

依赖：pandoc、PyMuPDF、Pillow、Noto Serif/Sans CJK SC、WeasyPrint 70.0、ziamath 0.13。`build.json` 固定底本哈希与所有裁图范围。构建程序先校验哈希，再生成 `assets/` 与 PDF。`qa/` 保留最终版所有页面PNG及联系表；`qa_review.md` 记录检查结果。

署名及许可说明已置于译文首页。中文改编亦采用 CC BY 4.0，非作者官方译文，不表示作者背书。未使用桌面、本地用户电脑或外部翻译服务。共享根CSV和原始论文保持不变。
