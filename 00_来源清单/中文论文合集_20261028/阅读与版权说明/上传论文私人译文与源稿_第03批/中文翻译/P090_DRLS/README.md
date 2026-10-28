# P090 DRLS 中文全文译文

- 最终 PDF：`translation.zh.pdf`（17 页）
- 可编辑译稿：`translation.zh.md`
- 原稿：`source/user-upload.pdf`（10 页）；上传来源证据：`source/user-upload-provenance.json`
- 构建：`build.json`；原稿页面及裁切准备：`prepare.py`
- 27 个原图资产：`assets/`（18 个公式、2 幅图、6 个表格片段、1 个版本声明）
- 覆盖核对：`COVERAGE.md`；视觉检查：`QA.md`；机器核验：`qa/verification.json`
- 最终页面渲染：`qa/final-pages/page-01.png` 至 `page-17.png`
- 原稿页面渲染：`qa/source-pages/page-01.png` 至 `page-10.png`
- 文件校验：`SHA256SUMS`

仅供用户私人阅读；保留底本权利与版本声明，不新增公开再分发许可。原文提及但用户未上传的补充材料 6.1、6.2 和 7 节不在译文范围，已明确披露，未作补写。

## 重建

在现有私有云工作目录中运行以下已审阅的自有构建程序。不要运行上传压缩包或论文仓库中的代码。

```bash
cd /workspace/shared/private-papers
PYTHONPATH=/tmp/cloud-pdf-deps python 中文翻译/scripts/build_cloud.py 中文翻译/P090_DRLS/build.json
python 中文翻译/P090_DRLS/verify.py
```

共享构建器使用 Pandoc、PyMuPDF、Pillow、WeasyPrint 70.0、ziamath 0.13、Noto Serif CJK SC、Noto Sans CJK SC 和 Noto Serif。构建器先验证源文件 SHA-256，再依据受限清单提取原图，渲染 Markdown / MathML；不会改变原稿。`prepare.py` 可重建初始清单与原稿页面渲染，`build.json` 是最终裁切和宽度的权威记录。

重建之后需重新逐页视觉检查，并更新 `build_report.json`、`QA.md` 与校验清单，不能直接继承本次的视觉审核结果。
