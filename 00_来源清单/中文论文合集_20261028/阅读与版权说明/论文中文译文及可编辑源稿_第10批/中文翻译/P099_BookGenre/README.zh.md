# P099 中文译本

已完成 arXiv:2505.03839v2 的10页主文全译，重排为23页中文PDF。正文、5个编号公式、4张表、36条参考文献和原文补充附录指引齐全。

- 主交付：`translation.zh.pdf`；可编辑全文：`translation.zh.md`。
- 图1的书封/引述简介、图2左侧第三方输入示例、表IV的书封/引述简介明确省略。作者图注与分析完整翻译，书目事实与全部预测结果保留。作者图2算法主体及图3–5保留，共6个裁片。
- 独立外链补充PDF未建立与主文相同版本的改编许可匹配，未翻译或再分发。核查见 `source-archive-audit.json`。
- 主文来源：https://arxiv.org/abs/2505.03839v2 ；许可：https://creativecommons.org/licenses/by/4.0/ 。作者署名、许可和改动说明在PDF/Markdown首页。
- 公式已与官方TeX核对。原式(5)的标签系数与第III节定义存在表面不一致；本译本保留原式并加注，不静默修正。
- `coverage.json` 是覆盖表；`qa_report.json` 与 `content-verification.json` 是检查记录。`qa/` 中仅有本中文译本的内部逐页渲染与抽取文字。
- 中文正文可检索；PDF公式采用矢量绘制，精确LaTeX保存在Markdown。
- `build.json` 可由上级共享的 `scripts/build_cloud.py` 重建（`PYTHONPATH=/tmp/cloud-pdf-deps`）。该流程读取既有原稿但不改写。原稿未加入交付清单；需要重建时应按版本来源自行取得相同SHA256文件。
- 交付仅限 `manifest.json` 的 `distribution_files` 白名单；不得把 `source/`、原文源码包、原文整页渲染或未授权补充文件加入新交付包。

所有工作仅在云端完成；未上传GitHub或任何公开站点。
