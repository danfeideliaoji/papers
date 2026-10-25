# 中文论文 PDF 翻译（首批可恢复检查点）

已完成 **1 / 174 篇**：[RCSC 中文全文 PDF](RCSC/RCSC_中文全文译文.pdf)（16 页）。这是中文全文，含定理证明、34 个编号公式、3 幅图、3 张表、算法及完整参考文献。原文图表与公式保留，配中文说明；参考文献保留英文书目。详见 [源稿与复现说明](RCSC/README.md)、[渲染审核](RCSC/review.md) 与 [完整性核验](RCSC/qa_report.json)。

所有新文件集中在本目录；原始论文、已有译本、源码与数据均保留。当前基线 `main@c9839663157a00ec3160ae1521757d5a4184caba`，工作分支 `translations/zh-pdf-audit-20261025`。本检查点不表示其余 173 篇已经翻译。

分支已推送，精确提交的 PDF 已重新下载核验。创建草稿 PR 的连接操作被 GitHub 以 `403 Resource not accessible by integration` 拒绝，当前没有 PR；没有换工具或关闭保护绕过。待当前连接具备 PR 写权限后再执行，或由用户在 GitHub 手动建立草稿。预备说明见 `pr_draft.md`，准确状态见 `remote_status.json`。

## 盘点

| 范围 | 数量 |
|---|---:|
| 论文清单条目 | 174 |
| 有正文 PDF | 138 |
| 正文总页数 | 2,169 |
| 缺正文 | 36 |
| 所有原始 PDF（含代码附带文档、旧稿和已有译本） | 272 |
| 所有可读取 PDF 总页数 | 2,719 |
| 异常 PDF | 2（均为代码附带 ICD 文档；不在 138 篇正文内） |

`inventory.csv` 逐篇记录 ID、来源路径、页数、字节、SHA-256、缺项和翻译阶段。`all_original_pdfs.csv` 包括每个 PDF，读取失败的文件保留错误信息。`baseline_pdf_sha256.json` 与 `repository_verification.json` 可验证原文件未变。

初始仓库没有 `AGENTS.md`、`.agents/skills` 或顶层 LICENSE；远端初始只有 main，无开放 PR。代码目录的许可证未被视为论文许可。

## 许可审核与可执行队列

本轮逐篇登记了 174 条许可状态；其中许多仍是待确认，**没有宣称全部论文已完成许可核验**。

- RCSC：官方 PDF 与仓库副本逐字节一致，按 PMLR CC BY 4.0 出版协议完成中文全文样例，已附作者、来源、许可及非官方译文/修改声明。
- 另 17 篇的论文许可已确认，等待逐图检查后进入翻译：16 篇 PMLR 正式版与仓库 SHA-256 相同，另有 P131 的文章专属 CC BY 4.0 声明。图像许可和补充材料范围仍需逐篇审查。
- P065、P082：找到允许改编的 PMLR 正式版，但与仓库 PDF 哈希不同；应比对后在本目录使用正式版，不将其许可擅自套用到不同版本，也不替换原文件。
- P123、P141、P149、P175：原文明确为 CC BY-NC-ND 4.0，公开中文全文需另行改编/再分发授权。
- 其余 150 篇的论文改编与再分发许可尚未确认（其中 AAAI 出版政策要求再出版许可）。可下载、可免费阅读、arXiv 默认分发许可、代码开源和已有译文均不能替代论文授权。

完整状态见 `rights_review.csv`。`license_notices.json` 是从原始 PDF 提取的许可/版权提示，不自动当作许可结论：例如，数据集 CC BY 或参考文献中的许可文字不覆盖论文本身。正式版本核验的 URL、时间和双方哈希见 `evidence/official_version_checks.json`。

主要证据：[PMLR 出版协议](https://proceedings.mlr.press/pmlr-license-agreement.html)、[RCSC 官方页](https://proceedings.mlr.press/v267/qin25e.html)、[P131 文章权利声明](https://link.springer.com/article/10.1007/s00362-023-01468-3)、[AAAI 再出版规则](https://ojs.aaai.org/index.php/AAAI/about)。

## 既有材料与继续工作

两份既有中文 PDF 以及 RANK、HyperAHSF、V2L Markdown、URDF 中文 DOCX 已登记，见 [复用审核](existing_translations.md)。没有重复生成或改动这些文件。MvCGE 不能按完整全文计数；VAMS 可保留为校对底稿，但需先确认公开发布和第三方素材权限。

下一批优先从 `licensed_queue_figure_review_pending` 条目选取，先逐项核实图像权利和全文范围，再翻译、渲染、校对并提交。36 篇缺原文与权利待确认条目继续排队；不因这些条目受阻而停止许可明确的论文。

若要解除明确的版权阻塞，最小补充是：该论文允许公开中文翻译与再分发的许可证链接，或权利人的授权（含图像适用范围）。无需密码、token 或新 API 密钥。未授权合并 PR、修改仓库权限或使用付费翻译服务。

恢复时先读 `task_plan.md` 和 `progress.json`。`scripts/inventory.py` 可重建盘点并保留同一哈希下的人工许可结论；`scripts/verify_sources.py` 仅下载已知官方地址用于版本核验，不上传论文。输出 PDF 在本地生成，未使用外部翻译服务。
