# 中文全文 PDF 翻译进度（可恢复）

基线：`main@c9839663157a00ec3160ae1521757d5a4184caba`。
授权：用户允许在本仓库新建分支、提交、推送并向 main 建立草稿 PR；不合并、不改权限。

- [x] 检查既有分支与 PR（初始仅 main，无开放 PR）
- [x] 检查仓库指令（基线树无 AGENTS.md、.agents/skills）
- [x] 克隆并保留原始文件
- [x] 逐篇盘点 174 条目、全部 PDF 页数与哈希
- [x] 审核既有中文译本，记录复用状态（初审，见 existing_translations.md）
- [ ] 核实逐篇原文许可证及图片例外；未确认不得公开全文翻译
- [x] RCSC：比对官方版、记录 CC BY 4.0 证据
- [x] RCSC：中文全文、公式、图表、参考文献与可复现源文件
- [x] 渲染逐页检查并记录完整性
- [ ] 保存首批提交、推送并建立草稿 PR
- [ ] 核验远端提交、PR 文件与可下载 PDF

## 继续方式

先阅读本文件、`progress.json`、`inventory.csv`、`rights_review.csv`、`existing_translations.md`。
使用本地 Python（pypdf、pdfplumber、reportlab、Pillow）与 Poppler；不调用付费翻译服务，不新增密钥，不向第三方翻译服务上传论文。
所有新文件写入本目录。原始 PDF、已有译本、源码和数据一律保留。
缺原文、许可未确认、图片权利未确认分别记录，不能将可下载/arXiv 默认许可/代码 LICENSE 视为全文翻译授权。
每批更新清单和校验记录再提交。只有明确允许改编和再分发且版本已核实的内容进入公开交付。
