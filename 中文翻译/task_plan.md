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
- [x] 保存首批提交并推送分支（2365d8e6664ffb7703b2dc188da2dab2cf1154c3）
- [ ] 建立草稿 PR：GitHub integration 返回 403 Resource not accessible by integration，已停止此动作，未换入口重试
- [x] 核验远端提交与可下载 PDF（精确提交下载，SHA-256 与本地一致）
- [ ] PR 内容核验：需先解除 PR 创建权限阻塞；当前没有 PR

## 继续方式

先阅读本文件、`progress.json`、`inventory.csv`、`rights_review.csv`、`existing_translations.md`。
使用本地 Python（pypdf、pdfplumber、reportlab、Pillow）与 Poppler；不调用付费翻译服务，不新增密钥，不向第三方翻译服务上传论文。
所有新文件写入本目录。原始 PDF、已有译本、源码和数据一律保留。
缺原文、许可未确认、图片权利未确认分别记录，不能将可下载/arXiv 默认许可/代码 LICENSE 视为全文翻译授权。
每批更新清单和校验记录再提交。只有明确允许改编和再分发且版本已核实的内容进入公开交付。

## 当前外部阻塞

目标：向 `danfeideliaoji/papers` 的 main 建立 head=`translations/zh-pdf-audit-20261025` 的草稿 PR。连接工具返回 403，原因是 integration 无此资源权限；查询确认无开放 PR。分支、完整样例、清单均已推送，不受影响。需要用户在有权限的 GitHub 会话中手动建立草稿 PR，或为当前 GitHub 连接补齐仓库 PR 写权限后再继续；不索要 token，不擅自改权限，不换工具绕过。
