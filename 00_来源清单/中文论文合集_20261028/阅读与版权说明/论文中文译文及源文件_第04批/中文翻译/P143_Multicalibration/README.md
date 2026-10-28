# P143 多重校准：中文全文译文

- `translation.zh.md`：完整合并译文（正文 10 页 + 官方补充材料 7 页）。
- `main.zh.md`、`supplement.zh.md`：按来源分开的译文。
- `P143_Multicalibration_中文全文译文.pdf`：排版后的完整译文。
- `build.json`：共享 `scripts/build_cloud.py` 使用的构建清单；`assets/` 保留算法和复杂公式裁图。
- `source/`：已核对哈希的官方原文、官方补充材料，以及出版页和许可协议证据。
- `qa/`：逐页渲染、视觉检查与版权核查记录。

构建：`PYTHONPATH=/tmp/cloud-pdf-deps python 中文翻译/scripts/build_cloud.py 中文翻译/P143_Multicalibration/build.json`（仓库根目录运行）。

完整覆盖摘要、章节 1–6、算法 1 全文、脚注、致谢、36 条正文参考文献，以及补充材料 A.1–A.4、B.1–B.2、C 的所有证明和 6 条补充参考文献。原稿没有普通图或数据表。保留原文中的标量下标、星号和花体集合符号；没有人为将普通变量改为粗体向量。

许可及改编说明见 `LICENSE.md`。本目录仅在 dot 云端工作区制作；未使用用户桌面、外部翻译服务或 GitHub 写操作。
