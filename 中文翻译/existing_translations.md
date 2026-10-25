# 既有中文材料复用审核

本次检查基线 `c9839663157a00ec3160ae1521757d5a4184caba` 的现有材料，全部保持原路径与原字节。结论是初步内容范围审核，不代表逐公式校对已完成，也不以既有译文存在作为公开发布授权。

| 材料 | 原文 / 现有译文 | 已核验情况 | 复用结论 |
|---|---|---|---|
| MvCGE | 22 页 / 6 页 PDF，并有 DOCX | 译文自述不逐条翻译参考文献和 Paper Checklist；PDF 无图像 XObject；实验段落明显压缩，仅有本文方法的部分结果表；无法覆盖原文全部图表、参考文献与检查表 | 可以参考术语和文字底稿，不能计为中文全文交付；原文逐篇许可仍待确认；本批不重复生成 |
| VAMS | 9 页 / 15 页 PDF，并有 DOCX | 有中文章节、编号公式、图表、算法和英文参考文献，抽取 6,256 个汉字、5 个图像 XObject；首图含 National Geographic 内容，涉及第三方素材 | 可作为后续复核底稿，尚未逐句/逐公式验收；AAAI 要求再出版许可，第三方图片权利也待确认；本批不重复生成、不复制到新目录 |
| RANK | 已有 `RANK_论文中文翻译.md` 和 assets | 发现可编辑中文材料与配图，未发现同名完整中文 PDF | 登记待复用；待许可、版本与全文覆盖核查后再决定编译 |
| HyperAHSF | 已有 `HyperAHSF_论文中文翻译.md` 和 assets | 有中文源稿与配图；仓库清单还标注原文为本地修改版 | 登记待复用；先核对正式版、AAAI 许可和图片权利 |
| V2L | 已有 `V2L_论文中文翻译.md` | 发现中文源稿，无对应已完成 PDF | 登记待复用；需核对 TPAMI 原文许可、版本与完整性 |
| URDF | `code_本地/中文版.docx` | 树中发现中文 DOCX；本次未读取 DOCX 内容，也未将其认定为论文全文译本 | 保留并登记待审核，不重复改写 |

AAAI 官方论文页：https://ojs.aaai.org/index.php/AAAI/article/view/35447 。[AAAI About](https://ojs.aaai.org/index.php/AAAI/about) 明确开放获取并不等于可再出版，版权归 AAAI，再出版需许可。

MvCGE 官方来源：https://proceedings.neurips.cc/paper_files/paper/2025/hash/7315c16422558ae81d65a812723d2cec-Abstract-Conference.html 。已确认来源，尚未确认覆盖该 PDF 的可改编再分发许可，不作许可推定。

精确页数、字节与 SHA-256 见 `existing_translations.csv`。图像 XObject 数是结构检查，不能独立证明图表完整性。保留原文件并记录差异，待权利与质量都核实后再复用。
