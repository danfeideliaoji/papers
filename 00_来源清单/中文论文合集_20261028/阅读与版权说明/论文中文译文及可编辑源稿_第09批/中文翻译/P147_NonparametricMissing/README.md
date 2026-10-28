# P147 缺失数据下的非参数分类

## 版本与权利基础

本项目的中文译文以 Torben Sell、Thomas B. Berrett、Timothy I. Cannings 的 **Nonparametric Classification with Missing Data** 的 **73页 arXiv:2305.11672v2（2024年5月2日）** 为底本，包含补充材料全部证明。

- 官方版本页：https://arxiv.org/abs/2305.11672v2
- 官方 PDF：https://arxiv.org/pdf/2305.11672v2
- 许可：Creative Commons Attribution 4.0 International（CC BY 4.0）
- 许可文本：https://creativecommons.org/licenses/by/4.0/
- 底本 SHA-256：`02ed1c44949085eadd0343d487709cdbf5883670a7158cc33ef146f1ba453096`

**重要：仓库原有文件是另一份24页华威大学作者接受稿（AAM），并非本次73页 arXiv v2。两个文件不逐字节相同；本项目不声称它们完全等同。原有24页 AAM 和下载的73页原版 PDF 均不被覆盖或修改。** 原版本核验记录见 `source-verification.json`。

五幅正文图及补充图 S1 均已检查，未发现第三方图像权利排除说明。译文使用原图和原公式的无损裁图，不重绘数据或改变结果。

## 文件

- `P147_NonparametricMissing_中文全文译文.pdf`：最终统一排版 PDF。
- `translation.zh.md`：完整中文 Markdown；数学表达式与原始裁图占位符保留。
- `sections/`：分节译稿，便于核对和修订。
- `assets/`：以3倍 PDF 点分辨率生成的原图、原算法及复杂公式 PNG。
- `build.json`：源文件哈希、输出路径和全部裁图坐标/显示尺寸。
- `assets.main.json`、`assets.s1.json`、`assets.s2.json`、`assets.s3.json`、`assets.s4.json`：各部分的裁图清单。
- `manifest.json`：完整覆盖、来源、输出和逐资产校验信息。
- `coverage.json`：章节、定理/证明、公式与源页覆盖。
- `qa/`：逐页渲染、裁图检查、文本抽取、源文异常记录及质量报告。
- `SHA256SUMS.txt`：交付文件和资产的 SHA-256 校验和。

最终状态以 `manifest.json` 和 `qa/final-qa.json` 为准；`qa/` 中的分节预览不应当作最终整篇文件。

## 署名与变更声明

原作者：Torben Sell、Thomas B. Berrett、Timothy I. Cannings。翻译与重新排版：OpenAI ChatGPT，2026年10月26日。本译文依据 CC BY 4.0 提供，不是作者审定或认可的官方中文版本。

变更包括英文叙述、标题、图注及参考文献题名的中文翻译与版式重排。图中刻度、原始结果、数学符号、公式编号和算法编号保留。原算法完整裁图与逐行中文说明并列；包含英文条件词的公式在相邻正文给出中文。参考文献的作者、刊名、卷页、年份、DOI/arXiv 标识符保留。

底本中可见的数学不一致不擅自修复，包括例3/设定3的系数、交叉验证候选集、若干上下界和轻尾/重尾假设方向；详见译文末尾说明以及 `qa/s2-coverage.md` 的原文异常清单。翻译不是对原始证明正确性的重新认证。

## 覆盖范围

正文第1–5节、全部定义/例子/定理/引理/命题、算法1、数值实验、五幅正文图及经费支持；补充材料 S1–S5、S1–S35 全部编号命题/引理/推论及完整证明、算法 S1、图 S1；42条参考文献。

源页分界精确为：正文1–22；S1为23至30页S2标题前；S2为30页y约108至55页y约435；S3为55页y约447至70页S4标题前；S4/S5及参考文献为70–73。章节分界跨页，不因整页切割而漏译或重复。

## 重建

工作保存在云端目录，未写入用户桌面，也未上传到 GitHub 或公开位置。

```bash
python assemble.py --complete
PYTHONPATH=/tmp/cloud-pdf-deps python ../scripts/build_cloud.py build.json
python qa/render_pages.py
```

渲染链：Pandoc Markdown/MathML → ziamath SVG 数学 → WeasyPrint PDF，正文使用 Noto CJK，含附加符号的拉丁姓名使用局部 Noto Serif。最终 PDF 的所有页面及全部原图/公式裁图均需视觉检查，并复核字符抽取及图像像素同一性。
