# 许可、署名及逐图审查

审查日期：2026-10-26 UTC。本文件仅记录该译件的来源和许可核验，不修改项目级 rights_review.csv。

## 来源及适用许可

Wei, Yake & Hu, Di. (2024). **MMPareto: Boosting Multimodal Learning with Innocent Unimodal Assistance**. Proceedings of the 41st International Conference on Machine Learning, PMLR 235:52559–52572.

- 官方论文页：https://proceedings.mlr.press/v235/wei24d.html
- 官方页链接的 PDF：https://raw.githubusercontent.com/mlresearch/v235/main/assets/wei24d/wei24d.pdf
- 出版协议：https://proceedings.mlr.press/pmlr-license-agreement.html
- CC BY 4.0：https://creativecommons.org/licenses/by/4.0/ （法律文本：https://creativecommons.org/licenses/by/4.0/legalcode）
- PMLR 出版协议第 2 条授予公众 CC BY 4.0 许可；第 3 条要求署名包含原始出版引文以及通向 PMLR 原论文页的链接。
- 原 PDF 首页面脚注明 Copyright 2024 by the author(s)。作者保留著作权。
- 下载官方 PDF 并计算 SHA-256，与本地源文件及 rights_review.csv 三者一致：`e571649b6e1245dd2175be90545002180bf67ddd2b871318da17cc9f67cb0bfd`。详见 official_source_verification.txt。

本中文译文依据上述许可制作。修改包括中译、重新排版、原图表裁切和中文释义、公式重新排版及显式标明的译注。未改写实验数据或把译注冒充原文。此为非官方改编，不暗示原作者认可。原文、原图版权归原权利人；本译文以 CC BY 4.0 提供。

## 逐图及表格例外核验

逐一检查源 PDF 所有 6 幅图（含子图）、9 张表、算法 1 的图内文字、图注、邻近说明和致谢；未发现单独的“保留所有权利”、禁止改编或排除出文章许可的署名行。学术来源引用按原文完整保留，不因没有单独限制而宣称底层数据集获得新的许可。

| 项目 | 原页 | 来源/署名情况 | 处理 |
|---|---:|---|---|
| 图 1(a)(c)(d) | 2 | Kinetics Sounds，引用 Arandjelovic & Zisserman (2017)；本文模型与梯度实验 | 依文章许可保留图像；中译图注，保留来源 |
| 图 1(b) | 2 | 图注明确写结果来自 Sener & Koltun (2018)，数据集为 Cityscapes/Cordts 等 (2016) | 保留该明确来源署名；未发现单独许可排除；属于文章中所呈现的实验结果图，不另行抓取第三方原图 |
| 图 2 | 4 | 本文多模态框架/算法示意，无第三方署名行 | 依文章许可保留，中文解释所有标签 |
| 图 3 | 5 | 本文公式函数图，无第三方署名行 | 依文章许可保留，中文图注 |
| 图 4 | 6 | 本文 CREMA-D 梯度实验，正文引用 Cao 等 (2014) | 依文章许可保留，中文图注及标签释义 |
| 图 5 | 7 | 本文实验景观；可视化方法注明 Li 等 (2018) | 保留该方法署名；未发现单独许可排除 |
| 图 6(a–d) | 12 | MultiMNIST/MNIST 数字样例；正文引用 Sabour 等 (2017)、LeCun 等 (1998) | 原文章图中有限样例按文章许可保留；不分发或重新许可底层数据集 |
| 表 1–5、7 | 7–9、13 | 本文实验结果；比较方法作者年份均保留 | 原表裁切，中文标题及全部表头释义 |
| 表 6 | 13 | 本文记号表 | 全部 24 行译为中文；原文范数/基数符号差异注明 |
| 表 8–9 | 14 | 本文计算时间与迭代次数 | 全部数据重排为中文表 |
| 算法 1 | 6 | 本文方法 | 全部输入、循环、条件与更新步骤中译；零范数分支未明确停止的问题标注 |

结论：文章官方 CC BY 4.0 授权及逐图信用行审查已完成，未发现明确的图级许可排除。此结论限于本论文改编，不是对所引用论文/数据集的独立再许可。
