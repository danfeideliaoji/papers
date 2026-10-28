# 许可、出处与改编说明

- 作品：Cross-Modal Alignment via Variational Copula Modelling。
- 作者：Feng Wu, Tsai Hor Chan, Fuying Wang, Guosheng Yin, Lequan Yu。
- 出版：Proceedings of the 42nd International Conference on Machine Learning, PMLR 267:67128-67146, 2025。
- 正式发表页：https://proceedings.mlr.press/v267/wu25d.html
- 许可依据：https://proceedings.mlr.press/pmlr-license-agreement.html ，第 2 条授予公众 CC BY 4.0，第 3 条要求引用正式发表信息并提供原论文 PMLR 链接。
- 许可：https://creativecommons.org/licenses/by/4.0/ ，法律文本：https://creativecommons.org/licenses/by/4.0/legalcode
- 本地原文 SHA-256：97afe69ed7db65233a9bdd0176a4ee0479ab554cab1267c19237054615c80bf0。
- 对照 `../rights_review.csv`：P117 的标题、路径及哈希一致；登记注明与官方 PDF 相同。2026-10-26 另访问正式发表页和出版协议，核对作者、篇名、页码及许可条款。

## 逐图与第三方署名例外检查

检查原文全部 19 页及图 1-5、表 1-11 的图题、表题和邻近文字，没有发现单独的“转载自”“经许可使用”“不在 CC 许可内”等第三方署名或许可排除声明。

|对象|原页|检查结果|
|---|---:|---|
|图1|3|作者框架示意，内含小型时间序列、文档、CXR 缩略图；未见独立来源署名或排除声明。作为原文整体图示保留，未提取或发布独立医学影像数据。|
|图2、3|8|作者拟合密度及训练参数曲线，无第三方署名例外。|
|图4|16|作者训练轮次密度图，无第三方署名例外。|
|图5|18|作者实验性能曲线，无第三方署名例外。|
|表1-11|5-8、13、18-19|作者整理的样本数、实验数据与统计结果；方法引用保留，无独立许可排除声明。|
|公式和推导|3-5、13-16|原数学内容；复杂推导与 Student t 密度保留准确裁图。|

该检查记录的是发表版本中可见的署名和例外，不另行声称拥有底层 MIMIC 数据或第三方软件的权利。本包不含原始患者数据或模型代码。

## 改编与署名

本包为非官方中文翻译与重排版，未经原作者审阅或认可。原作者保留原作品版权；译文注明了翻译、中文标签说明、版式变化，以及少量明确标记的译者注。图表、复杂推导保留原图；参考文献保留原文题名、作者和出版信息并加中文题名。译文及本包改编内容按 CC BY 4.0 提供。没有将原文疑似笔误静默改写成新的科学结论。
