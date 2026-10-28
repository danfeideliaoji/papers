# P117 中文全文翻译

主文件：`P117_Copula_中文全文译文.pdf`。

- `translation.zh.md`：可编辑全文译稿。
- `build.json`：源文件哈希、输出与 19 个图表/公式裁图的页码、坐标、宽度。
- `assets/`：5 幅图、11 张表、2 份完整 Gumbel 推导及 1 份 Student t 密度公式，均为 3 倍渲染原图。
- `LICENSE.md`：作者/正式出处、CC BY 4.0 和逐图署名例外审查。
- `QA.md`、`qa/`：逐页可视检查记录及最终页面渲染。
- `build_report.json`：构建报告。
- `SHA256SUMS.txt`：核心文件校验和。

## 完整性

覆盖原文 19/19 页：摘要，正文 1-5 节，算法 1，定理 3.1，编号公式 (1)-(2)，全部未编号公式，致谢、影响声明，56 条参考文献，附录概览和 A-F 全部小节。5 个图题、11 个表题全部翻译，原图英文标签在中文说明中对应；56 条文献提供中文题名并保留检索信息。不存在用摘要替代正文的章节。

## 构建

从 `papers-cloud` 根目录运行：

```sh
PYTHONPATH=/tmp/cloud-pdf-deps python 中文翻译/scripts/build_cloud.py 中文翻译/P117_Copula/build.json
```

版式约束直接记录在 Markdown 中，按上述命令即可重建。

## 忠实性说明

保留原文中的科学记号和可见不一致，包括 A.1 的数据量与表格差异、Frank CDF 的指数与括号、Gaussian 密度系数、三元 Gumbel 推导等。针对若干显著问题添加了明确的译者注；没有静默“纠错”。参考文献的实际复合词连字符（如 multi-modal、self-supervised、Loaiza-Maya、Hernandez-Boussard）保留。

仅使用 DOT 云端文件和工具完成；未调用外部翻译服务、用户桌面或个人 API 密钥，未推送 GitHub。
