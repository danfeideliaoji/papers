# P065 AIRMVC 中文全文翻译

## 使用的版本

采用 PMLR 正式出版的22页版本：<https://proceedings.mlr.press/v267/yang25am.html>。精确源文件单独保存在 `source/yang25am.pdf`。没有修改或覆盖仓库原始论文。

|版本|SHA-256|
|---|---|
|正式出版 PDF|73dd1ba5431b6cd8bb2528bdd860553849390ecd4225d8b1b1d3e45c420beadc|
|原仓库 PDF|5e48f880b6e3784c7c1fde70959477908aefbba02dcaab7854e1752b3e8174fa|

原仓库文件位于 `01_Multi-View/2025-ICML-Automatically Identify and Rectify Robust Deep Contrastive Multi-view Clustering in Noisy/paper.pdf`。两个版本均22页。逐页文字比较显示：正式版第1页无 arXiv:2505.21387v1 标记；正式版第10页新增 Impact Statement，导致第10–12页参考文献重排。第2–9、13–22页提取文本相同。这里的结论仅为提取文本比较，不宣称 PDF 字节或所有底层对象相同。详见 `source/version_diff.txt` 和 `source/version_comparison.json`。

## 文件

- `translation.zh.md`：整合后的完整译稿。
- `main.zh.md`、`refs.zh.md`、`appendix.zh.md`：分段编辑源。
- `build.json`：可复现构建清单，包括源哈希和每个图、表、公式的页码与裁剪坐标。
- `assets/`：300%比例提取的图、表、公式。公式使用原版图像，保留粗体/斜体/花体以及显式原编号。
- `P065_AIRMVC_中文全文译文.pdf`：最终译文。
- `LICENSE.md`：许可及署名、第三方素材检查边界。
- `QA.md`、`qa/`：页面渲染与核验记录。
- `hashes.json`：交付文件哈希。

## 重建

在仓库根目录运行：

```bash
PYTHONPATH=/tmp/cloud-pdf-deps python 中文翻译/scripts/build_cloud.py 中文翻译/P065_AIRMVC/build.json
```

共享构建脚本未修改；构建采用 pandoc、ziamath、WeasyPrint 与 Noto CJK，所有操作在 dot 云环境完成，无桌面访问、外部翻译服务或 GitHub 写入。

## 忠实性说明

主文原称六个数据集但列出含 SUNRGB-D 的七个名称；相应译注保留此不一致。第3.2节将混合预测引用为式（6），而其实际位于式（7）；第4节写“定理14”而正文定理编号为4.1；均保留原意并提示。表格数值、原图和编号不作静默更改。参考文献64条完整保留检索信息，题名附中文翻译。

## 最终核验结果

最终译文32页，40个原版图/表/公式裁剪。32页均已视觉核验，所有可提取字符裁剪边距不低于2pt，无裁切字符、未展开资产标记或页面越界。原仓库源文件哈希再次确认不变。最终PDF哈希与全部交付文件哈希见 `hashes.json`。
