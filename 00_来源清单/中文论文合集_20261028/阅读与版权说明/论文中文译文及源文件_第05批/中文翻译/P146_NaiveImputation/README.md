# P146 中文全文译文

- 中文题名：朴素插补对高维线性模型产生隐式正则化。
- 原作者：Alexis Ayme、Claire Boyer、Aymeric Dieuleveut、Erwan Scornet。
- 原始发表：Proceedings of the 40th International Conference on Machine Learning, PMLR 202:1320–1340, 2023。
- 官方入口：https://proceedings.mlr.press/v202/ayme23a.html
- 官方 PDF：https://proceedings.mlr.press/v202/ayme23a/ayme23a.pdf
- 许可：Creative Commons Attribution 4.0 International，https://creativecommons.org/licenses/by/4.0/ 。PMLR 出版协议第2条向公众授予 CC BY 4.0，第3条要求注明原始发表及官方论文链接：https://proceedings.mlr.press/pmlr-license-agreement.html 。
- 原文版权 © 2023 原作者。此中文翻译、题名翻译、图表说明翻译和重新排版是非官方改编，并非作者或出版方认可的版本。依 CC BY 4.0 提供本译文；再使用时保留上述原作署名、来源、许可和改编声明。

## 文件

- `translation.zh.md`：完整中文译文；全部正文、致谢、27条参考文献、附录A–F和全部证明。短公式采用 LaTeX；长附录推导通过原文裁图保真保留，Markdown 中 `{{asset}}` 对应 manifest。
- `P146_NaiveImputation_中文全文译文.pdf`：28页排版译文。
- `build.json`：源文件 SHA-256、源页码、全部53个资产的精确裁剪框及排版宽度。
- `assets/`：51个数学推导裁图、2个图形裁图，216 dpi。表1全文重排为中文表格。
- `QA.md`、`qa/`：逐页检查、逐资产几何检查与渲染图。
- `rights_evidence.json`、`SHA256SUMS`：许可核查记录及交付文件哈希。

## 再构建

在仓库根目录使用现有公共构建器（未修改）：

```bash
PYTHONPATH=/tmp/cloud-pdf-deps python 中文翻译/scripts/build_cloud.py 中文翻译/P146_NaiveImputation/build.json
```

`make_manifest.py` 仅生成本目录 manifest；`check_crops.py` 只读源 PDF 并检查裁图中每个非空白字符边界。该脚本的官方 PDF 临时路径可在其他机器上改为 manifest 所指的原始论文。

## 保真原则与原文疑点

保留原文公式编号(1)–(29)，未使用可能被渲染器丢弃的 `\tag`。原文的大小写、标量、向量和花体记号、索引、正负号、求和上下限以及逐行推导均依原文保留。不以数学修订替代翻译。原文自身存在疑似排印或推导问题，例如第4页最优参数前的 `1/(L²ρ)`，附录B.1中 `HᵀΓH` 与证明中的 `HΓHᵀ`，附录C中的部分常数/风险等式，附录D.1的等号、D.2推导部分范数因子、D.3部分缺失的 λ，以及E.3证明中 k/r 交替与求和上界。正文与裁图均不静默修正这些内容。读者如需在研究中使用结论，应同时查阅官方原文。

所有制作、渲染与检查均在 dot CLOUD 本地进行；没有使用用户桌面、外部翻译服务、API密钥或 GitHub 写入。
