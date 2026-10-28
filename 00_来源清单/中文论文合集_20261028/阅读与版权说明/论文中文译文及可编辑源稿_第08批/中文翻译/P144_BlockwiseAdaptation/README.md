# P144｜标签偏移与块状模态缺失下的多模态域适应

已完成 **53 页非官方中文全文 PDF**，覆盖原版 49 页正文、补充材料、全部证明和 60 条参考文献。已通过独立逐段、逐页及全部裁图复核；一处两作者引用已修正。

- `P144_BlockwiseAdaptation_中文全文译文.pdf`：阅读版。
- `translation.zh.md`：完整可编辑译文；`main.zh.md`、`supplement.zh.md`、`references.zh.md` 为分段源文件。
- `assets/`：133 个原版裁图（3 图、17 表块、113 公式块）。
- `manifest.json`、`build.json`：版本、覆盖范围、裁图坐标和构建配置。
- `rights_evidence.json`：精确版本/许可/三图排除项检查记录。
- `qa/QA.md`、`qa/page_review.json`、`qa/validation.json`：逐页目视与机器检查记录；`qa/pages/` 含全部 53 页渲染图。
- `qa/independent_review.md`、`qa/independent_review.json`：独立全文复核与最小修正记录。
- `SHA256SUMS.txt`：文件校验和。

原作：Zebin Wang、Ziang Dou、Molei Liu、Tianxi Cai。官方 [arXiv:2608.01282v1](https://arxiv.org/abs/2608.01282v1)，2026-08-02，依据 [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) 使用。官方 PDF 与仓库原件逐字节一致；原件未修改。翻译与重排：OpenAI ChatGPT，2026-10-26。非作者认可的官方版本；译者改动和署名详见 PDF 首页。

## 重建

在本目录执行：

```sh
PYTHONPATH=/tmp/cloud-pdf-deps python /workspace/shared/papers-cloud/中文翻译/scripts/build_cloud.py build.json
python render_qa.py
python qa/validate_final.py
```

重建会覆盖本目录的译文 PDF 和资产，不修改原作。任何修改后应重新进行逐页目视复核并更新 manifest 与校验和。
