# P093 中文交付文件

- `translation.zh.pdf`：26页中文译本。
- `translation.zh.md`：全文中文、可编辑LaTeX公式和中文重排表格。
- `assets/`：作者图表和已移除未清权照片的图示；图4/5还提供独立子图。
- `assets-authorized/`：仅包含已清除照片的图1(a)、图2安全派生PDF，供可重复构建。
- `manifest.json`：来源、许可、图像例外、文件清单与散列。
- `coverage.md`：逐页覆盖与图像保留/省略说明。
- `QA.md`、`audit-results.json`：视觉和程序核验结果。
- `SHA256SUMS`：交付文件SHA-256清单。
- `build.json`、`build_p093.py`：使用上一级共享`../scripts/build_cloud.py`构建器的P093配置及局部版式覆盖；共享脚本未改动。
- `audit.py`：针对原PDF表格的402个数字、公式和文献编号、重音提取校核程序。

原文PDF不在安全交付包中。需要重新构建时，从 https://arxiv.org/pdf/2509.17747v1 获取相同版本，并核对 SHA-256：`abb753930136609a403aca74f834de437973ccc3972105f854bf966e98fe0c14`。重建必须继续遵守照片/热图/肖像省略要求；不得用原图直接替换已清除照片的安全素材而声称已获许可。

当前云目录中已有的`source/`仅为原始核验输入，保持不变，不能递归加入新的用户分发包；`source-verification.json`为先前来源核验记录，不在安全包中。安全包采用白名单打包。无GitHub或其他公开上传。

许可：CC BY-NC-SA 4.0，非商业、署名、相同方式共享；完整说明见译本首页。第三方省略材料的许可未被授予。原作作者：Sheng Huang、Jiexuan Yan、Beiyan Liu、Bo Liu、Richang Hong；AI辅助中文翻译与排版，未经原作者审定。
