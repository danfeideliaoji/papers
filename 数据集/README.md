# 数据集

九个数据集各保留一份与统一五折对应的完整数据文件。Pascal07的目录名为VOC2007，IAPRTC12的目录名为IAPRTC-12。

| 数据集 | 唯一数据文件 | 样本数 | 视图数 | 标签数 |
|---|---|---:|---:|---:|
| Corel5k | [corel5k_six_view.mat](<Corel5k/corel5k_six_view.mat>) | 4999 | 6 | 260 |
| Pascal07 | [pascal07_six_view.mat](<VOC2007/pascal07_six_view.mat>) | 9963 | 6 | 20 |
| MIRFlickr | [mirflickr_six_view.mat](<MIRFlickr/mirflickr_six_view.mat>) | 25000 | 6 | 38 |
| ESPGame | [espgame_six_view.mat](<ESPGame/espgame_six_view.mat>) | 20770 | 6 | 268 |
| IAPRTC12 | [iaprtc12.mat](<IAPRTC-12/iaprtc12.mat>) | 4999 | 5 | 291 |
| Scene | [scene.mat](<Scene/scene.mat>) | 2407 | 2 | 6 |
| Emotions | [emotions.mat](<Emotions/emotions.mat>) | 593 | 2 | 6 |
| Yeast | [yeast.mat](<Yeast/yeast.mat>) | 2417 | 2 | 14 |
| 3Sources | [3sources.mat](<3Sources/3sources.mat>) | 169 | 3 | 6 |

MIRFlickr与ESPGame使用Git LFS保存；克隆后运行git lfs pull取得完整数据文件。IAPRTC12为4999样本五视图本地子集。

## 统一实验划分

[九个数据集的统一五折索引与缺失掩码](统一五折与缺失掩码_9数据集/README.md)：45套固定五折索引、180个视图掩码和225个标签掩码，基础种子20261026。需要验证集时约64/16/20，不需要验证集时约80/20，并共用测试折。索引对应上述数据文件的原始样本行号；训练未知标签不能当作负例。
