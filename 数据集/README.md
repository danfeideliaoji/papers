# 数据集

保留九个实验数据集：Corel5k、Pascal07、MIRFlickr、ESPGame、IAPRTC-12、Scene、Emotions、Yeast和3Sources。Pascal07在仓库中的目录名为VOC2007。

同名数据可能有多个版本；固定索引对应的数据身份与样本顺序，以统一划分包内的metadata.json和sources.json为准，不得将索引套用于其他版本。

| 数据集 | 仓库目录 | 文件数 |
|---|---|---:|
| Corel5k | [Corel5k](<Corel5k/>) | 16 |
| Pascal07 | [VOC2007](<VOC2007/>) | 1 |
| MIRFlickr | [MIRFlickr](<MIRFlickr/>) | 1 |
| ESPGame | [ESPGame](<ESPGame/>) | 1 |
| IAPRTC-12 | [IAPRTC-12](<IAPRTC-12/>) | 1 |
| Scene | [Scene](<Scene/>) | 8 |
| Emotions | [Emotions](<Emotions/>) | 5 |
| Yeast | [Yeast](<Yeast/>) | 1 |
| 3Sources | [3Sources](<3Sources/>) | 1 |

## 统一实验划分

[九个数据集的统一五折索引与缺失掩码](统一五折与缺失掩码_9数据集/README.md)：45套固定五折索引、180个视图掩码和225个标签掩码，基础种子20261026。需要验证集时约64/16/20，不需要验证集时约80/20，并共用测试折。训练未知标签不能当作负例。
