
# CMLL: Correlated Multi-Label Loss

The code accompanying **Towards Reliable Multi-label Classification via Conditional Dependency Modeling**, ECCV 2026.

## Abstract

Multi-label classification models are commonly trained by assuming conditional independence among labels, typically through the use of binary cross-entropy objectives. While this assumption simplifies optimization, it introduces a structural bias that ignores dependencies among labels and can adversely affect the calibration of predicted probabilities. In this work, we theoretically analyze the impact of this assumption and show that neglecting label dependencies leads to a miscalibration effect that is linked to the conditional dependency of the labels. Motivated by this observation, we propose **Pairwise Correlation Difference (PCD)**, a novel auxiliary loss designed to incorporate label dependency information during training. PCD aligns the pairwise correlations of model logits with those of the ground-truth labels, thereby encouraging the network to capture conditional dependencies among labels. We combine PCD with the standard binary cross-entropy loss to form the **Correlated Multi-Label Loss (CMLL)**, which serves as a objective for dependency-aware training. We provide theoretical justification for the functional form of the proposed loss and its connection to the dependency structure of the label distribution. Extensive experiments on three benchmark multi-label datasets demonstrate that CMLL consistently produces better-calibrated predictions while maintaining competitive classification accuracy compared to existing multi-label learning objectives. Additionally, we show that the proposed approach exhibits robustness to the choice of binning schemes used in calibration evaluation

## Train CMLL

Configuration is edited directly in `main.py` and the selected loader. 
### 1. Select the dataset

Keep exactly one `build` import active at the top of `main.py`, and set the matching `config['num_classes']` value:

| Dataset | Active import | `num_classes` |
| --- | --- | ---: |
| PASCAL VOC 2012 | `from dataloader_pascalvoc import build` | 20 |
| MS-COCO 2014 | `from dataloader_coco import build` | 80 |
| WIDER Attribute | `from dataloader_wider import build` | 14 |

For example, the supplied VOC configuration uses:

```python
from dataloader_pascalvoc import build

# In the existing config dictionary:
# 'num_classes': 20,
```

### 2. Choose the backbone

**ViT-B/32** is active by default. Its classifier is replaced with a linear layer producing `num_classes` logits. The current constructor loads no pretrained weights.

For **ResNet-50**, replace the active ViT construction block with:

```python
model = models.resnet50()
model.fc = nn.Linear(model.fc.in_features, config['num_classes'])
```

Retain the subsequent device selection and `model.to(device)` statements. All three loaders return RGB images, so retain the model's original three-channel input layer.

Select the corresponding optimizer in the existing optimizer block:

```python
# ResNet-50
optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)

# OR ViT-B/32 (the active setting in the supplied script)
optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
```

Keep only the optimizer assignment for the chosen backbone active.

## Citation

If this work is useful for your research, please cite:

```bibtex
@inproceedings{panda2026towards,
  title     = {Towards Reliable Multi-label Classification via Conditional Dependency Modeling},
  author    = {Panda, Arkapal and Pal, Aditya Shankar and Garain, Utpal},
  booktitle = {Computer Vision -- ECCV 2026},
  year      = {2026}
}
```
