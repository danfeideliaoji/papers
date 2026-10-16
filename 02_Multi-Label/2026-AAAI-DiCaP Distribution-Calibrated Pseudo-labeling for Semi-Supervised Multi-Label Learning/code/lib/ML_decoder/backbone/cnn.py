import torchvision
import timm
from torch import nn
from typing import List
from dataclasses import dataclass

_MODELS = {
    "resnet18": torchvision.models.resnet.resnet18,
    "resnet34": torchvision.models.resnet.resnet34,
    "resnet50": torchvision.models.resnet.resnet50,
    "resnet101": torchvision.models.resnet.resnet101,
    "resnet152": torchvision.models.resnet.resnet152,
    "tresnet_m": timm.models.tresnet.tresnet_m,
    "tresnet_l": timm.models.tresnet.tresnet_l,
    "tresnet_xl": timm.models.tresnet.tresnet_xl,
    "tresnet_v2_l": timm.models.tresnet.tresnet_v2_l,
}


@dataclass
class ModelInfo:
    layer_featuremap: str
    layer_fc: str
    dim_featuremap: int
    dim_fc: int
    dim_out: int
    downsample_ratio: int


_MODELS_INFO = {
    "resnet18": ModelInfo("layer4", "fc", 512, 512, 1000, 32),
    "resnet34": ModelInfo("layer4", "fc", 512, 512, 1000, 32),
    "resnet50": ModelInfo("layer4", "fc", 2048, 2048, 1000, 32),
    "resnet101": ModelInfo("layer4", "fc", 2048, 2048, 1000, 32),
    "resnet152": ModelInfo("layer4", "fc", 2048, 2048, 1000, 32),
    "tresnet_m": ModelInfo("body", "head.fc", 2048, 2048, 1000, 32),
    "tresnet_l": ModelInfo("body", "head.fc", 2432, 2432, 1000, 32),
    "tresnet_xl": ModelInfo("body", "head.fc", 2656, 2656, 1000, 32),
    "tresnet_v2_l": ModelInfo("body", "head.fc", 2048, 2048, 1000, 32),
}


def available_models() -> List[str]:
    """Returns the names of available CNN models"""
    return list(_MODELS.keys())


def create_cnn_backbone(name, *, num_classes=None, pretrained=False):
    assert name in _MODELS.keys()
    if "tresnet" in name:
        model = _MODELS[name](pretrained=pretrained)
        """info = {
            "layer_featuremap": "layer4",
            "layer_fc": "fc",
            "dim_featuremap": model.layer4[-1].conv1.in_channels,
            "dim_fc": model.fc.in_features,
            "dim_out": model.fc.out_features,
            "downsample_ratio": 32,
        }"""

    elif "resnet" in name:
        model = _MODELS[name](weights="DEFAULT" if pretrained else None)


    if num_classes is not None:
        info = _MODELS_INFO[name]
        setattr(model, info.layer_fc, nn.Linear(info.dim_fc, num_classes, bias=True))

    return model


class IntermediateLayerExtracter(nn.ModuleDict):
    def __init__(self, model: nn.Module, return_layer: str) -> None:
        if return_layer not in [name for name, _ in model.named_children()]:
            raise ValueError("return_layer are not present in model")
        layers = dict()
        for name, module in model.named_children():
            layers[name] = module
            if name == return_layer:
                break

        super().__init__(layers)
        self.return_layer = return_layer

    def forward(self, x):
        for name, module in self.items():
            x = module(x)
        return x


def create_featuremap_backbone(name, *, pretrained=False):
    model = create_cnn_backbone(name, pretrained=pretrained)
    info = _MODELS_INFO[name]
    model = IntermediateLayerExtracter(model, info.layer_featuremap)
    return model


def _model_info(name, model):
    if "tresnet" in name:
        info = {
            "layer_featuremap": "layer4",
            "layer_fc": "fc",
            "dim_featuremap": model.layer4[-1].conv1.in_channels,
            "dim_fc": model.fc.in_features,
            "dim_out": model.fc.out_features,
            "downsample_ratio": 32,
        }

    elif "resnet" in name:
        info = {
            "layer_featuremap": "body",
            "layer_fc": "head.fc",
            "dim_featuremap": model.body[-1][-1].conv1.in_channels,
            "dim_fc": model.head.fc.in_features,
            "dim_out": model.head.fc.out_features,
            "downsample_ratio": 32,
        }

    return info
