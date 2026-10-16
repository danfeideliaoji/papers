import torch.nn as nn
from lib.ML_decoder.backbone.cnn import create_cnn_backbone, create_featuremap_backbone, _MODELS_INFO
from lib.ML_decoder.backbone.classwise import ClasswiseEncoder, GroupLinear

class ClasswiseModel(nn.Module):
    def __init__(self, backbone_name, num_classes, dim_embed=512):
        super().__init__()
        self.dim_feature = _MODELS_INFO[backbone_name].dim_featuremap
        self.backbone = create_featuremap_backbone(backbone_name, pretrained=True)
        self.encoder = ClasswiseEncoder(num_classes, self.dim_feature, dim_embed)
        self.head = nn.Sequential(GroupLinear(num_classes, dim_embed, 1), nn.Flatten(1, 2))

    def forward(self, x, return_feature=False):
        x = self.backbone(x) # [batch_size, H // 32, W // 32, dim_feature]
        feature = self.encoder(x)  # [batch_size, num_classes, dim_embed]
        x = self.head(feature)
        if return_feature:
            return x, feature
        return x
