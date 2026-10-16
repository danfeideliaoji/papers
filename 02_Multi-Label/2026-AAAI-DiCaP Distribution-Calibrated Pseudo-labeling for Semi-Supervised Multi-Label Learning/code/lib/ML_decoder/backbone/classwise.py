from typing import Optional
import math
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn.modules.transformer import _get_activation_fn


class TransformerDecoderLayerWithoutSelfAttn(nn.Module):
    def __init__(
        self, d_model, nhead=8, dim_feedforward=2048, dropout=0.1, activation="relu", layer_norm_eps=1e-5, batch_first=True
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm3 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first)
        self.self_attn = None
        self.activation = _get_activation_fn(activation)

    def __setstate__(self, state):
        if "activation" not in state:
            state["activation"] = torch.nn.functional.relu
        super(TransformerDecoderLayerWithoutSelfAttn, self).__setstate__(state)

    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        tgt = tgt + self.dropout1(tgt)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(tgt, memory, memory)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt


class GroupLinear(nn.Module):
    def __init__(self, num_classes: int, in_features: int, out_features: int, *, bias: bool = True):
        super().__init__()
        self.num_classes = num_classes
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(num_classes, in_features, out_features))
        torch.nn.init.xavier_normal_(self.weight)
        if bias:
            self.bias = nn.Parameter(torch.Tensor(num_classes, out_features))
            torch.nn.init.constant_(self.bias, 0)
        else:
            self.bias = None

    def forward(self, x):
        batch_size, num_classes, in_features = x.shape
        y = torch.einsum("bci,cij->bcj", x, self.weight)

        if self.bias is not None:
            y += self.bias[None, :, :]

        return y


class ClasswiseEncoder(nn.Module):

    def __init__(self, num_classes, dim_feature, dim_embed):
        super().__init__()
        layer_decode = TransformerDecoderLayerWithoutSelfAttn(
            d_model=dim_embed, dim_feedforward=dim_embed * 4, dropout=0.1, nhead=8, batch_first=True
        )
        self.querys = nn.Embedding(num_embeddings=num_classes, embedding_dim=dim_embed)
        self.feature_projector = nn.Linear(dim_feature, dim_embed)
        self.query_projector = nn.Linear(dim_embed, dim_embed)
        self.decoder = nn.TransformerDecoder(layer_decode, num_layers=1)

        self.querys.requires_grad_(False)

    def forward(self, x):
        #  x: [batch_size, dim_feature, h, w] （featuremap)
        batch_size, _, h, w = x.shape

        x = x.flatten(2).transpose(1, 2)  # [batch_size, h*w, dim_feature]
        x = self.feature_projector(x)  # [batch_size, h*w, dim_embed]

        y = self.querys.weight  # [num_classes, dim_embed]
        y = self.query_projector(y)  # [num_classes, dim_embed]
        y = y.unsqueeze(0).expand(batch_size, -1, -1)  # [batch_size, num_classes, dim_embed]

        representations = self.decoder(y, x)  # [batch_size, num_classes, dim_embed]
        return representations
