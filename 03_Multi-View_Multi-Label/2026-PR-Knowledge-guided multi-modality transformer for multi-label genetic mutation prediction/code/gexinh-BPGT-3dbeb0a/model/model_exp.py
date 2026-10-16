"""Paper-aligned BPGT model wrapper.

This module keeps the historical ``model_markets`` entry point used by the
training scripts, while exposing clearer names that match the paper:
visual extractor (VE), gene encoder (GE), gene graph / KAM-driven priors,
and label decoder (LD).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.BPGT import BPGT, BPGTLayer
from model.MCAT import SNN
try:
    from model.KAT import KAT
except ImportError:
    KAT = None
from model.base_model import (
    AttnClassifier,
    Attn_Net_Gated,
    CLAM_SB,
    GCN,
    Graph_Tran,
    MCAT_Decoder,
    MLDecoder,
    SA_MIL,
    TransMIL,
    size_selection,
    transformer_FC,
)
from utils.utils import initialize_weights


def _require_kat():
    if KAT is None:
        raise ImportError("KAT requires the model.KAT dependencies, including einops.")


class ABMIL(nn.Module):
    """Attention-based MIL visual extractor."""

    def __init__(self, size_arg="dino", dropout=False, dropout_rate=0.5):
        super().__init__()
        size = size_selection(size_arg)
        layers = [nn.Linear(size[0], size[1]), nn.ReLU()]
        layers.append(
            Attn_Net_Gated(
                L=size[1],
                D=size[2],
                dropout=dropout,
                n_classes=1,
            )
        )
        self.attention_net = nn.Sequential(*layers)
        initialize_weights(self)

    def forward(self, patch_features, label=None, instance_eval=True):
        attention_scores, patch_features = self.attention_net(patch_features)
        attention_scores = torch.transpose(attention_scores, 1, 0)
        attention_scores = F.softmax(attention_scores, dim=1)
        return torch.mm(attention_scores, patch_features)


class BPGTModel(nn.Module):
    """Biological-knowledge-enhanced PathGenomic multi-label Transformer."""

    BPGT_ALIASES = {"bpgt", "geneformer", "geneformer_v2", "graphormer"}

    def __init__(
        self,
        num_classes,
        num_of_groups=-1,
        model_size="1024_small",
        zsl=0,
        adj=None,
        dropout=True,
        dropout_rate=0.5,
        k_sample=8,
        num_layers_decoder=1,
        num_layers_encoder=2,
        gcn="forward",
        graph_connect=None,
        agg_method="att",
        cls_token=True,
        region_cls=True,
        num_region=1,
        instance_loss_fn=None,
        spatial_mask=None,
        pe=None,
        num_graph_layer=2,
        word_embedd=None,
        graph_module="bpgt",
        kat_num_kernels=16,
        **unused_kwargs,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.graph_module = self._normalize_graph_module(graph_module)
        self.graph_connect = graph_connect or gcn
        self.agg_method = agg_method
        self.kat_num_kernels = kat_num_kernels
        self.cls_token = cls_token
        self.mlp = self.graph_module == "mlp"

        size = size_selection(model_size)
        initial_d_features = size[1]
        d_decoder_embed = size[1]

        self.visual_extractor, initial_d_features, d_decoder_embed, region_cls = (
            self._build_visual_extractor(
                agg_method=agg_method,
                model_size=model_size,
                dropout=dropout,
                dropout_rate=dropout_rate,
                num_classes=num_classes,
                num_region=num_region,
                k_sample=k_sample,
                num_layers_encoder=num_layers_encoder,
                kat_num_kernels=kat_num_kernels,
                default_initial_dim=initial_d_features,
                default_decoder_dim=d_decoder_embed,
            )
        )

        self.region_cls = bool(region_cls)
        if self.region_cls and num_region != 1 and agg_method != "trans_mil":
            region_model = {
                "model": "wsl",
                "n_classes": num_region,
                "size_arg": "hipt",
                "k_sample": k_sample,
                "dropout": dropout,
                "dropout_rate": dropout_rate,
            }
            self.region_cls_module = CLAM_SB(**region_model, instance_loss_fn=instance_loss_fn)
            self.region_attention_weight = nn.Parameter(torch.ones(1, dtype=torch.float32))

        d_label_embed = d_decoder_embed
        num_label_embed = 100 if num_of_groups < 0 else num_of_groups
        num_label_embed = min(num_label_embed, num_classes)

        self.if_word_embedd = word_embedd is not None
        d_query_embed = d_label_embed * 3 if self.graph_module == "gcn" else d_label_embed
        self.query_embed, self.word_proj = self._build_gene_queries(
            num_label_embed=num_label_embed,
            d_query_embed=d_query_embed,
            word_embedd=word_embedd,
            zsl=zsl,
        )
        self.zsl = zsl

        self.adj = self._as_float_tensor(adj)

        if self.mlp:
            self.mlp_classifier = nn.Linear(d_decoder_embed, num_classes)
            self.cls_token = False
        else:
            self.gene_encoder = self._build_gene_encoder(
                d_label_embed=d_label_embed,
                d_decoder_embed=d_decoder_embed,
                model_size=model_size,
                dropout=dropout,
                adj=adj,
                spatial_mask=spatial_mask,
                pe=pe,
                num_graph_layer=num_graph_layer,
                cls_token=cls_token,
            )

        if self.cls_token:
            self.cls_head = nn.Linear(d_label_embed, num_classes)

        self.forward_net = [nn.Linear(d_label_embed, d_decoder_embed), nn.ReLU()]
        if dropout:
            self.forward_net.append(nn.Dropout(dropout_rate))
        self.forward_net = nn.Sequential(*self.forward_net)
        self.gamma = nn.Parameter(torch.ones(1))

        if self.graph_module == "mcat":
            self.label_decoder = MCAT_Decoder(
                num_classes=num_classes,
                num_label_embed=num_label_embed,
                d_decoder_embed=d_decoder_embed,
                initial_num_features=initial_d_features,
                zsl=zsl,
                num_layers_decoder=num_layers_decoder,
                cls_token=cls_token,
            )
        elif not self.mlp:
            self.label_decoder = MLDecoder(
                num_classes=num_classes,
                num_label_embed=num_label_embed,
                d_decoder_embed=d_decoder_embed,
                initial_num_features=initial_d_features,
                zsl=zsl,
                num_layers_decoder=num_layers_decoder,
                cls_token=cls_token,
            )

        initialize_weights(self)

    @classmethod
    def _normalize_graph_module(cls, graph_module):
        graph_module = (graph_module or "bpgt").lower()
        if graph_module in cls.BPGT_ALIASES:
            return "bpgt"
        if graph_module in {"graph_transformer", "transformer"}:
            return "graph_transformer"
        return graph_module

    @staticmethod
    def _as_float_tensor(value):
        if isinstance(value, np.ndarray):
            return torch.from_numpy(value.astype(np.float32))
        return value

    def _build_visual_extractor(
        self,
        agg_method,
        model_size,
        dropout,
        dropout_rate,
        num_classes,
        num_region,
        k_sample,
        num_layers_encoder,
        kat_num_kernels,
        default_initial_dim,
        default_decoder_dim,
    ):
        size = size_selection(model_size)

        if agg_method == "att":
            extractor = AttnClassifier(
                num_classes=num_classes,
                aggregation="att",
                size_arg=model_size,
                dropout=dropout,
                dropout_rate=dropout_rate,
                k_sample=-1,
            )
            return extractor, size[2], size[2], False

        if agg_method == "dense":
            extractor = AttnClassifier(
                num_classes=num_classes,
                aggregation="mean",
                size_arg=model_size,
                dropout=dropout,
                dropout_rate=dropout_rate,
                k_sample=-1,
            )
            return extractor, size[2], size[2], False

        if agg_method == "self_att":
            extractor = AttnClassifier(
                num_classes=num_classes,
                aggregation="att",
                size_arg=model_size,
                dropout=dropout,
                dropout_rate=dropout_rate,
                k_sample=-1,
            )
            return extractor, size[2], size[2], False

        if agg_method == "abmil":
            extractor = ABMIL(size_arg=model_size, dropout=dropout, dropout_rate=dropout_rate)
            return extractor, size[1], size[1], False

        if agg_method == "samil":
            return SA_MIL(model_size), default_initial_dim, default_decoder_dim, False

        if agg_method == "hipt":
            extractor = transformer_FC(
                num_layer=num_layers_encoder,
                size_arg=model_size,
                dropout_rate=dropout_rate,
            )
            return extractor, default_initial_dim, default_decoder_dim, False

        if agg_method == "trans_mil":
            return TransMIL(model_size, n_classes=num_region), default_initial_dim, default_decoder_dim, True

        if agg_method == "kat":
            _require_kat()
            extractor = KAT(
                num_pk=kat_num_kernels,
                patch_dim=size[0],
                num_classes=num_classes,
                dim=size[1],
                depth=max(1, num_layers_encoder),
                heads=8,
                mlp_dim=size[1] * 2,
                num_kernal=kat_num_kernels,
                dropout=dropout_rate if dropout else 0.0,
            )
            return extractor, size[1], size[1], False

        raise ValueError(f"Unsupported visual extractor agg_method: {agg_method}")

    @staticmethod
    def _build_gene_queries(num_label_embed, d_query_embed, word_embedd, zsl):
        if zsl:
            return None, None
        if word_embedd is None:
            query_embed = nn.Embedding(num_label_embed, d_query_embed)
            query_embed.requires_grad_(False)
            return query_embed, None
        word_proj = nn.Linear(word_embedd.size(-1), d_query_embed, bias=True)
        word_proj.requires_grad_(True)
        return word_embedd, word_proj

    def _build_gene_encoder(
        self,
        d_label_embed,
        d_decoder_embed,
        model_size,
        dropout,
        adj,
        spatial_mask,
        pe,
        num_graph_layer,
        cls_token,
    ):
        if self.graph_module == "bpgt":
            kam_layer = BPGTLayer(
                d_label_embed,
                nhead=8,
                dim_feedforward=d_label_embed,
                dropout=0.1,
                activation="relu",
                layer_norm_eps=1e-5,
            )
            return BPGT(
                kam_layer,
                d_label_embed,
                num_layer=num_graph_layer,
                adj=adj,
                spatial_mask=spatial_mask,
                pe=pe,
                if_cls_token=cls_token,
            )

        if self.graph_module == "gcn":
            return GCN(nclass=d_decoder_embed, model_size=model_size, dropout=0.1)

        if self.graph_module == "graph_transformer":
            return Graph_Tran(
                in_dim=d_label_embed,
                dropout_rate=0.1,
                num_layer=2,
                adj=adj,
                if_cls_token=cls_token,
            )

        if self.graph_module == "mcat":
            return SNN(input_dim=768, model_size_omic="small")

        raise ValueError(f"Unsupported graph_module: {self.graph_module}")

    def _sample_patches(self, h, max_patches=500):
        total_patch = h.size(0)
        if total_patch <= max_patches:
            return h
        row_sequence = np.arange(total_patch)
        np.random.shuffle(row_sequence)
        return h[row_sequence[:max_patches], :]

    def _build_kat_inputs(self, h, h_coor=None):
        if h.dim() == 2:
            node_features = h.unsqueeze(0)
        else:
            node_features = h

        batch_size, num_patches, _ = node_features.shape
        device = node_features.device

        if h_coor is None:
            coords = torch.arange(num_patches, device=device, dtype=node_features.dtype).view(1, num_patches, 1)
        else:
            coords = h_coor.to(device=device, dtype=node_features.dtype)
            if coords.dim() == 2:
                coords = coords.unsqueeze(0)
            if coords.dim() > 3:
                coords = coords.squeeze(0)
            if coords.size(0) != batch_size:
                coords = coords.expand(batch_size, -1, -1)
            if coords.size(1) != num_patches:
                coords = torch.arange(num_patches, device=device, dtype=node_features.dtype).view(1, num_patches, 1)

        coords = coords[..., :2] if coords.size(-1) > 1 else coords
        coord_min = coords.min(dim=1, keepdim=True)[0]
        coord_range = (coords.max(dim=1, keepdim=True)[0] - coord_min).clamp_min(1.0)
        coords = (coords - coord_min) / coord_range

        active_kernels = min(self.kat_num_kernels, num_patches)
        kernel_index = torch.linspace(0, num_patches - 1, active_kernels, device=device).long()
        kernel_coords = coords[:, kernel_index, :]
        if active_kernels < self.kat_num_kernels:
            pad = kernel_coords[:, -1:, :].expand(batch_size, self.kat_num_kernels - active_kernels, -1)
            kernel_coords = torch.cat([kernel_coords, pad], dim=1)

        relative_distance = torch.cdist(kernel_coords, coords)
        distance_scale = relative_distance.reshape(batch_size, -1).max(dim=1)[0].view(batch_size, 1, 1)
        relative_distance = relative_distance / distance_scale.clamp_min(1.0)

        patch_mask = torch.ones(batch_size, num_patches, 1, device=device, dtype=node_features.dtype)
        kernel_mask = torch.zeros(batch_size, self.kat_num_kernels, 1, device=device, dtype=node_features.dtype)
        kernel_mask[:, :active_kernels, :] = 1
        return node_features, relative_distance, patch_mask, kernel_mask

    def _encode_visual_features(self, h, h_coor=None, label=None, instance_eval=True):
        y_region_logits = []
        y_region_prob = []
        results_dict = {}

        if self.agg_method == "dense":
            h = self._sample_patches(h)
            _, visual_features, _ = self.visual_extractor(h.unsqueeze(0))

        elif self.agg_method == "self_att":
            h = self._sample_patches(h)
            _, visual_features, _ = self.visual_extractor(h.unsqueeze(0))

        elif self.agg_method == "att":
            _, _, visual_features = self.visual_extractor(h.unsqueeze(0))

        elif self.agg_method == "abmil":
            visual_features = self.visual_extractor(h)

        elif self.agg_method == "samil":
            result = self.visual_extractor(h)
            visual_features = result["h_feat"]

        elif self.agg_method == "hipt":
            visual_features = self.visual_extractor(h.unsqueeze(0))

        elif self.agg_method == "trans_mil":
            result = self.visual_extractor(h)
            y_region_logits = result["logits"]
            y_region_prob = result["Y_prob"]
            visual_features = result["h_feat"]

        elif self.agg_method == "kat":
            kat_inputs = self._build_kat_inputs(h, h_coor=h_coor)
            _, visual_features = self.visual_extractor(*kat_inputs)

        else:
            raise ValueError(f"Unsupported visual extractor agg_method: {self.agg_method}")

        if (
            self.region_cls
            and hasattr(self, "region_cls_module")
            and self.agg_method != "trans_mil"
        ):
            y_region_logits, y_region_prob, _, attention, results_dict = self.region_cls_module(
                visual_features.squeeze(),
                label,
                instance_eval,
            )
            visual_features = (
                visual_features.squeeze()
                + self.region_attention_weight * attention.transpose(0, 1) * visual_features.squeeze()
            )

        return visual_features, y_region_logits, y_region_prob, results_dict

    def _gene_query_features(self):
        if self.zsl:
            return F.relu(self.wordvec_proj(self.query_embed))
        if self.if_word_embedd:
            if self.graph_module == "gcn":
                return self.query_embed
            return self.word_proj(self.query_embed)
        return self.query_embed.weight

    def _encode_gene_priors(self, gene_queries):
        if self.graph_connect == "forward":
            if self.graph_module == "gcn":
                return self.gene_encoder(gene_queries, adj=self.adj)
            return self.gene_encoder(gene_queries)

        if self.graph_connect == "bypass":
            forward_features = self.forward_net(gene_queries)
            if self.graph_module == "gcn":
                graph_features = self.gene_encoder(gene_queries, self.adj)
            else:
                graph_features = self.gene_encoder(gene_queries)
            return forward_features + self.gamma * graph_features

        if self.graph_connect == "abandon":
            return self.forward_net(gene_queries)

        raise ValueError(f"Unsupported graph_connect: {self.graph_connect}")

    def compute_l1_loss(self, weights):
        return torch.abs(weights).sum()

    def forward(self, h, h_coor=None, label=None, instance_eval=True):
        visual_features, y_region_logits, y_region_prob, results_dict = self._encode_visual_features(
            h,
            h_coor=h_coor,
            label=label,
            instance_eval=instance_eval,
        )

        attention_scores = []
        if self.mlp:
            if len(visual_features.size()) == 2:
                visual_features = visual_features.mean(dim=0)
            logits = self.mlp_classifier(visual_features)
        else:
            gene_queries = self._gene_query_features()
            gene_priors = self._encode_gene_priors(gene_queries)
            logits, virtual_node, attention_scores = self.label_decoder(visual_features, gene_priors)
            if self.cls_token:
                logits = self.cls_head(virtual_node)

        probs = torch.sigmoid(logits)
        results_dict["a_score"] = attention_scores
        results_dict["lasso_loss"] = self.compute_l1_loss(probs)
        results_dict["Y_region_logits"] = y_region_logits
        results_dict["Y_region_prob"] = y_region_prob
        return logits, probs, results_dict

    def relocate(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.visual_extractor = self.visual_extractor.to(device)
        if self.mlp:
            self.mlp_classifier = self.mlp_classifier.to(device)
        else:
            self.gene_encoder = self.gene_encoder.to(device)
            self.label_decoder = self.label_decoder.to(device)
            if self.cls_token:
                self.cls_head = self.cls_head.to(device)
        if self.query_embed is not None:
            self.query_embed = self.query_embed.to(device)
        if self.word_proj is not None:
            self.word_proj = self.word_proj.to(device)
        if self.adj is not None:
            self.adj = self.adj.to(device)


# Historical entry point used by utils.core_utils.
model_markets = BPGTModel
