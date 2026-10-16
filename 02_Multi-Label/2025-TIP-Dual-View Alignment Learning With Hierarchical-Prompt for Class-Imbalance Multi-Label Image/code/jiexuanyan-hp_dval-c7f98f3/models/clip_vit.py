from collections import OrderedDict

import torch
import torch.nn as nn

import clip

class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.token_embedding = clip_model.token_embedding
        self.positional_embedding = clip_model.positional_embedding
        self.transformer = clip_model.transformer
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, text):
        x = self.token_embedding(text).to(self.dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).to(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        return x

class CLIPVIT(nn.Module):

    def get_tokenized_prompts(self, classnames):
        template = "a photo of a {}."
        prompts = [template.format(c.replace("_", " ")) for c in classnames]
        # print(f"Prompts: {prompts}")
        prompts = torch.cat([clip.tokenize(p) for p in prompts])
        prompts = prompts.to(device = torch.device('cuda' if torch.cuda.is_available() else "cpu"))
        return prompts    

    def __init__(self, args, classnames, clip_model, embed_dim=768):
        super().__init__()

        self.final_dim = 512
        self.global_only = False
        
        self.conv1 = clip_model.visual.conv1
        self.class_embedding = clip_model.visual.class_embedding
        self.positional_embedding = clip_model.visual.positional_embedding
        self.ln_pre = clip_model.visual.ln_pre
        self.transformer = clip_model.visual.transformer
        self.ln_post = clip_model.visual.ln_post
        self.clipzero = False

        self.use_clip_proj = False

        self.prompts = self.get_tokenized_prompts(classnames)
        self.text_encoder = TextEncoder(clip_model)

        if not self.use_clip_proj:
            self.projection = nn.Sequential(OrderedDict([
                    ('fc1', nn.Linear(embed_dim, self.final_dim)),
                    ('act', nn.Tanh()),
                    ('fc2', nn.Linear(self.final_dim, self.final_dim))],)
            )

        self.projection_dist = clip_model.visual.proj
        self.topk = args['topk']
        #self.ratio = args['alpha']
    
    def forward_features(self, x):
        
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x)

        return x

    def forward(self, x, norm_pred=True):

        x = self.forward_features(x)
        dist_feat = x[:, 0] @ self.projection_dist

        prompts = self.prompts
        text_features = self.text_encoder(prompts)
        # print(text_features.shape)
        text_features = text_features.float()

        # For Global Head Only Ablation
        if self.global_only:
            score = dist_feat @ text_features.t()
            if norm_pred:
                score = score / score.norm(dim=-1, keepdim=True)
            return score, x[:, 1:], dist_feat

        # Default
        else:
            if not self.use_clip_proj:
            # pred_feat = self.projection(x[:, 1:])
                pred_feat = x[:, 1:] @ self.projection_dist
            else:
                pred_feat = x[:, 1:] @ self.projection_dist
            score1 = torch.topk(pred_feat @ text_features.t(),k=self.topk, dim=1)[0].mean(dim=1)
            score2 = dist_feat @ text_features.t()
            if norm_pred:
                score1 = score1 / score1.norm(dim=-1, keepdim=True)
                score2 = score2 / score2.norm(dim=-1, keepdim=True)
            
            score = (score1 + score2) / 2 
            #score = self.ratio * score1 + (1-self.ratio) * score2

            return score,pred_feat, dist_feat

    def encode_img(self, x):
        # import pdb; pdb.set_trace()
        x = self.forward_features(x)
        if self.clipzero:
            x = x @ self.proj
            return x[:, 1:, :], x[:, 0, :]
        else:
            pred_feat = x[:, 1:] @ self.projection_dist
            # dist_feat = self.projection_dist(x[:, 0])
            dist_feat = x[:, 0] @ self.projection_dist
            return pred_feat, dist_feat