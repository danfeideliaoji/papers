import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from torch.nn import Parameter
import pickle
import transformers 
import torchvision.models as models
from layers import *
from einops import rearrange

class TextExtract_Bert(nn.Module):
    def __init__(self):
        super(TextExtract_Bert, self).__init__()
        

        bert_path = '/model/zfr888/dual/bert-base-uncased'##****
        model_class, tokenizer_class, pretrained_weights = (transformers.BertModel, transformers.BertTokenizer, bert_path)
        
        self.tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
        self.text_embed = model_class.from_pretrained(pretrained_weights)
        self.text_embed.eval()
        for p in self.text_embed.parameters():
            p.requires_grad = False
        
        
    
    
    
    def forward(self, input_ids, attention_mask, token_type_ids):
        with torch.no_grad():
            out = self.text_embed(input_ids=input_ids,
                       attention_mask=attention_mask,
                       token_type_ids=token_type_ids)

        out = out.last_hidden_state

        # print("out:{}".format(out.shape))

        return out ### ([8, 32, 768])
        
      

    

        

class ImgPNet(nn.Module):

    def __init__(self):
        super(ImgPNet, self).__init__()

        resnet50 = models.resnet50(pretrained=True)
        self.ImageExtract = nn.Sequential(*(list(resnet50.children())[:-2]))##特征层
        self.avg_global = nn.AdaptiveMaxPool2d((1, 1))
        self.conv_1X1 = conv(2048, 768)

    def forward(self, image):

        image_feature = self.img_embedding(image)

        # print(text_feature.shape)
        image_feature = rearrange(image_feature,'b n h d -> b (h d) n')
        return image_feature   # 16 * 49 * 768

    def img_embedding(self, image):
        image_feature = self.ImageExtract(image)
        # print(image_feature.size())

        # image_feature = self.avg_global(image_feature)
        # print(image_feature.size())
        image_feature = self.conv_1X1(image_feature)
        # print(image_feature.size())

        return image_feature



class CustomCLIP(nn.Module):
    def __init__(self,  args):
        super().__init__()
       
        dim = 512
        dropout = 0.1
        head = 4
        self.b_c = args.batch_size
        self.image_encoder = ImgPNet()
        self.text_encoder = TextExtract_Bert()
        # self.attention = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=0.1)
        self.claim_document_text_attention = MultiHeadAttention(head, dim, dim, dim, dropout=dropout)
        # self.relu = nn.LeakyReLU(0.2)
        self.linear = nn.Linear(768, 512)
        self.learn_gap = nn.Parameter(torch.randn(1,49,512)) ##作q
        self.head = nn.Sequential(
                        
                        nn.Linear(512, 256),
                        nn.Dropout(0.5),
                        nn.Linear(256, 8),
                    )
        
        # self.weight_linear_image = nn.Linear(512, 1)  # Weight for image features
        # self.weight_linear_text = nn.Linear(512, 1) 
        
        
    def forward(self, image,input_ids, attention_mask, token_type_ids):
        
        
        b_c = self.b_c
        learner = self.learn_gap.repeat(b_c,1,1)
        
        
        image_features = self.image_encoder(image) ###[8, 49, 768]
        image_features = self.linear(image_features)##([8, 49, 512])
        attention_image_features = self.claim_document_text_attention(learner,image_features,image_features)###([8, 49, 512])
        attention_image_features = torch.mean(attention_image_features, dim=1)###[8,512]
         
            
        text_features = self.text_encoder(input_ids, attention_mask, token_type_ids) ###[8, 32,768]
        text_features = self.linear(text_features) ###[8,32,512]
        attention_text_features = self.claim_document_text_attention(learner,text_features,text_features)##([8, 49, 512])
        attention_text_features = torch.mean(attention_text_features, dim=1)###[8,512]
        
        
        # weights_image = 0.4
        # weights_text = 0.6
        # weighted_attention_features = weights_image * torch.sigmoid(attention_image_features) + weights_text * torch.sigmoid(attention_text_features)
        
        
        features =  torch.mean((attention_text_features,attention_image_features),dim=1)###[8,1024]
        
        
        # attention_image_features = (weighted_attention_features)
        
        pre_out = self.head(weighted_attention_features)
        

        return pre_out ##[8, 8]