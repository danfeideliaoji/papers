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
        
        self.linear = nn.Linear(768, 512)
        self.learn_gap = nn.Parameter(torch.randn(1,49,512)) ##作q
        self.head = nn.Sequential(
                        nn.Linear(512*2, 20),
                    )
        
        
    def forward(self, image,input_ids, attention_mask, token_type_ids):
        
        
        b_c = self.b_c
        # print("----------------step1 is Ok,and b_c is {}-----------------------".format(b_c))
        
        learner = self.learn_gap.repeat(b_c,1,1)
        # print("----------------step2 is Ok-----------------------")
        
        image_features = self.image_encoder(image) ###[8, 49, 768]
        # print("----------------step3 is Ok-----------------------")
        
        
        
        image_features = self.linear(image_features)##([8, 49, 512])
        # print("----------------step4 is Ok-----------------------")
        # print("img_feature:{}".format(image_features.shape))
         
        text_features = self.text_encoder(input_ids, attention_mask, token_type_ids) ###[8, 32,768]
        # print("----------------step5 is Ok-----------------------")
        
        text_features = self.linear(text_features) ###[8,32,512]
        # print("----------------step6 is Ok-----------------------")
        
        
        attention_text_features = self.claim_document_text_attention(learner,text_features,text_features)##([8, 49, 512])
        # print("----------------step7 is Ok-----------------------")
        attention_text_features = torch.sum(attention_text_features, dim=1)###[8,512]
        # print("attention_text_features:{}".format(attention_text_features.shape))
        
        
        attention_image_features = self.claim_document_text_attention(learner,image_features,image_features)###([8, 49, 512])
        # print("----------------step8 is Ok-----------------------")
        # print("attention_image_features:{}".format(attention_image_features.shape))
        
        attention_image_features = torch.sum(attention_image_features, dim=1)###[8,512]
        
        ##class_head
        features = torch.cat((attention_text_features,attention_image_features),dim=1)###[8,1024]
        # print("features:{}".format(features.shape))
        # print("----------------step9 is Ok-----------------------")
        
        
        pre_out = self.head(features)
        # print("----------------step10 is Ok-----------------------")

        return pre_out ##[8, 8]