from PIL import Image
import torch
import argparse
import warnings
import os
import torch.nn as nn
from dataset import *
import pandas as pd
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.optim.lr_scheduler import _LRScheduler
import numpy as np
import tqdm
from torch.autograd import Variable
from metrics import *
from dbl import *
from asl import *
warnings.filterwarnings('ignore')
from model import *

def get_args_parser():
    parser = argparse.ArgumentParser('MLCLIP script', add_help=False)
    parser.add_argument('--device', default='cuda', help='device to use for training / testing')
    parser.add_argument('--seed', default='0', type=int, help='seed')
    # parser.add_argument('--pretrain_clip_path', default='/model/zfr888/ljt/clip/ViT-B-16.pt', type=str, help='path of pretrained clip ckpt')
    parser.add_argument('--nb_classes', default=8, type=int, help='dataset classes')
    parser.add_argument('--dataset', default='voc-lt', type=str, help='dataset name')
    parser.add_argument('--batch_size', default=8, type=int, help='batch size')
    parser.add_argument('--epochs', default=50, type=int, help='train epochs')
    parser.add_argument('--from_scratch', action='store_true', default=False, help='freeze the paras of clip image encoder')
    return parser


def collate_fn(data):
    bert_path = '/model/zfr888/dual/bert-base-uncased'  # 指定 BERT 模型路径
    model_class, tokenizer_class, pretrained_weights = (transformers.BertModel, transformers.BertTokenizer, bert_path)
    tokenizer = tokenizer_class.from_pretrained(pretrained_weights)

    
    inputs = [i[0] for i in data]
    labels = [i[1] for i in data]
    sents = [i[2] for i in data]
    # print(sents,labels)
    labels = torch.tensor(labels)
    inputs = torch.tensor([item.cpu().detach().numpy() for item in inputs]).cuda()
    
    #编码
    data = tokenizer.batch_encode_plus(batch_text_or_text_pairs=sents,
                                   truncation=True,
                                   padding='max_length',
                                   max_length=50,
                                   return_tensors='pt',
                                   return_length=True)

    #input_ids:编码之后的数字
    #attention_mask:是补零的位置是0,其他位置是1
    input_ids = data['input_ids']
    attention_mask = data['attention_mask']
    token_type_ids = data['token_type_ids']

    return input_ids, attention_mask, token_type_ids, inputs, labels


def main(args):
    print(args)

    # fix the seed for reproducibility
    torch.manual_seed(args.seed)

    '''
    model
    '''
    model = CustomCLIP(args)
    model = model.cuda()
    


    
    
    ###断点续连
    # model.load_state_dict(torch.load('/model/zfr888/ljt/LMPT/waste_fc_epoch31/fc_voc-lt_btz_8.pt'))
    '''
    dataset and dataloader
    '''
    train_dataset =  Voc2007Classification('/data/zfr888/ljt/voc', 'trainval', inp_name='data/mydata/waste_glove_word2vec.pkl')
    test_dataset = Voc2007Classification('/data/zfr888/ljt/voc', 'test', inp_name='data/mydata/waste_glove_word2vec.pkl')

    train_loader = torch.utils.data.DataLoader(
                                            train_dataset,
                                            batch_size=args.batch_size,
                                            collate_fn=collate_fn,
                                            shuffle=True,
                                            drop_last=True
                                        )
    
    
    
    test_loader = torch.utils.data.DataLoader(
                                            test_dataset,
                                            batch_size=args.batch_size,
                                            collate_fn=collate_fn,
                                            shuffle=False,
                                            drop_last=False
                                        )
    '''
    loss function
    '''
    # loss_function = nn.MultiLabelSoftMarginLoss()
    # loss_function = nn.BCEWithLogitsLoss()
    if args.dataset == 'coco-lt':
        freq_file = '../data/coco/class_freq.pkl'
    elif args.dataset == 'voc-lt' or args.dataset == 'voc':
        freq_file='/data/zfr888/ljt/voc/class_freq.pkl'
    # if args.dataset == 'coco-lt':
    #     loss_function = ResampleLoss(
    #             use_sigmoid=True,
    #             reweight_func='rebalance',
    #             focal=dict(focal=True, balance_param=2.0, gamma=2),
    #             logit_reg=dict(neg_scale=2.0, init_bias=0.05),
    #             map_param=dict(alpha=0.1, beta=10.0, gamma=0.2),
    #             loss_weight=1.0
    #         )
    # elif args.dataset == 'voc-lt':
    #     loss_function = ResampleLoss(
    #             use_sigmoid=True,
    #             reweight_func='rebalance',
    #             focal=dict(focal=True, balance_param=2.0, gamma=2),
    #             logit_reg=dict(neg_scale=5.0, init_bias=0.05),
    #             map_param=dict(alpha=0.1, beta=10.0, gamma=0.3),
    #             loss_weight=1.0
    #         )
    # loss_function = nn.MultiLabelSoftMarginLoss()
    loss_function = AsymmetricLossOptimized(gamma_neg=4, gamma_pos=0, clip=0.05, disable_torch_grad_focal_loss=True)
    # loss_function = nn.BCEWithLogitsLoss()
    '''
    optimizer
    '''
    if args.from_scratch is True:
        print('not freeze parameters of CLIP image encoder')
        # optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
        optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    else:
        print('freeze parameters of CLIP image encoder')
        # optimizer = optim.SGD(model.fc.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
        optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

    exp_lr_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, patience = 5, mode = 'max', verbose = True, min_lr = 1e-7) 

    if args.dataset=='coco-2017' or 'coco-lt':
        dataset_classes = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
            'truck', 'boat', 'traffic_light', 'fire_hydrant', 'stop_sign',
            'parking_meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
            'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
            'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
            'sports_ball', 'kite', 'baseball_bat', 'baseball_glove', 'skateboard',
            'surfboard', 'tennis_racket', 'bottle', 'wine_glass', 'cup', 'fork',
            'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
            'broccoli', 'carrot', 'hot_dog', 'pizza', 'donut', 'cake', 'chair',
            'couch', 'potted_plant', 'bed', 'dining_table', 'toilet', 'tv',
            'laptop', 'mouse', 'remote', 'keyboard', 'cell_phone', 'microwave',
            'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
            'scissors', 'teddy_bear', 'hair_drier', 'toothbrush'
            ]
    elif args.dataset=='voc' or 'voc-lt': 
        dataset_classes = ['aeroplane', 'bicycle', 'bird', 'boat',
                     'bottle', 'bus', 'car', 'cat', 'chair',
                     'cow', 'diningtable', 'dog', 'horse',
                     'motorbike', 'person', 'pottedplant',
                     'sheep', 'sofa', 'train', 'tvmonitor']
    best_mAP = 0.0
    sf = nn.Softmax(dim=1)
    
    train_map=[]
    test_map=[]
    
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        gt_labels = []
        predict_p = []
        for i, (input_ids, attention_mask, token_type_ids,inputs,labels) in tqdm.tqdm(enumerate(train_loader),desc="Processing", ncols=100,total=len(train_loader)):
            
            labels = labels.to(torch.float32)
            labels = torch.squeeze(labels, 1)
            inputs = Variable(inputs.cuda())
            labels = Variable(labels.cuda())
            input_ids = Variable(input_ids.cuda())
            attention_mask = Variable(attention_mask.cuda())
            token_type_ids = Variable(token_type_ids.cuda())
            optimizer.zero_grad()     
            outputs = model(inputs,input_ids, attention_mask, token_type_ids)
            gt_labels.extend(labels.cpu().numpy().tolist())
            predict_p.extend(sf(outputs).cpu().detach().numpy())
            loss = loss_function(outputs, labels)
            # print(loss)
            running_loss += loss.data.item()
            loss.backward()
            optimizer.step()
        mAP, APs = eval_map(predict_p, gt_labels)
        print("train epoch[{}/{}] loss:{:.3f} train mAP:{}".format(epoch + 1, args.epochs, loss, mAP)) 
        train_map.append(mAP)
        model.eval()
                
        with torch.no_grad():
            gt_labels = []
            predict_p = []
            running_loss = 0.0
            for i, (input_ids, attention_mask, token_type_ids,inputs,labels) in tqdm.tqdm(enumerate(test_loader),desc="Processing", ncols=100,total=len(test_loader)):
                
                labels = labels.to(torch.float32)
                labels = torch.squeeze(labels, 1)
                inputs = Variable(inputs.cuda())
                labels = Variable(labels.cuda())
                input_ids = Variable(input_ids.cuda())
                attention_mask = Variable(attention_mask.cuda())
                token_type_ids = Variable(token_type_ids.cuda())
                outputs = model(inputs,input_ids, attention_mask, token_type_ids)
                gt_labels.extend(labels.cpu().numpy().tolist())
                predict_p.extend(sf(outputs).cpu().detach().numpy())
                loss = loss_function(outputs, labels)
                running_loss += loss.data.item()
            mAP, APs = eval_map(predict_p, gt_labels)
            print("test epoch[{}/{}] loss:{:.3f} test mAP:{}".format(epoch + 1, args.epochs, loss, mAP))
            test_map.append(mAP)
            current_mAP = mAP
            exp_lr_scheduler.step(current_mAP)
            if current_mAP > best_mAP:
                best_mAP = current_mAP
                if args.from_scratch is True:
                    torch.save(model.state_dict(), f'/model/zfr888/ljt/LMPT/bert_resnet/invoc_bert_resnet50_{args.dataset}_btz_{args.batch_size}_scratch.pt') 
                    print(f'checkpoint saved at /model/zfr888/ljt/LMPT/bert_resnet/invoc_bert_resnet50_{args.dataset}_btz_{args.batch_size}_scratch.pt')
                else:
                    torch.save(model.state_dict(), f'/model/zfr888/ljt/LMPT/bert_resnet/bert_resnet50_{args.dataset}_btz_{args.batch_size}.pt') 
                    print(f'checkpoint saved at /model/zfr888/ljt/LMPT/bert_resnet/bert_resnet50_{args.dataset}_btz_{args.batch_size}.pt')
                ltAnalysis(APs, args.dataset)

                
if __name__ == '__main__':
    parser = argparse.ArgumentParser('MLCLIP script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
