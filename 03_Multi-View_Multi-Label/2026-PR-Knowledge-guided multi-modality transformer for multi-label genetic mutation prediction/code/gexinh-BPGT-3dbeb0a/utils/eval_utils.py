import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
# from model.model_mil import MIL_fc, MIL_fc_mc
# from model.model_clam import CLAM_SB, CLAM_MB
from model.base_model import query2label, transformer_decoder
import pdb
import os
import pandas as pd
from utils.utils import *
from utils.core_utils import Accuracy_Logger
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt

def initiate_model(args, ckpt_path):
    print('Init Model')    
    # use model 
    if args.model_type == 'att_ml':
        args.adj = args.adj.to('cuda')

        # for region segmentation
        from topk.svm import SmoothTop1SVM
        instance_loss_fn = SmoothTop1SVM(n_classes = 2)
        if device.type == 'cuda':
            instance_loss_fn = instance_loss_fn.cuda()

        if args.downstream_model == 'ml_decoder':
            if args.agg_method == 'tran' and args.pretrained_model == 'hipt':
                args.model_size = 'hipt_att'
            if args.agg_method == 'cor_mil':
                args.model_size = 'hipt' if args.pretrained_model == 'hipt' else args.model_size 
                args.region_cls_model = 'corr'

            # TODO: add graphormer
            # model_dict = {"adj":args.adj, "dropout": args.drop_out, 'num_classes': args.n_label, 
            #             "model_size": args.model_size, 'k_sample': args.k_sample, 
            #             'dropout_rate':args.dropout_rate,
            #             "num_layers_decoder":args.decoder_layer,
            #             "gcn": args.gcn,
            #             "agg_method": args.agg_method,
            #             "num_layers_encoder": args.encoder_layer,
            #             "region_cls":args.region_cls,
            #             "num_region":args.num_region,
            #             "instance_loss_fn": instance_loss_fn,
            #             }

            # model = query2label(**model_dict)
            if args.graph_module == 'gcn': 
                model_dict = {"adj":args.adj, "dropout": args.drop_out, 'num_classes': args.n_label, 
                            "model_size": args.model_size, 'k_sample': args.k_sample, 
                            'dropout_rate':args.dropout_rate,
                            "num_layers_decoder":args.decoder_layer,
                            "gcn": args.gcn,
                            "agg_method": args.agg_method,
                            "num_layers_encoder": args.encoder_layer,
                            "region_cls":args.region_cls,
                            "num_region":args.num_region,
                            "instance_loss_fn": instance_loss_fn,
                            }
                model = query2label(**model_dict)

            elif args.graph_module == 'transformer':
                model_dict = {"adj":args.un_adj, "dropout": args.drop_out, 'num_classes': args.n_label, 
                        "model_size": args.model_size, 'k_sample': args.k_sample, 
                        'dropout_rate':args.dropout_rate,
                        "num_layers_decoder":args.decoder_layer,
                        "gcn": args.gcn,
                        "agg_method": args.agg_method,
                        "num_layers_encoder": args.encoder_layer,
                        "region_cls":args.region_cls,
                        "num_region":args.num_region,
                        "instance_loss_fn": instance_loss_fn,
                        "cls_token": args.cls_token,
                        }
                model = transformer_decoder(**model_dict)
            
            elif args.graph_module == 'graphormer':
                from .graph_utils import mutual_info_encoding, organ_positional_encoding
                pe = organ_positional_encoding(args, df=args.df) 
                mask = mutual_info_encoding(args, df=args.df, if_cls_token= args.cls_token)
                model_dict = {"adj":args.un_adj, "dropout": args.drop_out, 'num_classes': args.n_label, 
                        "model_size": args.model_size, 'k_sample': args.k_sample, 
                        'dropout_rate':args.dropout_rate,
                        "num_layers_decoder":args.decoder_layer,
                        "gcn": args.gcn,
                        "agg_method": args.agg_method,
                        "num_layers_encoder": args.encoder_layer,
                        "region_cls":args.region_cls,
                        "num_region":args.num_region,
                        "instance_loss_fn": instance_loss_fn,
                        "cls_token": args.cls_token,
                        'spatial_mask': mask,
                        'pe': pe,
                        'num_graph_layer':args.num_graph_layer,
                        'word_embedd': args.word_embedd,
                        'learnable_embedd': args.learnable_embedd,
                        }
                from model.base_model import graphormer_decoder
                model = graphormer_decoder(**model_dict)

    
    else: 
        # model dictionary
        model_dict = {"dropout": args.dropout}
        # model size
        if args.model_size is not None:
            model_dict.update({"size_arg": args.model_size})
        model_dict.update({'n_classes': args.n_classes})
        if args.model_type =='clam_sb':
            model = CLAM_SB(**model_dict)
        elif args.model_type =='clam_mb':
            model = CLAM_MB(**model_dict)
        elif args.model_type == 'mil':
            if args.n_classes > 2:
                model = MIL_fc_mc(**model_dict)
            else:
                model = MIL_fc(**model_dict)

    print_network(model)

    # load checkpoint
    ckpt = torch.load(ckpt_path)
    ckpt_clean = {}
    for key in ckpt.keys():
        if 'instance_loss_fn' in key:
            continue
        # ckpt_clean.update({key.replace('.module', ''):ckpt[key]})
        ckpt_clean.update({key.replace('module.', ''):ckpt[key]})
        

    model.load_state_dict(ckpt_clean, strict=True)
    model = model.cuda()
    # model.to('cuda')
    # model.relocate()
    model.eval()
    return model

def eval(dataset, args, ckpt_path):
    model = initiate_model(args, ckpt_path)
    
    print('Init Loaders')
    loader = get_simple_loader(dataset)
    patient_results, test_error, auc, df, _ = summary(model, loader, args)
    print('test_error: ', test_error)
    print('auc: ', auc)
    return model, patient_results, test_error, auc, df

def summary(model, loader, args):
    acc_logger = Accuracy_Logger(n_classes=args.n_classes)
    model.eval()
    test_loss = 0.
    test_error = 0.

    all_probs = np.zeros((len(loader), args.n_classes))
    all_labels = np.zeros(len(loader))
    all_preds = np.zeros(len(loader))

    slide_ids = loader.dataset.slide_data['slide_id']
    patient_results = {}
    for batch_idx, (data, label) in enumerate(loader):
        data, label = data.to(device), label.to(device)
        slide_id = slide_ids.iloc[batch_idx]
        with torch.no_grad():
            logits, Y_prob, Y_hat, _, results_dict = model(data)
        
        acc_logger.log(Y_hat, label)
        
        probs = Y_prob.cpu().numpy()

        all_probs[batch_idx] = probs
        all_labels[batch_idx] = label.item()
        all_preds[batch_idx] = Y_hat.item()
        
        patient_results.update({slide_id: {'slide_id': np.array(slide_id), 'prob': probs, 'label': label.item()}})
        
        error = calculate_error(Y_hat, label)
        test_error += error

    del data
    test_error /= len(loader)

    aucs = []
    if len(np.unique(all_labels)) == 1:
        auc_score = -1

    else: 
        if args.n_classes == 2:
            auc_score = roc_auc_score(all_labels, all_probs[:, 1])
        else:
            binary_labels = label_binarize(all_labels, classes=[i for i in range(args.n_classes)])
            for class_idx in range(args.n_classes):
                if class_idx in all_labels:
                    fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], all_probs[:, class_idx])
                    aucs.append(auc(fpr, tpr))
                else:
                    aucs.append(float('nan'))
            if args.micro_average:
                binary_labels = label_binarize(all_labels, classes=[i for i in range(args.n_classes)])
                fpr, tpr, _ = roc_curve(binary_labels.ravel(), all_probs.ravel())
                auc_score = auc(fpr, tpr)
            else:
                auc_score = np.nanmean(np.array(aucs))

    results_dict = {'slide_id': slide_ids, 'Y': all_labels, 'Y_hat': all_preds}
    for c in range(args.n_classes):
        results_dict.update({'p_{}'.format(c): all_probs[:,c]})
    df = pd.DataFrame(results_dict)
    return patient_results, test_error, auc_score, df, acc_logger


import numpy as np

#  CP CR CF1 OP OR OF1  mAP 的函数返回值
def calculate_metrics(labels, preds):
    """ labels: -> numpy, (samples, labels)
    """
    mAP = calculate_mAP(labels, preds)
    pc_top3, rc_top3, f1c_top3, po_top3, ro_top3, f1o_top3 = calculate_top3_metrics(labels, preds) # top-3

    return {'pc_top3': pc_top3, 'rc_top3': rc_top3, 'f1c_top3': f1c_top3,
            'po_top3': po_top3, 'ro_top3': ro_top3, 'f1o_top3': f1o_top3, 'mAP': mAP}


# top-3排序的标签
def calculate_top3_metrics(labels, preds):
    no_examples = labels.shape[0]

    top3 = np.zeros_like(preds)
    for ind_example in range(no_examples):
        top_pred_inds = np.argsort(preds[ind_example])[::-1]
        for k in range(3):
            top3[ind_example, top_pred_inds[k]] = 1

    pc_top3, rc_top3, f1c_top3 = prec_rec_f1(labels, top3)
    po_top3, ro_top3, f1o_top3 = prec_rec_f1(labels.flatten(), top3.flatten())

    return pc_top3, rc_top3, f1c_top3, po_top3, ro_top3, f1o_top3


def prec_rec_f1(labels, pred_labels):
    eps = np.finfo(np.float32).eps
    tp = labels * pred_labels
    # 
    if len(labels.shape) == 2:
        no_tp = np.sum(tp, axis=1) + eps
        no_pred = np.sum(pred_labels, axis=1) + eps
        no_pos = np.sum(labels, axis=1) + eps
    # 
    elif len(labels.shape) == 1:
        no_tp = np.sum(tp) + eps
        no_pred = np.sum(pred_labels) + eps
        no_pos = np.sum(labels) + eps

    prec_class = no_tp / no_pred + eps
    rec_class = no_tp / no_pos + eps
    f1_class = 2 * prec_class * rec_class / (prec_class + rec_class)

    return 100 * np.mean(prec_class), 100 * np.mean(rec_class), 100 * np.mean(f1_class)


# 计算mAP
def calculate_mAP(labels, preds):
    no_examples = labels.shape[0]
    no_classes = labels.shape[1]

    ap_scores = np.empty((no_classes), dtype=np.float)
    for ind_class in range(no_classes):
        ground_truth = labels[:, ind_class]
        out = preds[:, ind_class]

        sorted_inds = np.argsort(out)[::-1] # in descending order
        tp = ground_truth[sorted_inds]
        fp = 1 - ground_truth[sorted_inds]
        tp = np.cumsum(tp)
        fp = np.cumsum(fp)

        rec = tp / np.sum(ground_truth)
        prec = tp / (fp + tp)

        rec = np.insert(rec, 0, 0)
        rec = np.append(rec, 1)
        prec = np.insert(prec, 0, 0)
        prec = np.append(prec, 0)

        for ind in range(no_examples, -1, -1):
            prec[ind] = max(prec[ind], prec[ind + 1])

        inds = np.where(rec[1:] != rec[:-1])[0] + 1
        ap_scores[ind_class] = np.sum((rec[inds] - rec[inds - 1]) * prec[inds])

    return 100 * np.mean(ap_scores)