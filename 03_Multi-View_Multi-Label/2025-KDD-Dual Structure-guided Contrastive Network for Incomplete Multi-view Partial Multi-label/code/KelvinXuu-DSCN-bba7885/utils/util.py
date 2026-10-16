import os

import numpy as np
from numpy.random import randint
import torch 
import logging
import random


def setup_seed(seed_n):
    random.seed(seed_n)
    np.random.seed(seed_n)
    torch.manual_seed(seed_n)
    torch.cuda.manual_seed(seed_n)
    torch.cuda.manual_seed_all(seed_n)
    torch.backends.cudnn.deterministic = True
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.use_deterministic_algorithms(True)


def get_logger(args, main_dir='../results/logs/'):
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s: - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    fh = logging.FileHandler(
        main_dir + args.dataset_name + '_' + str(args.missing_rate_data) + '_' + str(args.missing_rate_label) + '_' + str(args.training_ratio) + '.logs')

    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    logger.addHandler(fh)
    
    return logger


def output_args(logger, args):
    inform = "dataset:{}\t\ttraining_rate:{:.1f}\t\tdata_ms_rate:{:.1f}\tlabel_ms_rate:{:.1f}".format(
            args.dataset_name, args.training_ratio, args.missing_rate_data, args.missing_rate_label
            )
    logger.info(inform)
    inform = "learning_rate:{:.0e}\tepochs:{}\t\t\t\tbatch_size:{}\t\tseed:{}".format(
                args.lr, args.epochs, args.batch_size, args.seed
            )
    logger.info(inform)
    inform = "alpha:{:.0e}\t\t\tbeta:{:.0e}\t\t\t\tmask_rate:{:.1f}\t\tearly_stop:{}".format(
            args.alpha, args.beta, args.mask_rate, args.early_stop
            )
    logger.info(inform)
    inform = "ins_temperature:{:.1f}\tlabel_temperature:{:.1f}\tthreshold:{:.2f}\t\tfold:{}".format(
            args.ins_temperature, args.label_temperature, args.threshold, args.fold_idx
            )
    logger.info(inform)
    inform = "network_dropout:{:.1f}\tembedding_dim:{}\t\tmid_dim:{}\t\tlayer_num:{}".format(
            args.dropout, args.hidden, args.mid_dim, args.layer_num
            )
    logger.info(inform)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.vals = []
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.vals = []
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val):
        self.vals.append(val)
        self.sum = np.sum(self.vals)
        self.count = len(self.vals)
        self.avg = np.mean(self.vals)
        self.std = np.std(self.vals)
        self.min = min(self.vals)
        self.min_ind = self.vals.index(self.min)
        self.max = max(self.vals)
        self.max_ind = self.vals.index(self.max)


def get_similarity(features):
    dist = -0.5 * torch.cdist(features, features, p=2) ** 2
    dist = torch.exp(dist)

    return dist


def fusion_mask_similarity(dists, mask):
    n = mask.shape[0]
    v = mask.shape[1]
    dist_fusion = torch.zeros((n, n))
    sum_matrix = torch.zeros((n, n))

    for i in range(v):
        expand_dist = torch.zeros((n, n))
        mask_v_matrix = torch.mul(mask[:, i].int().view(-1, 1), mask[:, i].int())
        expand_dist[mask_v_matrix == 1] = dists[i].flatten()
        dist_fusion += expand_dist
        sum_matrix += mask_v_matrix
    sum_matrix[sum_matrix == 0] = 1
    dist_fusion /= sum_matrix

    return dist_fusion


def calc_res(logger, final_res):
    metrics_num = 6
    metrics = [res[:metrics_num] for res in final_res]
    metrics_array = np.array(metrics)
    mean_values = np.mean(metrics_array, axis=0)
    std_values = np.std(metrics_array, axis=0)
    metrics_names = ["AP", "1-HL", "1-RL", "AUC", "1-OE", "1-Cov"]

    for i in range(metrics_num):
        lst = [f"{x * 100:.2f}" for x in metrics_array[:, i]]
        lst = list(map(float, lst))

        logger.info(f"{metrics_names[i]}: {lst}")

    inform = "\n"
    for i in range(metrics_num):
        inform += "\t\t\t{}\tmean: {:.2f}    std: {:.2f}\n".format(metrics_names[i], mean_values[i] * 100,
                                                                std_values[i] * 100)

    logger.info(inform)

