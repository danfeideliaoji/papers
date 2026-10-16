import argparse
import os, sys
import time
import random
import json

import torch
from torch.optim import lr_scheduler
import torch.optim
import torch.utils.data

import _init_paths
from lib.dataset.get_dataset import get_datasets

from lib.utils.logger import setup_logger
from lib.utils.meter import AverageMeter, AverageMeterHMS, ProgressMeter
from lib.utils.helper import clean_state_dict, function_mAP, get_raw_dict, ModelEma, add_weight_decay
from lib.utils.losses import AsymmetricLoss, TwoWayLoss_class_mine, ASL_UL, WeightedASL_UL, ASL_UL_weight
from sklearn import metrics

from lib.ML_decoder.backbone.ML_decoder import ClasswiseModel
from lib.ML_decoder.loss_UCL import unsupervised_contrastive_loss_flatten

from torch.utils.data import Subset, random_split, ConcatDataset
from lib.utils.helper_DiCaP import save_data_in_train, smooth_for_voc, smooth_for_large_dataset


NUM_CLASS = {'voc': 20, 'coco': 80, 'nus': 81, 'awa': 85}
# os.environ["CUDA_VISIBLE_DEVICES"] = '0'
def str2bool(input):
    if isinstance(input, bool):
        return input

    if input.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif input.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def parser_args():
    parser = argparse.ArgumentParser(description='Main')

    # data
    parser.add_argument('--dataset_name', default='voc', choices=['voc', 'coco', 'nus', 'awa'], 
                        help='dataset name')
    parser.add_argument('--dataset_dir',  default='./data', metavar='DIR', 
                        help='dir of all datasets')
    parser.add_argument('--img_size', default=224, type=int,
                        help='size of input images')
    parser.add_argument('--output', default='debug', metavar='DIR', 
                        help='path to output folder')
    parser.add_argument('--lb_ratio', default=0.05, type=float,
                        help='the ratio of lb:(lb+ub)')
    parser.add_argument('--cutout', default=0.5, type=float, 
                        help='cutout factor')
    parser.add_argument("--grid_perside", type=int, default=2,
                        help="Number of grids per side for local images. (example: 2)")

    # train
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 8)')
    parser.add_argument('--epochs', default=40, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-b', '--warmup_batch_size', default=32, type=int,
                        help='batch size for warmup')
    parser.add_argument('--lr', '--learning_rate', default=1e-4, type=float, metavar='LR', 
                        help='initial learning rate', dest='lr')
    parser.add_argument('--wd', '--weight_decay', default=1e-4, type=float,metavar='W', 
                        help='weight decay (default: 1e-2)', dest='weight_decay')
    parser.add_argument('-p', '--print_freq', default=400, type=int, metavar='N', 
                        help='print frequency (default: 10)')
    parser.add_argument('--amp', action='store_true', default=True,
                        help='apply amp')
    parser.add_argument('--early_stop', action='store_true', default=False,
                        help='apply early stop')
    parser.add_argument('--optim', default='adamw', type=str,
                        help='optimizer used')
    parser.add_argument('--warmup_epochs', default=12, type=int,
                        help='the number of epochs for warmup')
    parser.add_argument('--loss_lb', default='asl', type=str, 
                        help='used_loss for lb')
    parser.add_argument('--loss_ub', default='asl', type=str, 
                        help='used_loss for ub')
    parser.add_argument('--cat_strategy', default='f1', type=str, 
                        help='')
    parser.add_argument('--beta', default=1.0, type=float, 
                        help='')


    # random seed
    parser.add_argument('--seed', default=1, type=int,
                        help='seed for initializing training. ')

    # model
    parser.add_argument('--net', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--is_data_parallel', action='store_true', default=False,
                        help='on/off nn.DataParallel()')
    parser.add_argument("--parallel_devices", type=int, default=[0,1], nargs="+",
                        help="device list (example: [0,1])")
    parser.add_argument('--ema_decay', default=0.9997, type=float, metavar='M',
                        help='decay of model ema')
    parser.add_argument('--resume', default=None, type=str, 
                        help='path to latest checkpoint (default: none)')

    # new model
    parser.add_argument('--percent', default=0.8, type=float, metavar='M',
                        help='the percent of labeled data used for training, the rest is used for distribution estimation')
    parser.add_argument('--pos_threshold_target', type=float, default=0.5, help='the target ratio of positive samples for thresholding (default: 0.5)')
    parser.add_argument('--neg_threshold_target', type=float, default=0.01, help='the target ratio of negative samples for thresholding (default: 0.001)')

    parser.add_argument('--method', default='test', help='method')


    args = parser.parse_args()
    
    if args.lb_ratio <= 0.05:
        args.lb_bs = 4
        args.ub_bs = 60
    elif args.lb_ratio == 0.1:
        args.lb_bs = 8
        args.ub_bs = 56
    elif args.lb_ratio == 0.15:
        args.lb_bs = 12
        args.ub_bs = 52
    elif args.lb_ratio == 0.2:
        args.lb_bs = 16
        args.ub_bs = 48

    if args.dataset_name == 'voc':
        args.lb_bs = int(args.lb_bs / 2)
        args.ub_bs = int(args.ub_bs / 2)

    
    args.output = './output/' + args.output
    args.resume = '%s/%s/%s/warmup_%s_%s/warmup_model.pth.tar'%(args.output, args.dataset_name, args.lb_ratio, args.loss_lb, args.warmup_epochs)
    
    args.n_classes = NUM_CLASS[args.dataset_name]
    args.dataset_dir = os.path.join(args.dataset_dir, args.dataset_name) 
    
    args.output = os.path.join(args.output, args.dataset_name,  '%s'%args.lb_ratio,  '%s'%args.method)

    return args


def get_args():
    args = parser_args()
    return args


def main():
    args = get_args()
    
    
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    os.makedirs(args.output, exist_ok=True)

    logger = setup_logger(output=args.output, color=False, name="XXX")
    logger.info("Command: "+' '.join(sys.argv))

    path = os.path.join(args.output, "config.json")
    with open(path, 'w') as f:
        json.dump(get_raw_dict(args), f, indent=2)
    logger.info("Full config saved to {}".format(path))

    return main_worker(args, logger)

def main_worker(args, logger):

    # Data loading code
    lb_train_dataset, ub_train_dataset, val_dataset = get_datasets(args)
    print("len(lb_train_dataset):", len(lb_train_dataset)) 
    print("len(ub_train_dataset):", len(ub_train_dataset))
    print("len(val_dataset):", len(val_dataset))


    total_len = len(lb_train_dataset)
    split_ratio = args.percent  # 80% for superviced，20% for estimation
    train_len = int(total_len * split_ratio)
    valid_len = total_len - train_len
    lb_train_dataset_train, lb_train_dataset_valid = random_split(
        lb_train_dataset, [train_len, valid_len],
        generator=torch.Generator().manual_seed(args.split_seed)
    )

    # unsupervised set
    combined_unlabeled_dataset = ConcatDataset([lb_train_dataset_valid, ub_train_dataset])


    # superviced training set
    lb_train_loader = torch.utils.data.DataLoader(
        lb_train_dataset_train, batch_size=int(args.warmup_batch_size * split_ratio), shuffle=True,
        num_workers=args.workers, pin_memory=False)

    # estimation set
    lb_valid_loader = torch.utils.data.DataLoader(
        lb_train_dataset_valid, batch_size=64, shuffle=False,
        num_workers=args.workers, pin_memory=False)

    # unsupervised set
    ub_train_loader = torch.utils.data.DataLoader(
        combined_unlabeled_dataset, batch_size=args.ub_bs, shuffle=True,
        num_workers=args.workers, pin_memory=False)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=64, shuffle=False,
        num_workers=args.workers, pin_memory=False)


    args.dim_embed = 512
    model = ClasswiseModel(args.net, args.n_classes, args.dim_embed)

    if args.resume:
        if os.path.exists(args.resume):
            logger.info("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(os.path.join(args.resume))

            args.start_epoch = checkpoint['epoch']+1
            if 'state_dict' in checkpoint:
                if checkpoint['regular_mAP'][-1] > checkpoint['ema_mAP'][-1]:
                    state_dict = clean_state_dict(checkpoint['state_dict'])
                    logger.info("=> using regular model")
                else:
                    state_dict = clean_state_dict(checkpoint['state_dict_ema'])
                    logger.info("=> using ema model")
            else:
                raise ValueError("No model or state_dicr Found!!!")

            model.load_state_dict(state_dict)
            logger.info("=> loaded checkpoint '{}' (epoch {})"
                .format(args.resume, checkpoint['epoch']))
            
            del checkpoint
            del state_dict
            # del state_dict_ema
            torch.cuda.empty_cache() 
        else:
            logger.info("=> no checkpoint found at '{}'".format(args.resume))


    if args.is_data_parallel:
        model = torch.nn.DataParallel(model, device_ids=args.parallel_devices)

    model = model.cuda()
    ema_m = ModelEma(model, args.ema_decay)

    epoch_time = AverageMeterHMS('TT')
    eta = AverageMeterHMS('ETA', val_only=True)
    mAPs = AverageMeter('mAP', ':5.5f', val_only=True)
    mAPs_ema = AverageMeter('mAP_ema', ':5.5f', val_only=True)
    progress = ProgressMeter(
        args.epochs,
        [eta, epoch_time, mAPs, mAPs_ema],
        prefix='=> Test Epoch: ')

    # optimizer & scheduler
    optimizer = set_optimizer(model, args)
    args.steps_train = len(ub_train_loader)
    scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, steps_per_epoch=args.steps_train, epochs=args.epochs, pct_start=0.2)

    end = time.time()
    best_epoch = -1
    best_regular_mAP = 0
    best_regular_epoch = -1
    best_ema_mAP = 0
    regular_mAP_list = []
    ema_mAP_list = []
    best_mAP = 0
    best_CF1 = 0
    best_OF1 = 0
    thresholds = torch.zeros(args.n_classes).cuda()


    # Used loss
    if args.loss_lb == 'bce':
        criterion_lb = torch.nn.BCEWithLogitsLoss(reduction='none')
    elif args.loss_lb == 'asl':
        criterion_lb = AsymmetricLoss(gamma_neg=4, gamma_pos=0, clip=0.05, disable_torch_grad_focal_loss=True, reduction='none')
    elif args.loss_lb == 'tw':
        criterion_lb = TwoWayLoss_class_mine()

    criterion_ub = ASL_UL_weight()

    from torch.utils.tensorboard import SummaryWriter
    summary_writer = SummaryWriter(log_dir=args.output)

    ################################################################################################################
    #||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||#
    args.train_statis = os.path.join(args.output, 'train_statis')
    os.makedirs(args.train_statis, exist_ok=True)
    args.val_statis = os.path.join(args.output, 'val_statis')
    os.makedirs(args.val_statis, exist_ok=True)

    args.threshold0 = 0.5
    args.threshold1 = 0.5
    threshold0 = torch.ones(args.n_classes).cuda() * args.threshold0
    threshold1 = torch.ones(args.n_classes).cuda() * args.threshold1


    torch.cuda.empty_cache()
    for epoch in range(args.start_epoch, args.epochs):

        torch.cuda.empty_cache()

        weight = update_weight(lb_train_dataset_valid, ema_m.module, args, logger)


        loss = train(lb_train_loader, ub_train_loader, model, ema_m, optimizer, scheduler, epoch, args, logger, criterion_lb, criterion_ub, weight, threshold0, threshold1)    

        threshold0, threshold1 = dynamic_threshold_generate(lb_train_dataset, ema_m.module, threshold0, threshold1, args)


        if summary_writer:
            # tensorboard logger
            summary_writer.add_scalar('train_loss', loss, epoch)
            summary_writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], epoch)

        # evaluate on validation set
        logger.info("Test based on origin model")
        mAP, CF1, OF1 = validate(val_loader, model, args, logger, epoch, thresholds)
        logger.info("Test based on ema model")
        mAP_ema, CF1_ema, OF1_ema = validate(val_loader, ema_m.module, args, logger, epoch, thresholds)


        mAPs.update(mAP)
        mAPs_ema.update(mAP_ema)
        epoch_time.update(time.time() - end)
        end = time.time()
        eta.update(epoch_time.avg * (args.epochs - epoch - 1))

        regular_mAP_list.append(mAP)
        ema_mAP_list.append(mAP_ema)

        progress.display(epoch, logger)

        if summary_writer:
            # tensorboard logger
            summary_writer.add_scalar('val_mAP', mAP, epoch)
            summary_writer.add_scalar('val_mAP_ema', mAP_ema, epoch)


        # remember best (regular) mAP and corresponding epochs
        if mAP > best_regular_mAP:
            best_regular_mAP = max(best_regular_mAP, mAP)
            best_regular_epoch = epoch
        if mAP_ema > best_ema_mAP:
            best_ema_mAP = max(mAP_ema, best_ema_mAP)
            best_ema_epoch = epoch

        if mAP_ema > mAP:
            mAP = mAP_ema
            CF1 = CF1_ema
            OF1 = OF1_ema

        is_best = mAP > best_mAP
        if is_best:
            best_epoch = epoch
            best_mAP = mAP
            best_CF1 = CF1
            best_OF1 = OF1
            state_dict = model.state_dict()
            state_dict_ema = ema_m.module.state_dict()
            save_checkpoint({
                'epoch': epoch,
                'state_dict': state_dict,
                'state_dict_ema': state_dict_ema,
                'regular_mAP': regular_mAP_list,
                'ema_mAP': ema_mAP_list,
                'best_regular_mAP': best_regular_mAP,
                'best_ema_mAP': best_ema_mAP,
                'optimizer' : optimizer.state_dict(),
            }, is_best=True, filename=os.path.join(args.output, 'best_model.pth.tar'))

        logger.info("{} | Set best mAP {} , CF1 {} , OF1 {} in ep {}".format(epoch, best_mAP, best_CF1, best_OF1, best_epoch))
        logger.info("   | best regular mAP {} in ep {}".format(best_regular_mAP, best_regular_epoch))


        # early stop
        if args.early_stop:
            if best_epoch >= 0 and epoch - max(best_epoch, best_regular_epoch) > 5:
                if len(ema_mAP_list) > 1 and ema_mAP_list[-1] < best_ema_mAP:
                    logger.info("epoch - best_epoch = {}, stop!".format(epoch - best_epoch))
                    break

    print("Best mAP:", best_mAP)

    if summary_writer:
        summary_writer.close()
    
    return 0

    
def set_optimizer(model, args):

    if args.optim == 'adam':
        parameters = add_weight_decay(model, args.weight_decay)
        optimizer = torch.optim.Adam(params=parameters, lr=args.lr, weight_decay=0)  # true wd, filter_bias_and_bn
    elif args.optim == 'adamw':
        param_dicts = [
            {"params": [p for n, p in model.named_parameters() if p.requires_grad]},
        ]
        optimizer = getattr(torch.optim, 'AdamW')(
            param_dicts,
            args.lr,
            betas=(0.9, 0.999), eps=1e-08, weight_decay=args.weight_decay
        )

    return optimizer

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    if is_best:
        torch.save(state, filename)


def train(lb_train_loader, ub_train_loader, model, ema_m, optimizer, scheduler, epoch, args, logger, criterion_lb, criterion_ub, weight, threshold0, threshold1):


    global_weight_matrix = weight['global_weight_matrix']  # [num_classes, num_bins]
    bin_edges = weight['bin_edges']  # [num_bins+1]
    

    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
    loss_lb_ori = AverageMeter('L_lb_ori', ':5.3f')
    loss_ub_ori = AverageMeter('L_ub_ori', ':5.3f')
    loss_lb_pat = AverageMeter('L_lb_pat', ':5.3f')
    loss_ub_pat = AverageMeter('L_ub_pat', ':5.3f')
    losses = AverageMeter('Loss', ':5.3f')
    lr = AverageMeter('LR', ':.3e', val_only=True)
    mem = AverageMeter('Mem', ':.0f', val_only=True)
    progress = ProgressMeter(
        args.steps_train,
        [loss_lb_ori, loss_ub_ori, loss_lb_pat, loss_ub_pat, lr, losses, mem],
        prefix="Epoch: [{}/{}]".format(epoch, args.epochs))

    def get_learning_rate(optimizer):
        for param_group in optimizer.param_groups:
            return param_group['lr']

    lr.update(get_learning_rate(optimizer))
    logger.info("lr:{}".format(get_learning_rate(optimizer)))

    # switch to train mode
    model.train()
    outputs_ori_list = []
    outputs_pat_list = []
    outputs_mix_list = []
    targets_list = []
    lb_train_iter = iter(lb_train_loader)

    pseudo_label = []
    true_label = []

    UCL = unsupervised_contrastive_loss_flatten(temperature=0.1, use_ema=False).cuda()


    for i, ((inputs_w_ub_list, inputs_s_ub_list), labels_ub, index) in enumerate(ub_train_loader):

        try:
            (_, inputs_s_lb_list), labels_lb, _= next(lb_train_iter)
        except:
            lb_train_iter = iter(lb_train_loader)
            (_, inputs_s_lb_list), labels_lb, _ = next(lb_train_iter)

        lb_batch_size = labels_lb.shape[0]
        ub_batch_size = labels_ub.shape[0]

        # cuda
        inputs_w_ub_ori = inputs_w_ub_list[0].cuda(non_blocking=True)
        
        inputs_s_ub_ori = inputs_s_ub_list[0].cuda(non_blocking=True)
        
        inputs_s_lb_ori = inputs_s_lb_list[0].cuda(non_blocking=True)
        
        labels_lb = labels_lb.cuda(non_blocking=True).float()

        # Generate the pseudo labels
        with torch.no_grad():
            outputs_ub_ori, feature_w = ema_m.module(inputs_w_ub_ori, return_feature=True)
            outputs_ub_ori_ = torch.sigmoid(outputs_ub_ori)

            thresh_mid = torch.ones(args.n_classes) * 0.5
            thresh_mid = thresh_mid.cuda()
            thresholds_flatten_mid = thresh_mid.unsqueeze(0).repeat(ub_batch_size, 1)

            # Generate pseudo-labels
            pseudo_labels_ub = torch.zeros_like(labels_ub).cuda()
            pseudo_labels_ub[torch.logical_and(outputs_ub_ori_ >= thresholds_flatten_mid, outputs_ub_ori_ >= threshold1)] = 1
            pseudo_labels_ub[torch.logical_and(outputs_ub_ori_ <= thresholds_flatten_mid, outputs_ub_ori_ <= threshold0)] = -1

            batch_size, num_classes = outputs_ub_ori_.shape
            sample_weights = torch.zeros_like(outputs_ub_ori_)
            
            for b in range(batch_size):
                for c in range(num_classes):
                    pred_val = outputs_ub_ori_[b, c].item()
                    bin_idx = torch.searchsorted(bin_edges, pred_val).item() - 1
                    bin_idx = min(max(0, bin_idx), len(bin_edges) - 2) 
                    sample_weights[b, c] = global_weight_matrix[bin_idx]
            
            
            sample_weights[pseudo_labels_ub == -1] = 1.0 - sample_weights[pseudo_labels_ub == -1] 
            sample_weights[pseudo_labels_ub == 0] = 0

        pseudo_label.append(pseudo_labels_ub.detach().cpu())
        true_label.append(labels_ub.detach().cpu())

        inputs_ori = torch.cat([inputs_s_lb_ori, inputs_s_ub_ori], dim=0)
        
        with torch.cuda.amp.autocast(enabled=args.amp):
            outputs_ori, feature_s_all = model(inputs_ori, return_feature=True)
            
        n_lb_ori = inputs_s_lb_ori.shape[0]
        logits_lb_ori = outputs_ori[:n_lb_ori]
        logits_ub_ori = outputs_ori[n_lb_ori:]

        feature_s = feature_s_all[n_lb_ori:]

        Lx_ori = criterion_lb(logits_lb_ori, labels_lb).sum()
        Lu_ori = criterion_ub(logits_ub_ori, pseudo_labels_ub, sample_weights).sum()
        
        if(pseudo_labels_ub == 0).sum() > 0:
            loss_0 = UCL(feature_w[pseudo_labels_ub == 0], feature_s[pseudo_labels_ub == 0])
            loss_1 = UCL(feature_s[pseudo_labels_ub == 0], feature_w[pseudo_labels_ub == 0])
            loss_ucl = loss_0 + loss_1
        else:
            loss_ucl = torch.tensor(0.0).cuda()

        loss = Lx_ori + Lu_ori + loss_ucl 

        # record loss
        loss_lb_ori.update(Lx_ori.item(), lb_batch_size)
        loss_ub_ori.update(Lu_ori.item(), ub_batch_size)
        losses.update(loss.item(), lb_batch_size + ub_batch_size)

        
        mem.update(torch.cuda.max_memory_allocated() / 1024.0 / 1024.0)

        # add list
        outputs_ori_list.append(torch.sigmoid(outputs_ub_ori_).cpu().numpy())

        targets_list.append(labels_ub.numpy())

        # compute gradient and do SGD step
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        # one cycle learning rate
        scheduler.step()
        lr.update(get_learning_rate(optimizer))
        ema_m.update(model)

        if i % args.print_freq == 0:
            progress.display(i, logger)


    return losses.avg


@torch.no_grad()
def validate(val_loader, model, args, logger, epoch, thresholds):
    batch_time = AverageMeter('Time', ':5.3f')
    mem = AverageMeter('Mem', ':.0f', val_only=True)

    progress = ProgressMeter(
        len(val_loader),
        [batch_time, mem],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()
    outputs_ori_list = []
    outputs_pat_list = []
    outputs_mix_list = []
    targets_list = []
        
    end = time.time()
    for i, (inputs_list, targets, index) in enumerate(val_loader):

        batch_size = targets.shape[0]

        inputs_ori = inputs_list[0].cuda(non_blocking=True)
        
        targets = targets.cuda(non_blocking=True)
        # mixed precision ---- compute outputs
        with torch.cuda.amp.autocast(enabled=args.amp):
            outputs_ori = model(inputs_ori)
                    
        outputs_ori_ = outputs_ori
        

        outputs_ori_sm = torch.sigmoid(outputs_ori_)
        

        # add list
        outputs_ori_list.append(outputs_ori_sm.detach().cpu())
        
        targets_list.append(targets.detach().cpu())

        # record memory
        mem.update(torch.cuda.max_memory_allocated() / 1024.0 / 1024.0)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i, logger)
    import numpy as np
    labels = np.concatenate(targets_list)
    outputs_ori = np.concatenate(outputs_ori_list)
    

    # calculate mAP
    mAP_ori = function_mAP(labels, outputs_ori)
    
    
    print("Calculating mAP:")  
    logger.info("  mAP_ori: {}".format(mAP_ori))
    

    return mAP_ori, 0 , 0

import matplotlib.pyplot as plt
import numpy as np
import os

@torch.no_grad()
def update_weight(lb_train_dataset, model, args, logger):

    loader = torch.utils.data.DataLoader(
        lb_train_dataset, batch_size=128, shuffle=False,
        num_workers=args.workers, pin_memory=False)

    model.eval()
    import numpy as np

    outputs_ori_list = []
    targets_list = []


    for i, ((inputs_list, _), targets, index) in enumerate(loader):
        inputs_ori = inputs_list[0].cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)
        
        with torch.cuda.amp.autocast(enabled=args.amp):
            outputs_ori = model(inputs_ori)
        
        outputs_ori_sm = torch.sigmoid(outputs_ori)        

        outputs_ori_list.append(outputs_ori_sm.detach().cpu())        
        targets_list.append(targets.detach().cpu())


    labels = np.concatenate(targets_list)
    outputs_ori_np = np.concatenate(outputs_ori_list)
    mAP_ori = function_mAP(labels, outputs_ori_np)

    true_label_np = labels
    outputs_np = outputs_ori_np


    num_classes = args.n_classes
    num_bins = 20 

    bin_edges = np.linspace(0, 1, num_bins + 1)
    bin_centers = [(bin_edges[i] + bin_edges[i+1])/2 for i in range(num_bins)]


    all_preds = outputs_np.flatten()
    all_labels = true_label_np.flatten()
    
    global_pos_ratio = np.zeros(num_bins)  
    global_bin_counts = np.zeros(num_bins) 
    
    for i in range(num_bins):
        bin_mask = (all_preds >= bin_edges[i]) & (all_preds < bin_edges[i + 1])
        bin_labels = all_labels[bin_mask]
        global_bin_counts[i] = len(bin_labels)
        if len(bin_labels) > 0:
            global_pos_ratio[i] = np.mean(bin_labels)

    if args.dataset_name == 'voc':
        global_pos_ratio_smooth = smooth_for_voc(global_pos_ratio, bin_centers, num_bins)
    else:
        global_pos_ratio_smooth = smooth_for_large_dataset(global_pos_ratio, bin_centers, num_bins)

    global_weight_matrix = global_pos_ratio_smooth
    global_weight_matrix_tensor = torch.tensor(global_weight_matrix).float().cuda()
    
    return {
        'global_weight_matrix': global_weight_matrix_tensor,
        'bin_edges': torch.tensor(bin_edges).float().cuda()
    }


# Use the predictions of all labeled data to obtain dynamic dual thresholds
@torch.no_grad()
def dynamic_threshold_generate(lb_train_dataset, model, threshold0, threshold1, args):
    model.eval()
    loader = torch.utils.data.DataLoader(
        lb_train_dataset, batch_size=128, shuffle=False,
        num_workers=args.workers, pin_memory=False)
    outputs_new = torch.ones(1, args.n_classes).cuda()
    targets_new = torch.ones(1, args.n_classes).cuda()

    for i, ((inputs_w_list, inputs_s_list), targets, index) in enumerate(loader):

        inputs_ori = inputs_w_list[0].cuda(non_blocking=True)
        inputs_s_ori = inputs_s_list[0].cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)

        with torch.cuda.amp.autocast(enabled=args.amp):
            outputs = model(inputs_ori)
        outputs = torch.sigmoid(outputs)
        outputs_new = torch.cat((outputs_new, outputs), dim = 0)
        targets_new = torch.cat((targets_new, targets), dim = 0)

    outputs_l = outputs_new[1:, :]
    targets_l = targets_new[1:]

    targets_t = ~(targets_l.bool()) + 0
    pred0 = outputs_l * targets_l
    pred1 = outputs_l * targets_t

    for i in range(args.n_classes):
        outputs0_new = pred0[:, i][torch.nonzero(pred0[:, i])].view(-1) # Presence
        outputs1_new = pred1[:, i][torch.nonzero(pred1[:, i])].view(-1) # Absence  
        threshold1[i] = 0.5 * (outputs0_new.min() + outputs0_new.max()) 
        threshold0[i] = 0.5 * (outputs1_new.min() + outputs1_new.max()) 

    return threshold0, threshold1

# Select pseudo-labels
def unsup_generate(threshold0, threshold1, outputs_u_w, targets_u): 
    mask = []
    dim0, dim1 = targets_u.shape
    for m in range(dim0):
        for n in range(dim1):
            if  targets_u[m, n] == 1:   #Pseudo-label indicating the presence of an attribute
                mask.append(outputs_u_w[m, n].ge(threshold0[n]).float())
            else:    
                mask.append(outputs_u_w[m, n].lt(threshold1[n]).float())
    mask=torch.as_tensor(mask).reshape(dim0, dim1)
    return mask.cuda()

if __name__ == '__main__':
    main()