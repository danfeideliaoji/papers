""" A pipe line for HIPT 

    python -m torch.distributed.launch --nproc_per_node=8 [][]_extract_feature_hipt.py
"""


""" import packages
"""

""" In order to use another method as a package 
    reference: https://www.jianshu.com/p/696a601efd54
"""
import sys
# add the path of detectron to your Projects
sys.path.append('./model/HIPT_4K')

# WSI relavent
import numpy as np
import os

# for DDP mode
os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1, 2, 3'
# os.environ['CUDA_VISIBLE_DEVICES'] = '4, 5, 6, 7'
os.environ["OMP_NUM_THREADS"] = "6"  # 设置OpenMP计算库的线程数
os.environ["MKL_NUM_THREADS"] = "6"  # 设置MKL-DNN CPU加速库的线程数。
import torch
torch.set_num_threads(6)

import argparse
import datetime
import json
import math
import sys
import time
from pathlib import Path

import h5py

import openslide
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,3,4,5,6,7'
# os.environ["RANK"] = '0,1,2,3,4,5,6,7'
# os.environ["RANK"] = '0'
# os.environ['WORLD_SIZE'] = '1'
# os.environ['LOCAL_RANK'] = '0,1,2,3,4,5,6,7'
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from torchvision import models as torchvision_models
from torchvision import transforms

import model.vision_transformer as vits
import utils.utils_dino as utils
from datasets.dataset_h5 import (Generic_WSI_Patch, Whole_Slide_Bag,
                                 Whole_Slide_Bag_FP)
from model.vision_transformer import DINOHead, vit_small
from utils.file_utils import save_hdf5
from utils.utils import collate_features, print_network
from utils.utils_dino import DataAugmentationDINO, DINOLoss

# Dino
torchvision_archs = sorted(name for name in torchvision_models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(torchvision_models.__dict__[name]))

# TODO: 'i' is the index of split csv 
i = 0

""" Arg parser
"""
def get_args_parser():
    parser = argparse.ArgumentParser('DINO', add_help=False)

    # key parameters
    parser.add_argument('--train_model', default=False,type=int, help='if train the model or juse use the pretrained one')
    parser.add_argument('--epochs', default=100, type=int, help='Number of epochs of training.') # default =100
    parser.add_argument("--warmup_epochs", default=10, type=int,
        help="Number of epochs for the linear learning-rate warm up.") # default = 10
    
    parser.add_argument('--pretrained_weights', default='', type=str,
        help="Path to pretrained weights to load.")
    parser.add_argument("--checkpoint_key", default="teacher", type=str,
        help='Key to use in the checkpoint (example: "teacher")')

    parser.add_argument('--data', default='tcga', choices=['tcga', 'cptac'], help='choose the dataset')

    # WSI
    parser.add_argument('--data_h5_dir', type=str, default='./data/cptac')
    parser.add_argument('--data_root_dir', type=str, default='./data/cptac')
    parser.add_argument('--ori_patch_size', type=int, default=256, help='the original patch size of WSI')
    parser.add_argument('--split_dir', type=str, default=None, 
                    help='manually specify the set of splits to use, ' 
                    +'instead of infering from the task and label_frac argument (default: None)')
    parser.add_argument('--task', type=str, choices=['cptac',  'tpca'], default='cptac')
    parser.add_argument('--label_frac', type=float, default=1.0,
                    help='fraction of training labels (default: 1.0)')
    parser.add_argument('--slide_ext', type=str, default= '.svs')

    # extract features
    # parser.add_argument('--feat_dir', type=str, default='/home/huanggexin/data/cptac/pretrained_dino')
    parser.add_argument('--feat_dir', type=str, default='/home/huanggexin/data/cptac/pretrained_dino_20')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--no_auto_skip', default=False, action='store_true')
    parser.add_argument('--custom_downsample', type=int, default=1)
    parser.add_argument('--target_patch_size', type=int, default=-1)

    # Model parameters
    parser.add_argument('--arch', default='vit_small', type=str,
        choices=['vit_tiny', 'vit_small', 'vit_base', 'xcit', 'deit_tiny', 'deit_small'] \
                + torchvision_archs + torch.hub.list("facebookresearch/xcit:main"),
        help="""Name of architecture to train. For quick experiments with ViTs,
        we recommend using vit_tiny or vit_small.""")
    parser.add_argument('--patch_size', default=16, type=int, help="""Size in pixels
        of input square patches - default 16 (for 16x16 patches). Using smaller
        values leads to better performance but requires more memory. Applies only
        for ViTs (vit_tiny, vit_small and vit_base). If <16, we recommend disabling
        mixed precision training (--use_fp16 false) to avoid unstabilities.""")
    parser.add_argument('--out_dim', default=65536, type=int, help="""Dimensionality of
        the DINO head output. For complex and large datasets large values (like 65k) work well.""")
    parser.add_argument('--norm_last_layer', default=True, type=utils.bool_flag,
        help="""Whether or not to weight normalize the last layer of the DINO head.
        Not normalizing leads to better performance but can make the training unstable.
        In our experiments, we typically set this paramater to False with vit_small and True with vit_base.""")
    parser.add_argument('--momentum_teacher', default=0.996, type=float, help="""Base EMA
        parameter for teacher update. The value is increased to 1 during training with cosine schedule.
        We recommend setting a higher value with small batches: for example use 0.9995 with batch size of 256.""")
    parser.add_argument('--use_bn_in_head', default=False, type=utils.bool_flag,
        help="Whether to use batch normalizations in projection head (Default: False)")

    # Temperature teacher parameters
    parser.add_argument('--warmup_teacher_temp', default=0.04, type=float,
        help="""Initial value for the teacher temperature: 0.04 works well in most cases.
        Try decreasing it if the training loss does not decrease.""")
    parser.add_argument('--teacher_temp', default=0.04, type=float, help="""Final value (after linear warmup)
        of the teacher temperature. For most experiments, anything above 0.07 is unstable. We recommend
        starting with the default value of 0.04 and increase this slightly if needed.""")
    parser.add_argument('--warmup_teacher_temp_epochs', default=0, type=int,
        help='Number of warmup epochs for the teacher temperature (Default: 30).')

    # Training/Optimization parameters
    parser.add_argument('--use_fp16', type=utils.bool_flag, default=True, help="""Whether or not
        to use half precision for training. Improves training time and memory requirements,
        but can provoke instability and slight decay of performance. We recommend disabling
        mixed precision if the loss is unstable, if reducing the patch size or if training with bigger ViTs.""")
    parser.add_argument('--weight_decay', type=float, default=0.04, help="""Initial value of the
        weight decay. With ViT, a smaller value at the beginning of training works well.""")
    parser.add_argument('--weight_decay_end', type=float, default=0.4, help="""Final value of the
        weight decay. We use a cosine schedule for WD and using a larger decay by
        the end of training improves performance for ViTs.""")
    parser.add_argument('--clip_grad', type=float, default=3.0, help="""Maximal parameter
        gradient norm if using gradient clipping. Clipping with norm .3 ~ 1.0 can
        help optimization for larger ViT architectures. 0 for disabling.""")
    parser.add_argument('--batch_size_per_gpu', default=32, type=int,
        help='Per-GPU batch-size : number of distinct images loaded on one GPU.')
    
    parser.add_argument('--freeze_last_layer', default=1, type=int, help="""Number of epochs
        during which we keep the output layer fixed. Typically doing so during
        the first epoch helps training. Try increasing this value if the loss does not decrease.""")
    parser.add_argument("--lr", default=0.0005, type=float, help="""Learning rate at the end of
        linear warmup (highest LR used during training). The learning rate is linearly scaled
        with the batch size, and specified here for a reference batch size of 256.""")

    parser.add_argument('--min_lr', type=float, default=1e-6, help="""Target LR at the
        end of optimization. We use a cosine LR schedule with linear warmup.""")
    parser.add_argument('--optimizer', default='adamw', type=str,
        choices=['adamw', 'sgd', 'lars'], help="""Type of optimizer. We recommend using adamw with ViTs.""")
    parser.add_argument('--drop_path_rate', type=float, default=0.1, help="stochastic depth rate")

    # Multi-crop parameters
    parser.add_argument('--global_crops_scale', type=float, nargs='+', default=(0.4, 1.),
        help="""Scale range of the cropped image before resizing, relatively to the origin image.
        Used for large global view cropping. When disabling multi-crop (--local_crops_number 0), we
        recommand using a wider range of scale ("--global_crops_scale 0.14 1." for example)""")
    parser.add_argument('--local_crops_number', type=int, default=8, help="""Number of small
        local views to generate. Set this parameter to 0 to disable multi-crop training.
        When disabling multi-crop we recommend to use "--global_crops_scale 0.14 1." """)
    parser.add_argument('--local_crops_scale', type=float, nargs='+', default=(0.05, 0.4),
        help="""Scale range of the cropped image before resizing, relatively to the origin image.
        Used for small local view cropping of multi-crop.""")

    # Misc
    # parser.add_argument('--data_path', default='/path/to/imagenet/train/', type=str,
    #     help='Please specify path to the ImageNet training data.')

    parser.add_argument('--output_dir', default="./checkpoints/hipt", type=str, help='Path to save logs and checkpoints.')
    parser.add_argument('--saveckp_freq', default=10, type=int, help='Save checkpoint every x epochs.')
    parser.add_argument('--seed', default=0, type=int, help='Random seed.')
    parser.add_argument('--num_workers', default=10, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument("--dist_url", default="env://", choices=["env://", "tcp://localhost:23456" ],type=str, help="""url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument("--local_rank", default=0, type=int, help="Please ignore and do not set this argument.")

    # for dataset
    parser.add_argument('--mag', type = int, default=-1, choices=[-1, 40, 20, 10, 5], 
					help='choose the magnification of slides')
    parser.add_argument('--specific_region', type = bool, default=True,
					help='if process the specific region for tcga/cptac dataset')
    parser.add_argument('--r', type = int, default=0,
					help='the index of specific region for tcga/cptac dataset')
    parser.add_argument('--s', default='pancan', choices=['pancan', 'persample'], help='choose the source of tcga dataset label')
    parser.add_argument('--atr', default='driven', choices=['freq', 'driven'], help='choose the attribution of tcga dataset')
    
    return parser


# load parser
parser = argparse.ArgumentParser('DINO', parents=[get_args_parser()])
args = parser.parse_args()
Path(args.output_dir).mkdir(parents=True, exist_ok=True)

# new version
if args.data == 'cptac':
    args.region_list = ['BR', 'UCEC', 'LUAD', 'LSCC', 'CCRCC', 'CO', 'OV', 'GBM', 'HNSCC',  'PDA']
    # args.region = region_list if not args.specific_region else [region_list[args.r]]

    args.data_root_dir = './data/cptac' if not args.specific_region else './data/cptac/' + args.region_list[args.r]
    args.data_h5_dir = './data/cptac' if not args.specific_region else './data/cptac/' + args.region_list[args.r]
    
    ckpt_path = os.path.join(args.output_dir, 'checkpoint0020.pth')

    args.feat_dir = 'pretrained_dino_feature'
    abs_path = '/data2/WSI_CPTAC'
    csv_path = args.data_root_dir + '/Step_3_.csv'

elif args.data == 'tcga':

    if args.s == 'pancan':
        source = 'pan'
    else:
        source = 'per'

    if args.atr == 'freq':
        tag = 'f'
    else:
        tag = 'd'
    
    args.region_list = ['BRCA', 'UCEC', 'LUAD','LUSC', 'KIRC', 'COAD', 'OV', 'GBM', 'HNSC', 'PAAD',
        'LGG', 'PRAD', 'THCA', 'SKCM', 'STAD', 'BLCA', 
        'LIHC', 'CESC', 'KIRP', 'SARC', 'ESCA', 'PCPG',
        'READ', 'LAML', 'TGCT', 'THYM', 'ACC', 'MESO', 
        'UVM', 'KICH', 'UCS', 'DLBC', 'CHOL']
    # args.region = region_list if not args.specific_region else [args.region_list[args.r]]

    args.data_root_dir = './data/tcga' if not args.specific_region else './data/tcga/' + args.region_list[args.r]
    args.data_h5_dir = './data/tcga' if not args.specific_region else './data/tcga/' + args.region_list[args.r]
    
    vit_4k_ckpt_path = os.path.join(args.output_dir, 'vit4k_xs_dino.pth')
    vit_256_ckpt_path = os.path.join(args.output_dir, 'vit256_small_dino.pth')

    args.feat_dir = 'pretrained_hipt_feature'


    abs_path = '/data2/WSI_TCGA'
    csv_path = args.data_root_dir + '/{}_{}_Step_3.csv'.format(source, tag)



if args.mag != -1:
    # ratio = 40/ args.mag since the max/default magnification is 40 
    add_path = '/mag{}x_patch{}_fp'.format(args.mag, int(args.ori_patch_size))
else:
    args.mag = 40
    add_path = '/mag{}x_patch{}_fp'.format(args.mag, int(args.ori_patch_size))


args.split_dir = args.split_dir = os.path.join(args.data_root_dir + add_path + '/splits', args.data+'_{}'.format(int(args.label_frac*100)))
args.data_h5_dir = args.data_h5_dir + add_path
args.feat_dir = os.path.join(args.data_h5_dir, args.feat_dir)

csv_path = args.data_root_dir + add_path +'/{}_{}_Step_3.csv'.format(source, tag) if args.data == 'tcga' else args.data_root_dir + add_path +'/Step_3.csv'
dir_path_df = pd.read_csv(os.path.join(args.data_root_dir + add_path + '/Step_2.csv')) 



from data.data_loader import (Generic_ML_Dataset, Generic_ML_Table,
                              Generic_Split)

total_dataset = Generic_ML_Table(csv_path = csv_path,
                            dataset=args.data,
                            shuffle = False, 
                            seed = args.seed,
                            region_list = args.region_list
                            )


def custom_downsample_per_slide(slide_mag, expect_mag, default_mag=40, if_patch=False):
    """ custom script for downsample WSI with various magnifications

        Args:
            if_patch: when patching slides, we need to 

            default_mag: the default magnification of WSI, data with '.svs' format is 40x

    """
    if if_patch:
        expect_ratio = int(default_mag/expect_mag)
        factual_ratio = int(default_mag/slide_mag)
        custom_downsample = factual_ratio if factual_ratio<=expect_ratio else expect_ratio

    else:
        expect_ratio = int(slide_mag/expect_mag)
        custom_downsample = 1 if expect_ratio <= 1 else expect_ratio
    return custom_downsample


def compute_w_loader(file_path, output_path, wsi, model,
 	batch_size = 8, verbose = 0, print_every=20, pretrained=False, 
	custom_downsample=1, target_patch_size=-1):
	"""
	args:
		file_path: directory of bag (.h5 file)
		output_path: directory to save computed features (.h5 file)
		model: pytorch model
		batch_size: batch_size for computing features in batches
		verbose: level of feedback
		pretrained: use weights pretrained on imagenet
		custom_downsample: custom defined downscale factor of image patches
		target_patch_size: custom defined, rescaled image size before embedding
	"""
	dataset = Whole_Slide_Bag_FP(file_path=file_path, wsi=wsi, pretrained=pretrained, 
		custom_downsample=custom_downsample, target_patch_size=target_patch_size)
	x, y = dataset[0]
	kwargs = {'num_workers': 4, 'pin_memory': True} if device.type == "cuda" else {}
	loader = DataLoader(dataset=dataset, batch_size=batch_size, **kwargs, collate_fn=collate_features)

	if verbose > 0:
		print('processing {}: total of {} batches'.format(file_path,len(loader)))

	mode = 'w'
	for count, (batch, coords) in enumerate(loader):
		with torch.no_grad():	
			if count % print_every == 0:
				print('batch {}/{}, {} files processed'.format(count, len(loader), count * batch_size))
			batch = batch.to(device, non_blocking=True)
			
			features = model.forward(batch)
			features = features.cpu().numpy()

			asset_dict = {'features': features, 'coords': coords}
			save_hdf5(output_path, asset_dict, attr_dict= None, mode=mode)
			mode = 'a'
	
	return output_path

""" eval pretrained feature for train, val, and test set.
"""
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# TODO: build model
from model.HIPT_4K.hipt_4k import HIPT_4K
model = HIPT_4K(model256_path=vit_256_ckpt_path, model4k_path=vit_4k_ckpt_path)


# model = vits.__dict__[args.arch](patch_size=args.patch_size, num_classes=0)
for p in model.parameters():
    p.requires_grad = False

# if torch.cuda.device_count() > 1:
#     	model = nn.DataParallel(model)
# utils.init_distributed_mode(args)
# model.to(device)
# model = nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
model.eval()
""" load weight
"""
# ckpt_path = os.path.join(assets_dir, enc_name+'.pt')
args.pretrained_weights = args.output_dir 


assert os.path.isfile(vit_4k_ckpt_path)
assert os.path.isfile(vit_256_ckpt_path)
# state_dict = torch.load(ckpt_path, map_location="cpu")
# if args.checkpoint_key is not None and args.checkpoint_key in state_dict:
#     print(f"Take key {args.checkpoint_key} in provided checkpoint dict")
#     # state_dict = torch.load(ckpt_path, map_location="cpu")['teacher']
#     state_dict = torch.load(ckpt_path, map_location="cpu")[args.checkpoint_key]
# state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()} # remove `module.` prefix
# state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()} # remove `backbone.` prefix induced by multicrop wrapper
# msg = model.load_state_dict(state_dict, strict=False)
# print('Pretrained weights found at {} and loaded with msg: {}'.format(args.pretrained_weights, msg))
# missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)


# create folders
os.makedirs(args.feat_dir, exist_ok=True)
os.makedirs(os.path.join(args.feat_dir, 'pt_files'), exist_ok=True)
os.makedirs(os.path.join(args.feat_dir, 'h5_files'), exist_ok=True)
dest_files = os.listdir(os.path.join(args.feat_dir, 'pt_files'))

# load data respectively

train_dataset, val_dataset, test_dataset = total_dataset.return_splits(split_bool_path='{}/splits_{}.csv'.format(args.split_dir, i))
data_set_ind = []


train_dataset_id = train_dataset.slide_data['slide_id'].values.tolist()
val_dataset_id = val_dataset.slide_data['slide_id'].values.tolist()
test_dataset_id = test_dataset.slide_data['slide_id'].values.tolist()

train_dataset_id = list(set(train_dataset_id))
val_dataset_id = list(set(val_dataset_id))
test_dataset_id = list(set(test_dataset_id))

print('The number of Train, Val, and Test set is {}, {}, and {}'.format(len(train_dataset_id),len(val_dataset_id),len(test_dataset_id)))

data_set_ind = [train_dataset_id, val_dataset_id, test_dataset_id]

for data in data_set_ind:
    total =len(data)

    for bag_candidate_idx in range(total):
        # confirm related url
        slide_id = data[bag_candidate_idx]
        h5_file_path = os.path.join(args.data_h5_dir,'patches','{}.h5'.format(slide_id))
        
        # modify this part for different sever
        #/data2/WSI_CPTAC/BR/BR/01BR001-0684a407-f446-486d-9160-b483cb.svs

        wsi_path = dir_path_df.loc[dir_path_df['slide_id']==slide_id,'full_path'].values.tolist()
        
        wsi_path = os.path.join(abs_path,  wsi_path[0].split('/',3)[-1])
        wsi = openslide.open_slide(wsi_path)

        # wsi = openslide.open_slide(wsi_path[0])
        bag_name = slide_id + '.h5'

        print('\nprogress: {}/{}'.format(bag_candidate_idx+1, total))
        print(slide_id)

        if not args.no_auto_skip and slide_id+'.pt' in dest_files:
            print('skipped {}'.format(slide_id))
            continue 

        output_path = os.path.join(args.feat_dir, 'h5_files', bag_name)

        #
        slide_mag = dir_path_df.loc[dir_path_df['slide_id']==slide_id,'mag']
        custom_downsample = custom_downsample_per_slide(slide_mag, args.mag)
        
        # load model 
        time_start = time.time()
        output_file_path = compute_w_loader(h5_file_path, output_path, wsi, 
        model = model, batch_size = args.batch_size, verbose = 1, print_every = 20, 
        custom_downsample=custom_downsample, target_patch_size=args.target_patch_size)

        time_elapsed = time.time() - time_start
        print('\ncomputing features for {} took {} s'.format(output_file_path, time_elapsed))
        file = h5py.File(output_file_path, "r")

        features = file['features'][:]
        print('features size: ', features.shape)
        print('coordinates size: ', file['coords'].shape)
        features = torch.from_numpy(features)
        bag_base, _ = os.path.splitext(bag_name)
        torch.save(features, os.path.join(args.feat_dir, 'pt_files', bag_base+'.pt'))


