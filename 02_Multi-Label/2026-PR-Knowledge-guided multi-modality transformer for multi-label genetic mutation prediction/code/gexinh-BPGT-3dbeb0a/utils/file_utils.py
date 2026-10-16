import pickle
import pandas as pd
import os
import argparse

try:
    import h5py
except ImportError:
    h5py = None


def _require_h5py():
    if h5py is None:
        raise ImportError(
            "h5py is required for HDF5 read/write utilities. Install h5py before "
            "running WSI preprocessing or feature-cache export code."
        )

def save_pkl(filename, save_object):
	writer = open(filename,'wb')
	pickle.dump(save_object, writer)
	writer.close()

def load_pkl(filename):
	loader = open(filename,'rb')
	file = pickle.load(loader)
	loader.close()
	return file


allpath=[]
allname=[]
def getallfile(path):

	allfilelist=os.listdir(path)
    # 遍历该文件夹下的所有目录或者文件
	for file in allfilelist:
		filepath=os.path.join(path,file)
		# 如果是文件夹，递归调用函数
		if os.path.isdir(filepath):
			getallfile(filepath)
		# 如果不是文件夹，保存文件路径及文件名
		elif os.path.isfile(filepath):
			allpath.append(filepath)
			allname.append(file)
	return allpath, allname

def get_img_path(data_flag, source_path, region_list):
    slides = []
    full_path = []
    for i in region_list:
        if data_flag == 'cptac':				
            cur_source = os.path.join(source_path, i, i)
            # slides.extend(sorted(os.listdir(cur_source)))
            cur_slides = sorted(os.listdir(cur_source))	
            cur_slides = [slide for slide in cur_slides if os.path.isfile(os.path.join(cur_source, slide))]
            cur_path = [os.path.join(cur_source, slide) for slide in cur_slides if os.path.isfile(os.path.join(cur_source, slide))]
            slides.extend(cur_slides)
            full_path.extend(cur_path)

        elif data_flag == 'tcga':
            cur_source = os.path.join(source_path, i)
            files, names = getallfile(cur_source)
            cur_path = sorted(files)
            cur_path = [abs_path for abs_path in cur_path if abs_path[-4:] == '.svs' ]
            
            # should also get rid of abs pathes of slides that is not tumor
            cur_path = [pathes for pathes in cur_path if int(pathes.split('/',-1)[-1][13]) != 1]

            cur_slides = [slide.split('/',-1)[-1] for slide in cur_path]
            # pick up normal slide. e.g., barcode: TCGA-A7-A6VV-01Z, check the first number of '01Z' whehter 0[tumor] or not [normal] 
            cur_slides = [slide for slide in cur_slides if int(slide[13]) != 1]
            
            slides.extend(cur_slides)
            full_path.extend(cur_path)
    return full_path, slides


def save_hdf5(output_path, asset_dict, attr_dict= None, mode='a'):
    _require_h5py()
    file = h5py.File(output_path, mode)
    for key, val in asset_dict.items():
        data_shape = val.shape
        if key not in file:
            data_type = val.dtype
            chunk_shape = (1, ) + data_shape[1:]
            maxshape = (None, ) + data_shape[1:]
            dset = file.create_dataset(key, shape=data_shape, maxshape=maxshape, chunks=chunk_shape, dtype=data_type)
            dset[:] = val
            if attr_dict is not None:
                if key in attr_dict.keys():
                    for attr_key, attr_val in attr_dict[key].items():
                        dset.attrs[attr_key] = attr_val
        else:
            dset = file[key]
            dset.resize(len(dset) + data_shape[0], axis=0)
            dset[-data_shape[0]:] = val
    file.close()
    return output_path

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


def get_args_parser_for_dataset():
    parser = argparse.ArgumentParser(description='generate dataset-related csv and graph',add_help=False)

    parser.add_argument('--data', default='tcga', choices=['tcga', 'cptac', 'no'], help='choose the dataset')
    parser.add_argument('--label_space', default='react', choices=['react', 'oncokb', 'expand', 'final'], help='choose the genetic category of tcga dataset label')
    parser.add_argument('--s', default='pancan', choices=['pancan', 'persample'], help='choose the source of tcga dataset label')
    parser.add_argument('--atr', default='driven', choices=['freq', 'driven'], help='choose the attribution of tcga dataset')

    parser.add_argument('--inter', default=False, type=int, help='choose if construct label space with the intersection (mutual) set')
    parser.add_argument('--iter', default=True, type = int, help='whether iterately implement the script')

    parser.add_argument('--patch_size', default=256, type = int, help='choose the patch size of slides')
    parser.add_argument('--mag', type = int, default=-1, choices=[-1, 40, 20, 10, 5], 
                help='choose the magnification of slides')

    parser.add_argument('--specific_region', type = int, default=True,
            help='if process the specific region for tcga/cptac dataset')
    parser.add_argument('--mix_list', type=list, default=None, nargs='+', help='1234 -> [1,2,3,4]')
    parser.add_argument('--r', default=0, type=int, help='choose the specific region to construct label space')

    return parser


# TODO: 层级方法 把共有的部分提出来
def root_arg(data, label_source, label_tag, args):
    if data == 'cptac':
        args.list_path = './data/processed data/list'
        # args.save_path = './data/cptac'  
        args.data_root = './data/cptac'
        args.label_path = './data/selected SNV label/'
        args.label_path = './data/SNV label/'
        args.region_list = ['BR', 'UCEC', 'LUAD', 'LSCC', 'CCRCC', 'CO', 'OV', 'GBM', 'HNSCC',  'PDA']
        # args.r = None

        args.start_ind = 3

        if args.specific_region == True:
            args.region = args.region_list[args.r]
            args.data_root_dir = os.path.join(args.data_root, args.region_list[args.r])
        else:
            if args.mix_list is not None:
                if args.count == 0:
                    args.region = ''
                    for i in args.mix_list:
                        args.region = args.region + '_' + args.region_list[int(i)][5:] 
                    args.data_root_dir = args.data_root + '/region' + args.region
            else:
                args.data_root_dir = args.data_root + '/whole_dataset'
                args.region = 'total'       
        args.abs_path = '/data2/WSI_CPTAC'

    elif data == 'tcga':
        args.list_path = './data/processed data/list'
        args.abs_path = '/data2/WSI_TCGA'
        # args.save_path = './data/tcga'  
        args.data_root = './data/tcga'  

        if args.label_space == "react":
            args.label_path = './data/TCGA_mutation/'
            if label_source == 'pancan':
                args.label_path = './data/TCGA_mutation/pancan'
                source = 'pan'
            else:
                args.label_path = './data/TCGA_mutation/persample'
                source = 'per'

            if label_tag == 'freq':
                args.label_path = args.label_path + '/5_freq/TCGA_5%_mut_gene_mutation_'
                tag = 'f'
            else:
                args.label_path = args.label_path + '/cancer driver/TCGA_driver_gene_mutation_'
                tag = 'd'
            args.prefix = '/{}_{}_'.format(source, tag)

        elif args.label_space == "oncokb":
            args.label_path = './data/OncoKB label/PANCAN_TCGA_OncoKB_'
            args.prefix = '/oncokb_'
        
        elif args.label_space == "expand":
            args.label_path = './data/PANCAN_TCGA_CancerDriver_Expanded/PANCAN_TCGA_CancerDriver_Expanded_'
            args.prefix = '/expand_'
        elif args.label_space == 'final':
            args.prefix = '/'

        args.region_list = ['TCGA-BRCA', 'TCGA-UCEC', 'TCGA-LUAD','TCGA-LUSC', 'TCGA-KIRC', 'TCGA-COAD', 'TCGA-OV', 'TCGA-GBM', 'TCGA-HNSC', 'TCGA-PAAD',
        'TCGA-LGG', 'TCGA-PRAD', 'TCGA-THCA', 'TCGA-SKCM', 'TCGA-STAD', 'TCGA-BLCA', 
        'TCGA-LIHC', 'TCGA-CESC', 'TCGA-KIRP', 'TCGA-SARC', 'TCGA-ESCA', 'TCGA-PCPG',
        'TCGA-READ', 'TCGA-LAML', 'TCGA-TGCT', 'TCGA-THYM', 'TCGA-ACC', 'TCGA-MESO', 
        'TCGA-UVM', 'TCGA-KICH', 'TCGA-UCS', 'TCGA-DLBC', 'TCGA-CHOL']

        if args.specific_region == True:
            args.region = args.region_list[args.r][5:]
            args.data_root_dir = os.path.join(args.data_root, args.region_list[args.r][5:])
            
        else:
            if args.mix_list is not None:
                if args.count == 0:
                    args.region = ''
                    for i in args.mix_list:
                        args.region = args.region + '_' + args.region_list[int(i)][5:] 
                    args.data_root_dir = args.data_root + '/region' + args.region

            else:
                args.data_root_dir = args.data_root + '/whole_dataset'
                args.region = 'total'

        args.start_ind = 2

    return args



def param_selection_for_dataset(data, label_source, label_tag, args):
    """ used for generate dataset
    """
    args = root_arg(data, label_source, label_tag, args)       

    if args.count == 0:
        if args.mag != -1:
            if args.mag == 20:
                args.custom_downsample = 2
            elif args.mag == 10:
                args.custom_downsample = 4
            elif args.mag ==5:
                args.custom_downsample = 8
            elif args.mag ==40:
                args.custom_downsample = 1
            # ratio = 40/ args.mag since the max/default magnification is 40 
            args.patch_size = args.patch_size * args.custom_downsample
            
            args.add_path = '/mag{}x_patch{}_fp'.format(args.mag, int(args.patch_size/(40/args.mag)))

        else:
            args.add_path = '/mag{}x_patch{}_fp'.format(40, int(args.patch_size))

        args.data_root_dir = args.data_root_dir  + args.add_path
        args.save_path = args.data_root_dir

    args.csv_1_path = args.save_path + args.prefix + 'Step_3.csv' if data == 'tcga' else args.data_root_dir + '/Step_3.csv'
    args.csv_2_path = args.save_path + args.prefix + 'Step_3_.csv' if data == 'tcga' else args.data_root_dir + '/Step_3_.csv'

    args.count = args.count + 1

    return args


def param_selection_for_pretrain(data, label_source, label_tag, args):
    """ 
    """
    args = root_arg(data, label_source, label_tag, args)     
    if args.data == 'cptac':             
        
        args.abs_path = '/data2/WSI_CPTAC'

    elif args.data == 'tcga':
        
        args.abs_path = '/data2/WSI_TCGA'


    #TODO: add checkpoint path
    # args.ckpt_path = os.path.join(args.output_dir, 'checkpoint0020.pth')
    # args.ckpt_path = os.path.join(args.output_dir, 'vits_tcga_brca_dino.pt')

    args.feat_dir = 'pretrained_{:s}_feature'.format(args.pretrained_model)

    if args.mag != -1:
        # ratio = 40/ args.mag since the max/default magnification is 40 
        args.add_path = '/mag{}x_patch{}_fp'.format(args.mag, int(args.ori_patch_size))
    else:
        args.mag = 40
        args.add_path = '/mag{}x_patch{}_fp'.format(args.mag, int(args.ori_patch_size))


    args.split_dir = args.split_dir = os.path.join(args.data_root_dir + args.add_path + '/splits', args.data+'_{}'.format(int(args.label_frac*100)))
    args.data_h5_dir = args.data_root_dir + args.add_path
    args.feat_dir = os.path.join(args.data_h5_dir, args.feat_dir)

    args.csv_path = args.data_root_dir + args.add_path +args.prefix + 'Step_3.csv' if args.data == 'tcga' else args.data_root_dir + args.add_path +'/Step_3.csv'
    args.dir_path_df = pd.read_csv(os.path.join(args.data_root_dir + args.add_path + '/Step_2.csv')) 

    return args


def param_selection(args):
    """ used for train downstream model
    """
    args.task = args.data
    arg_key = []
    for i in args._get_kwargs():
        arg_key.append(i[0])

    # add magnification
    if args.mag != -1:
        # ratio = 40/ args.mag since the max/default magnification is 40 
        args.add_path = '/mag{}x_patch{}_fp'.format(args.mag, int(args.patch_size))
    else:
        args.mag = 40
        args.add_path = '/mag{}x_patch{}_fp'.format(args.mag, int(args.patch_size))
    
    # Choose downstream model
    if 'downstream_model' in arg_key:
        if args.downstream_model == 'geneh':
            args.k_sample = 500
            args.pretrained_model = 'resnet'
            # args.max_epochs = 400
            args.max_epochs = 200
            args.model_size = '1024_small'
            args.loss = 'ce'
        elif args.downstream_model == 'deephis':
            args.pretrained_model = 'resnet'
            args.model_size = '1024_small'
            args.loss = 'ce'
            args.max_epochs = 200
        elif args.downstream_model == 'cor_mil_binary':
            args.pretrained_model = 'dino'
            args.model_size = 'dino'
            args.loss = 'ce'
            args.max_epochs = 200
        
        elif args.downstream_model == 'ml_decoder':
            # args.pretrained_model = 'dino'
            # args.model_size = 'dino'
            args.model_type = 'att_ml'
        
        elif args.downstream_model == 'exp_pipe':
            from utils.model_selection import model_selection
            args = model_selection(args)

    # Choose pretrained model
    if 'pretrained_model' in arg_key:
        if args.pretrained_model == 'dino': 
            args.feat_file = 'pretrained_dino_feature'

        elif args.pretrained_model == 'resnet':
            args.feat_file = 'pretrained_resnet_feature'
            args.model_size = '1024_min'
        
        elif args.pretrained_model == 'hipt':
            args.feat_file = 'pretrained_hipt_feature'
            args.model_size = 'hipt'
            # args.patch_size = 4096
    else:
        args.feat_file = 'pretrained_dino_feature'

        # Old version
        # args.data_feature_dir = './data/cptac/pretrained_dino'
        # args.data_feature_dir = '/home/huanggexin/data/tcga/pretrained_dino_tcga'
        # args.data_feature_dir = '/home/huanggexin/data/tcga/pretrained_resnet'


            

    if args.data == 'cptac':
        args.data_root = './data/cptac'
        args.label_path = './data/selected SNV label/'
        
        args.region_list = ['BR', 'UCEC', 'LUAD', 'LSCC', 'CCRCC', 'CO', 'OV', 'GBM', 'HNSCC',  'PDA']
        if args.specific_region == True:
            args.region = args.region_list[args.region_ind]
            args.data_root_dir = os.path.join(args.data_root, args.region_list[args.region_ind])
        else:
            if args.mix_list is not None:
                args.region = ''
                for i in args.mix_list:
                    args.region = args.region + '_' + args.region_list[int(i)]
                args.data_root_dir = args.data_root + '/region' + args.region
            else:
                args.data_root_dir = args.data_root + '/whole_dataset'
                args.region = 'total'

        #TODO: old parameter, should modify. 2022/8/14
        # args.region_ind = None
        # args.region = 'total'
        args.csv_path = args.data_root_dir + args.add_path + '/Step_3_.csv'
        args.start_ind = 3


    elif args.data == 'tcga':
        
        args.data_root = './data/tcga'

        args.region_list = ['TCGA-BRCA', 'TCGA-UCEC', 'TCGA-LUAD','TCGA-LUSC', 'TCGA-KIRC', 'TCGA-COAD', 'TCGA-OV', 'TCGA-GBM', 'TCGA-HNSC', 'TCGA-PAAD',
        'TCGA-LGG', 'TCGA-PRAD', 'TCGA-THCA', 'TCGA-SKCM', 'TCGA-STAD', 'TCGA-BLCA', 
        'TCGA-LIHC', 'TCGA-CESC', 'TCGA-KIRP', 'TCGA-SARC', 'TCGA-ESCA', 'TCGA-PCPG',
        'TCGA-READ', 'TCGA-LAML', 'TCGA-TGCT', 'TCGA-THYM', 'TCGA-ACC', 'TCGA-MESO', 
        'TCGA-UVM', 'TCGA-KICH', 'TCGA-UCS', 'TCGA-DLBC', 'TCGA-CHOL']

        if args.specific_region == True:
            args.region = args.region_list[args.region_ind][5:]
            args.data_root_dir = os.path.join(args.data_root, args.region_list[args.region_ind][5:])
        else:
            if args.mix_list is not None:
                args.region = ''
                for i in args.mix_list:
                    args.region = args.region + '_' + args.region_list[int(i)][5:] 
                args.data_root_dir = args.data_root + '/region' + args.region
            else:
                args.data_root_dir = args.data_root + '/whole_dataset'
                args.region = 'total'

        args.start_ind = 2
        

        if 'label_category' in arg_key:
            if args.label_category == 'gene':
                args.label_path = './data/TCGA_mutation/pancan' if args.s == 'pancan' else './data/TCGA_mutation/persample'
                source = 'pan' if args.s == 'pancan' else 'per'

                args.label_path = args.label_path + '/5_freq/TCGA_5%_mut_gene_mutation_' if args.atr == 'freq' else '/cancer driver/TCGA_driver_gene_mutation_'
                tag = 'f' if args.atr == 'freq' else 'd'

                args.csv_path = args.data_root_dir + args.add_path + '/{}_{}_Step_3_.csv'.format(source, tag)
                args.csv_1_path = args.data_root_dir + args.add_path + '/{}_{}_Step_3.csv'.format(source, tag)
           
            elif args.label_category == 'oncokb':
                args.label_path = './data/OncoKB label/PANCAN_TCGA_OncoKB_'

                args.csv_path = args.data_root_dir + args.add_path + '/Step_3_.csv'
                args.csv_1_path = args.data_root_dir + args.add_path + '/Step_3.csv'


            elif args.label_category == 'subtype':            
                args.label_path = './data/label/tcga_subtyping/'
                args.csv_path = args.data_root_dir + args.add_path + '/subtype_Step_3_.csv'
                args.csv_1_path = args.data_root_dir + args.add_path + '/subtype_Step_3.csv'

                args.label_ind = 0
    
    if not os.path.exists(args.csv_path) and hasattr(args, 'csv_1_path') and os.path.exists(args.csv_1_path):
        args.csv_path = args.csv_1_path

    args.data_feature_dir = os.path.join(args.data_root_dir + args.add_path,  args.feat_file)

    f = pd.read_csv(args.csv_path)
    gene_list = f.columns.values.tolist()[args.start_ind:]
  
    # args.exp_name = 'model_{}_gene_{}_data_{}_label_{}{}'.format(args.model_type, gene, args.data,source,tag)   # model_N_gene_i_data_tcga_label_panf
    if 'model_type' in arg_key:
        if args.model_type == 'att_ml':
            if args.downstream_model == 'mlp':
                args.exp_name = 'type_{}_model_{}_data_{}_region_{}'.format(args.model_type, args.downstream_model, args.data, args.region)
            elif args.downstream_model == 'ml_decoder':
                args.exp_name = 'type_{}_model_{}_graph_{}_data_{}_region_{}'.format(args.model_type, args.downstream_model, args.graph_module, args.data,args.region)
            elif args.downstream_model == 'exp_pipe':
                args.exp_name = 'model_{}/graph_{}_data_{}_region_{}'.format(args.model, args.graph_module, args.data, args.region)

        else:
            gene = gene_list[args.label_ind]
            if args.label_category != 'subtype':
                args.exp_name = 'model_{}_gene_{}_data_{}_region_{}_label_{}{}'.format(args.downstream_model, gene, args.data,args.region,source,tag)   # model_N_gene_i_data_tcga_label_panf
            else:
                args.exp_name = 'model_{}_task_{}_data_{}_region_{}'.format(args.downstream_model, gene, args.data,args.region)
    
    args.results_dir = './experiments/' + args.exp_name # /experiments/exp_name/
    

    return gene_list, args
