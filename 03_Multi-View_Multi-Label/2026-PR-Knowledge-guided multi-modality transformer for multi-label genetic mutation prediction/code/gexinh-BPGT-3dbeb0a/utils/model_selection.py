"""Model and module selection for paper-aligned BPGT experiments."""


PAPER_BACKBONES = [
    "genehe-ve",
    "deephis-ve",
    "att-ve",
    "trans-ve",
    "kat",
    "hipt",
]

PAPER_GRAPH_MODULES = ["mlp", "gcn", "graph_transformer", "mcat", "bpgt"]


def normalize_backbone_name(name):
    return (name or "trans-ve").lower().replace("_", "-")

def model_selection(args):
    # transfer multi-label classification
    args.model = normalize_backbone_name(args.model)
    args.requires_coords = False

    if args.model in ['deephis', 'deephis-ve']:
        args.agg_method = 'dense'
        args.model_size = 'dino'
        args.pretrained_model = 'dino'

    elif args.model in ['geneh', 'genehe', 'genehe-ve', 'geneh-ve']:
        args.agg_method = 'self_att'
        args.model_size = 'dino'
        args.pretrained_model = 'dino'

    elif args.model in ['att-mil', 'att_mil', 'abmil', 'att-ve']:
        args.agg_method = 'abmil'
        args.model_size = 'dino'
        args.pretrained_model = 'dino'
        
    elif args.model == 'hipt':
        args.agg_method = 'hipt'
        args.model_size = 'hipt_att'
        args.pretrained_model = 'hipt'

    elif args.model in ['trans-mil', 'trans_mil', 'trans-ve']:
        args.agg_method = 'trans_mil'
        args.model_size = 'dino'
        args.pretrained_model = 'dino'
        args.region_cls_model = 'corr'

    elif args.model == 'kat':
        args.agg_method = 'kat'
        args.model_size = 'dino'
        args.pretrained_model = 'dino'
        args.requires_coords = True

    else:
        raise ValueError(f'Unsupported visual extractor backbone: {args.model}')

    return args

def module_selection(args, model_dict):
    graph_module = 'bpgt' if args.graph_module in ['bpgt', 'geneformer', 'geneformer_v2', 'graphormer'] else args.graph_module
    args.graph_module = graph_module

    if graph_module == 'bpgt':
        """ construct the graph 
        """
        from .graph_utils import mutual_info_encoding, organ_positional_encoding
        pe = organ_positional_encoding(args, df=args.df) 
        mask = mutual_info_encoding(args, df=args.df, if_cls_token= args.cls_token)        
        model_dict["pe"]=pe
        model_dict["spatial_mask"] = mask

    elif graph_module == 'gcn':
        from .graph_utils import mutual_info_encoding, organ_positional_encoding
        pe = organ_positional_encoding(args, df=args.df) 
        mask = mutual_info_encoding(args, df=args.df, if_cls_token= args.cls_token)        
        model_dict["pe"]=pe
        model_dict["spatial_mask"] = mask

        model_dict["adj"] = args.adj + pe + mask

    elif graph_module == 'graph_transformer':
        from .graph_utils import mutual_info_encoding, organ_positional_encoding
        pe = organ_positional_encoding(args, df=args.df) 
        mask = mutual_info_encoding(args, df=args.df, if_cls_token= args.cls_token)        
        model_dict["pe"]=pe
        model_dict["spatial_mask"] = mask

        model_dict["adj"] = args.adj + mask + pe 

    elif graph_module == 'mlp':
        pass
    elif graph_module == 'mcat':
        pass

    return model_dict
