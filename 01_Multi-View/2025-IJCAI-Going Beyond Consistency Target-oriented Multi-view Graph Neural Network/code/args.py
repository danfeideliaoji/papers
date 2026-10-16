


import argparse

def parameter_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:1')
    parser.add_argument("--path", type=str, default="./datasets/", help="Path of datasets")
    parser.add_argument("--dataset", type=str, default="yelp_0", help="Name of datasets")
    parser.add_argument("--seed", type=int, default=12, help="Random seed for train-test split. Default is 12.")
    parser.add_argument("--shuffle_seed", type=int, default=42, help="Random seed for train-test split. Default is 42.")
    parser.add_argument("--fix_seed", action='store_true', default=True, help="xx")

    parser.add_argument("--early-stop", type=bool, default=True, help="If early stop")
    parser.add_argument("--patience", type=int, default=20, help="Patience for early stop")
    parser.add_argument("--verbose", type=int, default=1, help="Show training details.")

    parser.add_argument("--n_repeated", type=int, default=5, help="Number of repeated times. Default is 10.")

    parser.add_argument("--knns", type=int, default=20, help="k of KNN graph. default=20")
    parser.add_argument("--common_neighbors", type=int, default=2, help="Number of common neighbors (when using pruning strategy 2)")
    parser.add_argument("--pr1", action='store_true', default=True, help="Using prunning strategy 1 or not")
    parser.add_argument("--pr2", action='store_true', default=True, help="Using prunning strategy 2 or not")


    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=5e-4, help="Weight decay") #me
    parser.add_argument("--num_epoch", type=int, default=200, help="Number of training epochs. Default is 200.")

    parser.add_argument("--data-split-mode", type=str, default="Ratio", help="Data split mode: Number or Ratio")
    parser.add_argument("--train_ratio", type=int, default=0.1, help="Train data ratio. Default is 0.1.")

    parser.add_argument("--alpha", nargs='+', type=float, default = 0.01, help="default=0.001")
    parser.add_argument("--bias", nargs='+', type=float, default = 1.0, help="default=1")
    parser.add_argument("--hdim", nargs='+', type=int, default=[512], help="Number of hidden dimensions")
    parser.add_argument("--layers", type=int, default=2, help="number of layer.")
    parser.add_argument("--dropout", type=float, default=0.7, help="Dropout rate.")


    args = parser.parse_args()

    return args