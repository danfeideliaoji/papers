import argparse
from datasets.dataloader import *
from utils.util import *
from models.DSCN import DSCN


def get_args(dataset_name="corel5k"):
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-name', type=str, default="corel5k")
    parser.add_argument('--training-ratio', type=float, default=0.7)
    parser.add_argument('--missing-rate-data', type=float, default=0.5)
    parser.add_argument('--missing-rate-label', type=float, default=0.5)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--fold-idx', type=int, default=0)
    parser.add_argument('--ins-temperature', type=float, default=0.5)
    parser.add_argument('--label-temperature', type=float, default=1.0)
    parser.add_argument('--alpha', type=float, default=1e-2)
    parser.add_argument('--beta', type=float, default=5e-3)
    parser.add_argument('--threshold', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--hidden', type=int, default=512)
    parser.add_argument('--mask-rate', type=float, default=0.3)
    parser.add_argument('--mid-dim', type=int, default=1024)
    parser.add_argument('--layer-num', type=int, default=3)
    parser.add_argument('--weight-decay', type=float, default=1.0)
    parser.add_argument('--run-times', type=int, default=10)
    parser.add_argument('--early-stop', type=bool, default=False)

    args = parser.parse_args()
    args.dataset_name = dataset_name
    if dataset_name == "corel5k":
        args.hidden = 768
        args.alpha = 3e-2
        args.beta = 5e-3
        args.mask_rate = 0.5

    elif dataset_name == "pascal07":
        args.layer_num = 2
        args.alpha = 1e-1
        args.beta = 5e-3
        args.ins_temperature = 0.1

    elif dataset_name == "espgame":
        args.alpha = 3e-2
        args.beta = 5e-3
        
    elif dataset_name == "iaprtc12":
        args.alpha = 1e-2
        args.beta = 1e-3

    elif dataset_name == "mirflickr":
        args.alpha = 1e-2
        args.beta = 5e-3

    return args


if __name__ == '__main__':

    for t in ["corel5k"]:
        args = get_args(t)
        # if code is run with view missing rate of 50% and label missing rate of 50%, the epoch set in get_args() is enough.
        # When the missing rate is lower, you need to employ an early stopping strategy because it requires more epochs to converge.
        # Uncommenting the following two lines of code will enable the early stopping mechanism.
        # args.early_stop = True
        # args.epochs = 1000
        final_res = []
        for idx in range(args.run_times):
            setup_seed(args.seed)
            args.fold_idx = idx

            print("Loading {} dataset.....".format(t))
            train_dataset = MatDataset(args, mode="train")
            test_dataset = MatDataset(args, mode="test")
            val_dataset = MatDataset(args, mode="val")
            print("Loading {} dataset over!!!".format(t))

            args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            args.view_num = train_dataset.view_num
            args.class_num = train_dataset.class_num
            args.total_samples = train_dataset.total_samples
            args.input_dims = []
            for i in range(args.view_num):
                args.input_dims.append(train_dataset.data[i].shape[1])

            logger = get_logger(args, main_dir='../results/logs/')
            logger.info(f"--------------------Training {args.fold_idx + 1} time--------------------")

            output_args(logger, args)

            model = DSCN(args, logger).to(args.device)
            res = model.run_train(train_dataset, val_dataset, test_dataset)

            final_res.append(res)

            logger.handlers.clear()

        logger = get_logger(args, main_dir='../results/records/')
        logger.info("--------------------Training over--------------------")

        output_args(logger, args)
        calc_res(logger, final_res)

        logger.handlers.clear()
