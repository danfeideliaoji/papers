import os
import os.path as osp
import utils
from utils import AverageMeter
import argparse
import time
import torch
import numpy as np
import random
from data import *
from torch import nn
from torch.nn import functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, CosineAnnealingWarmRestarts, CosineAnnealingLR
import copy
from torch.utils.data import DataLoader, Subset
from model import *

from metrics import *


def set_random_seeds(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    torch.Generator().manual_seed(seed)


def BCE_loss(target_pre, sub_target, reduction='mean'):
    assert torch.sum(torch.isnan(torch.log(target_pre))).item() == 0
    assert torch.sum(torch.isnan(torch.log(1 - target_pre + 1e-5))).item() == 0

    res = torch.abs(sub_target.mul(torch.log(target_pre + 1e-5)) + (1 - sub_target).mul(torch.log(1 - target_pre + 1e-5)))

    if reduction == 'mean':
        return torch.mean(res)
    elif reduction == 'sum':
        return torch.sum(res)
    elif reduction == 'none':
        return res


def test(model, num_views, test_loader, device, ensemble_weight):
    model.eval()
    # num_correct, num_sample = 0, 0
    pred = []
    true_label = []
    for X, Y, indexes in test_loader:

        for v in range(num_views):
            X[v] = X[v].to(device)
        Y = Y.to(device)

        with torch.no_grad():
            hiddens, restore_X = model(X)
            commons = model.Extra_Commonality(hiddens)  # extract the commonality information
            comples = model.Extra_Complementarity(hiddens)  # extract the complementarity information
            fused_commonality = model.Fuse_Commonality(commons)
            fused_complementarities = model.Fuse_Commonality_Complementarity(fused_commonality, comples)

            compl_logits = model.Complementarity_Classify(fused_complementarities)

            ensemble = model.AdaboostComplementarity(compl_logits, Y, args.trd, trained_weight_cls=ensemble_weight)
            prediction = ensemble

            # Y_pre = torch.sigmoid(prediction)
            Y_pre = prediction
            true_label.append(Y.cpu().numpy())
            pred.append(Y_pre.cpu().numpy())

            # num_sample += Y.shape[0]
    pred = torch.tensor(np.concatenate(pred))
    print('Pred Shape:', pred.shape)
    true_label = np.concatenate(true_label)

    trd = 0.4
    pre_labels = np.array((torch.sigmoid(pred) > trd).cpu(), dtype=np.int64)

    outputs = pred.cpu().numpy()
    true_labels = torch.tensor(true_label).cpu().numpy()
    print('True Label Shape:', true_label.size)
    metrics_print, metrics = all_metrics(outputs, pre_labels, true_labels)

    print(metrics_print)
    return metrics


def main(args):
    dataset = Scene(normalize=True)

    print('X: ', dataset.X)
    print('y:', dataset.Y)

    print('DataSet:', dataset.data_name)
    num_samples = len(dataset)

    ratio = 0.8

    num_classes = dataset.num_classes
    num_views = dataset.num_views
    dims = dataset.dims

    HM_list = []
    AP_list = []
    OE_list = []
    RL_list = []
    Subset_list = []
    Coverage_list = []
    MaF1_list = []
    MiF1_list = []

    for fold in range(5):

        folds_path = 'scene_5folds.mat'
        folds = sio.loadmat(folds_path)
        train_index = np.squeeze(folds[str(fold) + '_train'], axis=0)
        test_index = np.squeeze(folds[str(fold) + '_test'], axis=0)

        train_loader = DataLoader(Subset(dataset, train_index), batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(Subset(dataset, test_index), batch_size=args.batch_size, shuffle=False)

        device = torch.device("cuda:" + str(args.device) if torch.cuda.is_available() else "cpu")

        model = Tester(num_views, dims, args.dim_hidden, args.dim_com, num_classes, device, n_rule=2)  # 3
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1.e-4)  # 1e-4



        model.to(device)
        model.train()
        criterion = nn.BCELoss()

        for epoch in range(1, args.epochs + 1):
            for X, Y, indexes in train_loader:
                loss = 0
                for v in range(num_views):
                    X[v] = X[v].to(device)
                    # print('Train_X Shape:', X[v].shape)

                # print('Train Y shape:', Y.shape)
                Y = Y.to(device)
                hiddens, restore_X = model(X)

                commons = model.Extra_Commonality(hiddens)  # extract the commonality information
                comples = model.Extra_Complementarity(hiddens)  # extract the complementarity information
                fused_commonality = model.Fuse_Commonality(commons)
                decompose_loss = model.DecomposeLoss(commons, comples, fused_commonality, args.tau)
                fused_complementarities = model.Fuse_Commonality_Complementarity(fused_commonality, comples)


                compl_logits = model.Complementarity_Classify(fused_complementarities)

                ensemble, ensemble_weight = model.AdaboostComplementarity(compl_logits, Y, args.trd)
                prediction = ensemble

                cls_loss = criterion(torch.sigmoid(prediction), Y.to(torch.float32))

                loss = cls_loss + args.alpha * decompose_loss
                optimizer.zero_grad()

                loss.backward()
                optimizer.step()

            if epoch % 10 == 0:  # 中间测试
                print(f'====> {epoch}')

                temp_result = test(model, num_views, test_loader, device, ensemble_weight)

        result = test(model, num_views, test_loader, device, ensemble_weight)

        HM_list.append(result[0])
        AP_list.append(result[1])
        OE_list.append(result[2])
        RL_list.append(result[3])
        Subset_list.append(result[4])
        Coverage_list.append(result[5])
        MaF1_list.append(result[6])
        MiF1_list.append(result[7])
    print('Mean HM:', np.mean(HM_list), '±', np.std(HM_list))
    print('Mean AP:', np.mean(AP_list), '±', np.std(AP_list))
    print('Mean OE:', np.mean(OE_list), '±', np.std(OE_list))
    print('Mean RL:', np.mean(RL_list), '±', np.std(RL_list))
    print('Mean Subset:', np.mean(Subset_list), '±', np.std(Subset_list))
    print('Mean Coverage:', np.mean(Coverage_list), '±', np.std(Coverage_list))
    print('Mean MaF1:', np.mean(MaF1_list), '±', np.std(MaF1_list))
    print('Mean MiF1:', np.mean(MiF1_list), '±', np.std(MiF1_list))


########################## Best Hyperparameter
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=64)  # 64
    parser.add_argument('--epochs', type=int, default=15)
    # parser.add_argument('--annealing_step', type=int, default=50)
    parser.add_argument('--lr', type=float, default=0.00003)  # 0.00003
    parser.add_argument('--dataset', type=str, default='emotions')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--dim_hidden', type=int, default=512)
    parser.add_argument('--dim_com', type=int, default=512)
    parser.add_argument('--alpha', type=float, default=0.001)  # 0.01
    parser.add_argument('--tau', type=float, default=1.)
    parser.add_argument('--trd', type=float, default=0.52)  # 0.55

    args = parser.parse_args()

    seed = 42
    set_random_seeds(seed)
    main(args)
