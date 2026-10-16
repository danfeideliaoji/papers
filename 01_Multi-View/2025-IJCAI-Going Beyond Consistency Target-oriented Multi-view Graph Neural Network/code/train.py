
import warnings
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from args import parameter_parser
from utils import *
from Dataloader import load_multi_data
from model import TGNN
import copy
import configparser
import scipy
import random

if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    args = parameter_parser()
    if args.fix_seed:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)

    multi_view_dataset = ['MNIST10k', 'esp-game']

    for index, item in enumerate(multi_view_dataset):
        args.device = 'cuda:0'
        args.dataset = item
        print('--------------Multi-view Datasets: {}--------------------'.format(args.dataset))
        conf = configparser.ConfigParser()
        config_path = './config_demo' + '.ini'
        conf.read(config_path)
        args.num_epoch = int(conf.getfloat(args.dataset, 'epoch'))
        args.lr = conf.getfloat(args.dataset, 'lr')
        args.weight_decay = conf.getfloat(args.dataset, 'weight_decay')
        args.dropout = conf.getfloat(args.dataset, 'dropout')
        args.alpha = conf.getfloat(args.dataset, 'alpha')
        args.train_ratio = conf.getfloat(args.dataset, 'train_ratio')
        args.bias = conf.getfloat(args.dataset, 'bias')
        adj_list, feature, labels, train_mask, valid_mask, test_mask = load_multi_data(args)

        n = adj_list[0].shape[0]
        num_classes = len(np.unique(labels))
        num_view = len(adj_list)
        labels = labels.to(args.device)

        all_ACC = []
        all_F1_ma = []
        all_F1_mi = []

        for n_num in range(args.n_repeated):
            input_dims = []
            best_valid_acc = 0
            for i in range(num_view):
                input_dims.append(feature[i].shape[1])

            model = TGNN(args, num_view, n, input_dims, num_classes, args.dropout, args.hdim, args.device, adj_list).to(args.device)
            loss_function1 = torch.nn.NLLLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

            with tqdm(total=args.num_epoch, desc="Training") as pbar:
                for epoch in range(args.num_epoch):
                    model.train()
                    Loss_other, Output = model(feature, adj_list, labels[train_mask], train_mask)
                    invariant_output_softmax = F.log_softmax(Output, dim=1)
                    L_cro = loss_function1(invariant_output_softmax[train_mask], labels[train_mask])
                    Loss = Loss_other + L_cro
                    optimizer.zero_grad()
                    Loss.backward()
                    optimizer.step()

                    pred_labels = torch.argmax(Output, 1).data.cpu().numpy()
                    Train_ACC, _, _, F1_ma, F1_mi = get_evaluation_results(
                        labels.cpu().detach().numpy()[train_mask],
                        pred_labels[train_mask])

                    with torch.no_grad():
                        model.eval()
                        _, Output = model(feature, adj_list, labels[valid_mask], valid_mask)
                        pred_labels = torch.argmax(Output, 1).cpu().detach().numpy()
                        Valid_ACC, _, _, F1_ma, F1_mi = get_evaluation_results(labels.cpu().detach().numpy()[valid_mask],
                                                               pred_labels[valid_mask])


                    if (Valid_ACC >= best_valid_acc):
                        best_valid_acc = Valid_ACC
                        best_model = copy.deepcopy(model)
                        best_epoch = epoch

                    pbar.set_postfix({
                        "Epoch": epoch + 1,
                        'Total_loss': '{:.4f}'.format(Loss.item()),
                        'Train_ACC': '{:.2f}'.format(Train_ACC * 100),
                        'Valid_ACC': '{:.2f}'.format(Valid_ACC * 100),
                    })
                    pbar.update(1)

                test_model = best_model
                with torch.no_grad():
                    test_model.eval()
                    _, Output = test_model(feature, adj_list, labels[test_mask], test_mask)
                    print("Evaluating the model")
                    pred_labels = torch.argmax(Output, 1).cpu().detach().numpy()
                    ACC, _, _, F1_ma, F1_mi = get_evaluation_results(labels.cpu().detach().numpy()[test_mask],
                                                                     pred_labels[test_mask])
                    print("ACC: {:.2f}, F1_ma: {:.2f}, F1_mi: {:.2f}".format(ACC * 100, F1_ma * 100, F1_mi * 100))

                all_ACC.append(ACC)
                all_F1_ma.append(F1_ma)
                all_F1_mi.append(F1_mi)
                print("-----------------------")
                print("ACC: {:.2f} ({:.2f})".format(np.mean(all_ACC) * 100, np.std(all_ACC) * 100))
                print(
                    "F1_macro : {:.2f} ({:.2f})".format(np.mean(all_F1_ma) * 100, np.std(all_F1_ma) * 100))
                print(
                    "F1_micro : {:.2f} ({:.2f})".format(np.mean(all_F1_mi) * 100, np.std(all_F1_mi) * 100))
                print("-----------------------")
