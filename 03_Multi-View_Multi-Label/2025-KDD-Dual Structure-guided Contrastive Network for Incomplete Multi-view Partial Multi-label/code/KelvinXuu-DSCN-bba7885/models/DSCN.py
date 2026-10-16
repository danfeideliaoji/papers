from argparse import Namespace
import torch.optim
from utils.evaluation import do_metric
import copy
from utils.util import *
from datasets.dataloader import get_dataloader
from models.basemodel import *
from utils.loss import *
from torch.nn.functional import normalize
import time


class DSCN(nn.Module):
    def __init__(self, args: Namespace, logger):
        super(DSCN, self).__init__()
        self.label_adj = None
        self.sample_adj = None
        self.args = args
        self.logger = logger
        self.view_num = args.view_num
        self.class_num = args.class_num
        self.hidden = self.args.hidden
        self.encoder_dims = [[self.args.input_dims[i]] + [self.args.mid_dim]*self.args.layer_num + [self.args.hidden] for i in range(self.view_num)]
        self.encoders = [Encoder(self.encoder_dims[i], dropout=self.args.dropout) for i in range(self.view_num)]
        self.encoders = nn.ModuleList(self.encoders)
        self.instance_project = [InstanceProject(self.hidden) for _ in range(self.view_num)]
        self.instance_project = nn.ModuleList(self.instance_project)
        self.attentionFusion = AttentionFusionLayer(self.hidden, self.view_num)
        self.classify = ClassificationLayer(self.hidden, self.class_num)

        self.criterion_MSIC = InstanceLoss(self.args.ins_temperature)
        self.criterion_HMBC = WeightBCELoss()
        self.criterion_LCL = LabeContrastivelLoss(self.args.label_temperature)
        self.loss_MSIC = torch.tensor(0.0).to(self.args.device)
        self.loss_LCL = torch.tensor(0.0).to(self.args.device)
        self.loss_HMBC = torch.tensor(0.0).to(self.args.device)
        self.alpha = self.args.alpha
        self.alpha = self.args.beta
        self.threshold = self.args.threshold
        self.mask_rate = self.args.mask_rate

        self.optimizer = torch.optim.AdamW(self.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)

    def make_adj(self, train_dataset):
        train_mask_data = train_dataset.mask_data
        train_mask_label = train_dataset.mask_label
        train_datas = train_dataset.data
        train_true_labels = train_dataset.labels

        start = time.time()
        dists = []
        for i in range(self.view_num):
            data_com = train_datas[i][train_mask_data[:, i] == 1]
            dist = get_similarity(data_com)

            dists.append(dist)
        print("Make view similarity time:{:.2f}s".format(time.time() - start))

        start = time.time()
        sim_f = fusion_mask_similarity(dists, train_mask_data)

        print("Make fusion similarity time:{:.2f}s".format(time.time() - start))

        # label adj
        start = time.time()
        temp_label = torch.zeros((self.class_num, train_mask_data.shape[0]))  # (C,N)
        for i, row in enumerate(train_true_labels.T):
            nonzero_indices = (row != 0).nonzero().squeeze()
            if nonzero_indices.numel() == 0:
                temp_label[i] = torch.zeros(train_true_labels.shape[0])
            else:
                selected_values = sim_f.T[:, nonzero_indices]
                if selected_values.ndimension() == 1:
                    selected_values = selected_values.reshape(-1, 1)

                temp_label[i] = selected_values.max(dim=1, keepdim=True).values.T
        temp_label = train_true_labels.T + (1 - train_mask_label.T) * temp_label

        norms = torch.norm(temp_label, p=2, dim=1, keepdim=True)
        label_adj = torch.mm(temp_label, temp_label.T) / (norms * norms.T + 1e-8)
        label_adj = label_adj.to(self.args.device)
        sim_f = sim_f.to(self.args.device)

        print("Make label correlation time:{:.2f}s".format(time.time() - start))

        return sim_f, label_adj

    def run_train(self, train_dataset, val_dataset, test_dataset, mode="train"):
        print("run.....")
        best_result = 0
        best_model = self.state_dict()
        best_epoch = -1

        self.sample_adj, self.label_adj = self.make_adj(train_dataset)

        train_loader = get_dataloader(self.args, train_dataset, mode)
        
        best_model_dict = dict()
        for epoch in range(self.args.epochs):
            self.train()
            losses = AverageMeter()
            loss_MSIC = AverageMeter()
            loss_LCL = AverageMeter()
            loss_HMBC = AverageMeter()
            for batch_No, (train_data, true_label, mask_data, mask_label, indices) in enumerate(train_loader):
                self.loss_MSIC = torch.tensor(0.0).to(self.args.device)
                self.loss_LCL = torch.tensor(0.0).to(self.args.device)
                self.loss_HMBC = torch.tensor(0.0).to(self.args.device)
                self.loss_mask_label = torch.tensor(0.0).to(self.args.device)
                self.loss_mask_data = torch.tensor(0.0).to(self.args.device)
                true_label = true_label.to(self.args.device)
                mask_data = mask_data.to(self.args.device)
                mask_label = mask_label.to(self.args.device)

                sample_adj = self.sample_adj[indices[:, None], indices]

                z, z_pro, z_fusion, label_pred, label_pred_fusion = self(train_data, mask_data)
                
                # loss_MSIC
                for v in range(self.view_num):
                    for w in range(v + 1, self.view_num):

                        # v-th view mask
                        mask_v = torch.ones_like(mask_data[:, v])  # all one
                        ones_positions = torch.nonzero(mask_data[:, v] == 1).squeeze()  # position of 1
                        drop_count = int(self.args.mask_rate * len(ones_positions))  # mask num
                        drop_indices = torch.randperm(len(ones_positions))[:drop_count]  # mask_idx
                        mask_v[ones_positions[drop_indices]] = 0  # mask[idx]=0
                        mask_v = mask_v * mask_data[:, v]  # update mask

                        # w-th view mask
                        mask_w = torch.ones_like(mask_data[:, w])
                        ones_positions = torch.nonzero(mask_data[:, w] == 1).squeeze()
                        drop_count = int(self.args.mask_rate * len(ones_positions))
                        drop_indices = torch.randperm(len(ones_positions))[:drop_count]
                        mask_w[ones_positions[drop_indices]] = 0
                        mask_w = mask_w * mask_data[:, w]

                        self.loss_MSIC += self.criterion_MSIC(z_pro[v], z_pro[w], mask_v, mask_w, sample_adj)  

                self.loss_MSIC *= self.alpha

                # loss_HMBC (loss_CMC + loss_VMC)
                for i in range(self.view_num):
                    self.loss_HMBC += self.criterion_HMBC(true_label[mask_data[:, i].bool()],
                                                        label_pred[i][mask_data[:, i].bool()],
                                                        mask_label[mask_data[:, i].bool()])
                self.loss_HMBC += self.criterion_HMBC(true_label, label_pred_fusion, mask_label)
                self.loss_HMBC /= (self.view_num + 1)


                # loss_CLC
                label_sum = torch.zeros_like(label_pred_fusion)
                for v in range(self.view_num):
                    label_sum += label_pred[v] * mask_data[:, v][:, None]
                row_sum = torch.sum(mask_data, dim=1)
                row_sum[row_sum == 0] = 1
                label_sum /= row_sum.view(-1, 1)
                self.loss_LCL = self.criterion_LCL.forward(label_pred_fusion, label_sum, self.label_adj, threshold=self.threshold)
                self.loss_LCL *= self.alpha

                loss = self.loss_MSIC + self.loss_HMBC + self.loss_LCL
                loss_MSIC.update(self.loss_MSIC.item())
                loss_HMBC.update(self.loss_HMBC.item())
                loss_LCL.update(self.loss_LCL.item())
                losses.update(loss.item())

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            inform = (
                "[train] ==> epoch[{}/{}], loss_MSIC:{:.2f}, loss_HMBC:{:.2f}, loss_LCL:{:.2f}, loss:{:.2f}".format(
                    epoch + 1, self.args.epochs, loss_MSIC.avg, loss_HMBC.avg, loss_LCL.avg, losses.avg,))
            self.logger.info(inform)

            if (epoch + 1) % 1 == 0:

                evaluation_results, inform_val = self.test(val_dataset, epoch, mode="val")
                self.logger.info(inform_val)

                if 2 * evaluation_results[0] + evaluation_results[1] + evaluation_results[2] + evaluation_results[3] > best_result:
                    best_result = 2 * evaluation_results[0] + evaluation_results[1] + evaluation_results[2] + evaluation_results[3]
                    best_model = copy.deepcopy(self.state_dict())
                    best_epoch = epoch

            if self.args.early_stop and epoch - best_epoch >= 50:
                break
        
        self.load_state_dict(best_model)

        final_test_res, final_test_inform = self.test(test_dataset, best_epoch, mode="test")

        final_test_inform = "FINAL TEST RESULTS: " + final_test_inform
        self.logger.info(final_test_inform)

        return final_test_res

    def test(self, test_dataset, epoch, mode="test"):
        self.eval()
        with torch.no_grad():
            test_datas = test_dataset.data
            test_mask_data = test_dataset.mask_data.to(self.args.device)
            test_true_labels = test_dataset.labels.to(self.args.device)

            z, z_pro, z_fusion, label_pred, label_pred_fusion = self(test_datas, test_mask_data)

            evaluation_results = do_metric(label_pred_fusion.cpu().numpy(), test_true_labels.cpu().numpy())
            inform = (
                '[{}] ==> epoch[{}/{}]  AP:{:.4f}  1-HL:{:.4f}  1-RL:{:.4f}  AUC:{:.4f}  1-OE:{:.4f}  1-Cov:{:.4f}'.format(
                    mode, epoch + 1, self.args.epochs, evaluation_results[0], evaluation_results[1], evaluation_results[2],
                    evaluation_results[3], evaluation_results[4], evaluation_results[5]
                ))

        return evaluation_results, inform

    def forward(self, datas, mask_data):
        z = [None for _ in range(self.view_num)]
        z_pro = [None for _ in range(self.view_num)]
        label_pred = [None for _ in range(self.view_num)]

        for i in range(self.view_num):
            datas[i] = datas[i].to(self.args.device)
            z[i] = self.encoders[i](datas[i])
            z_pro[i] = normalize(self.instance_project[i](z[i]), dim=1)
            label_pred[i] = self.classify(z[i])

        z_fusion = self.attentionFusion(z, mask_data)

        label_pred_fusion = self.classify(z_fusion)

        return z, z_pro, z_fusion, label_pred, label_pred_fusion
