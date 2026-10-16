import torch
import torch.nn as nn


class InstanceLoss(nn.Module):
    def __init__(self, temperature):
        super(InstanceLoss, self).__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss(reduction="sum")

    def forward(self, z_i, z_j, mask_i, mask_j, adj):
        if torch.sum(mask_i) == 0 or torch.sum(mask_j) == 0:
            return 0
        adj = 1 - adj
        adj = adj.fill_diagonal_(1)
        adj = torch.cat((adj, adj), dim=1)
        adj = torch.cat((adj, adj), dim=0)
        mask_ij = torch.cat((mask_i, mask_j))
        mask = mask_ij.view(-1, 1) * mask_ij
        n = z_i.shape[0]
        N = 2 * n
        z = torch.cat((z_i, z_j), dim=0)
        sim = torch.matmul(z, z.T) / self.temperature
        sim = sim * adj
        sim[mask == 0] = -1e5  # except missing instances
        sim = sim.fill_diagonal_(-1e5)  # self

        label = torch.cat((torch.arange(n, N), torch.arange(0, n))).to(z_i.device)
        loss = self.criterion(sim, label)

        # Subtract the excess constant
        n1 = 2 * torch.sum(1 - mask_i * mask_j)  # the num when mask[i,v]*mask[i,w]=0
        n2 = torch.sum(1 - mask_ij)  # v-th instance missing i.e., mask[i,v]=0.
        loss += n2 * torch.log(torch.tensor(1 / N))  #
        loss -= (n1 - n2) * 1e5  #
        loss /= N

        return loss


class WeightBCELoss(nn.Module):
    def __init__(self, ):
        super(WeightBCELoss, self).__init__()

    def forward(self, y_true, y_pred, mask_label):
        if torch.sum(mask_label) == 0:
            return torch.tensor(0.)

        eps = 1e-5
        loss = -(mask_label * (y_true * torch.log(y_pred + eps) + (1 - y_true) * torch.log(1 - y_pred + eps)))
        loss = torch.sum(loss) / torch.sum(mask_label)

        return loss


class LabeContrastivelLoss(nn.Module):
    def __init__(self, temperature):
        super(LabeContrastivelLoss, self).__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss(reduction="sum")

    def forward(self, y_i, y_j,similarity, threshold):
        similarity = (similarity < threshold)   # The position of negative pairs

        n = y_i.size(1)
        N = 2 * n

        similarity_cat = torch.cat(
            [torch.cat([similarity, similarity], dim=1), torch.cat([similarity, similarity], dim=1)], dim=0)

        similarity_cat[torch.arange(0, n), torch.arange(n, N)] = 1
        similarity_cat[torch.arange(n, N), torch.arange(0, n)] = 1
        similarity_cat[torch.arange(0, N), torch.arange(0, N)] = 0

        y_i = y_i.t()
        y_j = y_j.t()
        y = torch.cat((y_i, y_j), dim=0)

        similarity_mat = torch.matmul(y, y.T) / self.temperature
        norms = torch.norm(y, p=2, dim=1, keepdim=True)
        norms = norms * norms.T
        similarity_mat /= norms

        similarity_mat[similarity_cat == 0] = -1e5

        label = torch.cat((torch.arange(n, N), torch.arange(0, n))).to(y_i.device)
        loss = self.criterion(similarity_mat, label)

        return loss / N
