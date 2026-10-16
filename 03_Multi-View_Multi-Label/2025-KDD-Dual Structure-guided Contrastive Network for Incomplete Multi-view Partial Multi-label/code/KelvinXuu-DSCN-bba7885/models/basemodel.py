
import torch
import torch.nn as nn
from torch.nn import functional as F


class EncoderLayer(nn.Module):
    def __init__(self, input_dim, output_dim, activation='relu', dropout=0.2):
        super(EncoderLayer, self).__init__()
        self.dropout = dropout
        self.activation = activation

        self.linear = nn.Linear(input_dim, output_dim)

        self.batch_norm = nn.BatchNorm1d(output_dim)

        # Choose activation function
        if self.activation == 'sigmoid':
            self.activation_fn = nn.Sigmoid()
        elif self.activation == 'leakyrelu':
            self.activation_fn = nn.LeakyReLU(0.2, inplace=True)
        elif self.activation == 'tanh':
            self.activation_fn = nn.Tanh()
        elif self.activation == 'relu':
            self.activation_fn = nn.ReLU()
        else:
            raise ValueError('Unknown activation type %s' % self.activation)

        if self.dropout > 0:
            self.dropout_layer = nn.Dropout(self.dropout)
        else:
            self.dropout_layer = nn.Identity()

    def forward(self, x, active=True, batchnorm=True, dropout=True):
        x = self.linear(x)
        if batchnorm:
            x = self.batch_norm(x)
        if active:
            x = self.activation_fn(x)
        if dropout:
            x = self.dropout_layer(x)

        return x


class Encoder(nn.Module):
    """Output latent representation using multiple EncoderLayers"""

    def __init__(self, encoder_dim, activation='relu', dropout=0.2):
        super(Encoder, self).__init__()
        self.encoder_layers = nn.ModuleList()
        self.encoder_dims = len(encoder_dim) - 1
        for i in range(self.encoder_dims):
            self.encoder_layers.append(EncoderLayer(encoder_dim[i], encoder_dim[i + 1],
                                                    activation=activation, dropout=dropout
                                                    ))
        self.bn = nn.BatchNorm1d(encoder_dim[-1])
        # self.encoder_layers.append(nn.BatchNorm1d(encoder_dim[-1]))

    def forward(self, x):
        x = self.encoder_layers[0](x, dropout=True)

        for layer in self.encoder_layers[1:-1]:
            x = x + layer(x, dropout=True)

        x = self.encoder_layers[-1](x, dropout=False)
        # x = self.bn(x)

        return x


class ClassificationLayer(nn.Module):
    def __init__(self, latent_dim, n_classifications):
        super(ClassificationLayer, self).__init__()

        self._latent_dim = latent_dim
        self._n_classifications = n_classifications

        self.classification = nn.Sequential(
            nn.Linear(self._latent_dim, self._latent_dim),
            nn.BatchNorm1d(self._latent_dim),
            nn.ReLU(),

            nn.Linear(self._latent_dim, self._n_classifications),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.classification(x)
        return y


class InstanceProject(nn.Module):
    def __init__(self, latent_dim):
        super(InstanceProject, self).__init__()
        self._latent_dim = latent_dim
        self.instance_projector = nn.Sequential(
            nn.Linear(self._latent_dim, self._latent_dim, bias=True),
            nn.BatchNorm1d(self._latent_dim),
            nn.ReLU(),
            nn.Linear(self._latent_dim, self._latent_dim, bias=False),
            nn.BatchNorm1d(self._latent_dim),
        )

    def forward(self, x):
        return self.instance_projector(x)


class AttentionFusionLayer(nn.Module):
    def __init__(self, latent_dim, view_num):
        super(AttentionFusionLayer, self).__init__()
        self._latent_dim = latent_dim
        self._view_num = view_num
        self.mlp = nn.Sequential(
            nn.Linear(self._latent_dim * self._view_num, self._latent_dim * self._view_num),
            nn.BatchNorm1d(self._latent_dim * self._view_num),
            nn.ReLU(),
        )
        self.output_layer = nn.Linear(self._latent_dim * self._view_num, self._view_num, bias=True)

    def forward(self, h_list, mask, tau=10.0):
        h = torch.cat(h_list, dim=1)
        act = self.output_layer(self.mlp(h))
        act = F.sigmoid(act) / tau
        act[mask == 0] = -1e8
        e = F.softmax(act, dim=1)
        h = torch.zeros_like(h_list[0])
        for i in range(self._view_num):
            h += e[:, i].unsqueeze(1) * h_list[i]
            
        return h
