import torch
import torch.nn as nn
from torch.nn import Sequential as Seq, Linear as Lin,LayerNorm, ReLU,Sigmoid
class MLP(nn.Module):
    def __init__(self, input_dim, output_dim,dropout):
        super(MLP, self).__init__()
        self.dropout = dropout
        self.LeakyReLUrate = 0.01
        if input_dim <=512:
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, 256),
                nn.LeakyReLU(self.LeakyReLUrate),
                # nn.Dropout(p=self.dropout),
                nn.Linear(256, output_dim),
            )
        elif input_dim <= 1024:
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, 512),
                nn.LeakyReLU(self.LeakyReLUrate),
                # nn.Dropout(p=self.dropout),
                nn.Linear(512, 256),
                nn.LeakyReLU(self.LeakyReLUrate),
                # nn.Dropout(p=self.dropout),
                nn.Linear(256, output_dim)
            )
        else:
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, 2048),
                nn.LeakyReLU(self.LeakyReLUrate),
                # nn.Dropout(p=self.dropout),
                nn.Linear(2048, 1024),
                nn.LeakyReLU(self.LeakyReLUrate),
                # nn.Dropout(p=self.dropout),
                nn.Linear(1024, 512),
                nn.LeakyReLU(self.LeakyReLUrate),
                # nn.Dropout(p=self.dropout),
                nn.Linear(512, 256),
                nn.LeakyReLU(self.LeakyReLUrate),
                # nn.Dropout(p=self.dropout),
                nn.Linear(256, output_dim)
            )
    def forward(self, x):
        x = x.float()
        x = self.encoder(x)
        return x
def merge_views(data, args):
    merged_outputs = []
    for view_data in data:
        input_dim = view_data.shape[1]
        output_dim = args.nhidden
        mlp_model = MLP(input_dim, output_dim, args.dropout)
        output = mlp_model(view_data)
        output = output.detach()
        merged_outputs.append(output)

    return merged_outputs