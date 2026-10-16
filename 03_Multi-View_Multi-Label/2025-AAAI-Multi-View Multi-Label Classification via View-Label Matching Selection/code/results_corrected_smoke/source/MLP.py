# 将各视图不同维度的原始特征映射到统一隐藏维度。
# 注意：merge_views() 不创建优化器，而且 detach 输出；这些 MLP 是冻结的随机投影。

import torch
import torch.nn as nn
from torch.nn import Sequential as Seq, Linear as Lin,LayerNorm, ReLU,Sigmoid
# 按输入特征维度选择不同深度的全连接网络。
class MLP(nn.Module):
    # 较大的输入使用更多逐级降维层，所有视图输出维度由 output_dim 指定。
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
    # 线性层要求浮点输入，输出 [样本数, output_dim]。
    def forward(self, x):
        x = x.float()
        x = self.encoder(x)
        return x
# 每个视图独立建立一个 MLP，得到形状相同的特征以便后续堆叠。
# 模型对象只在本函数中使用；它们不属于后续图模型的可训练参数。
def merge_views(data, args):
    merged_outputs = []
    for view_data in data:
        input_dim = view_data.shape[1]
        output_dim = args.nhidden
        mlp_model = MLP(input_dim, output_dim, args.dropout)
        output = mlp_model(view_data)
        # 切断输入投影的计算图，后续训练不会更新此 MLP；固定种子对重放很重要。
        output = output.detach()
        merged_outputs.append(output)

    return merged_outputs