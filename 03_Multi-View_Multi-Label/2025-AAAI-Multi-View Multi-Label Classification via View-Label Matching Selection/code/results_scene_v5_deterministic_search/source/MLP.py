# 将各视图不同维度的原始特征映射到统一隐藏维度。
# 修正版由 strict_training.VAMS 注册各视图 MLP，参与优化并保存到检查点。

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
# 旧冻结投影入口禁止继续使用；旧模型核验从 results/source 导入原实现。
def merge_views(data, args):
    raise RuntimeError('冻结随机投影接口已停用，请使用 strict_training.VAMS 的可训练输入投影')
