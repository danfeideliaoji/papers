# 一层 VAMS 图消息传递：论文算法 1 要求先更新节点，再更新边。
# state_ins、state_label、state_cross 分别表示视图/邻居图、标签图和视图—标签跨图边。

import torch


# 把边更新和节点更新封装为一个传播层，保持三类图状态同步。
class ConvLayer(torch.nn.Module):

    # global_model 为原作者预留接口，当前 forward() 没有执行全局更新。
    def __init__(self, edge_model=None, node_model=None, global_model=None):
        super(ConvLayer, self).__init__()
        self.edge_model = edge_model
        self.node_model = node_model
        self.global_model = global_model

    # 最后一次节点更新必须通过随后的边更新进入匹配得分。
    def forward(self, state_ins, state_label, state_cross):
        state_ins, state_label, state_cross = self.node_model(state_ins, state_label, state_cross)
        state_ins, state_label, state_cross = self.edge_model(state_ins, state_label, state_cross)
        return state_ins, state_label, state_cross

    # 打印当前传播层包含的边更新器与节点更新器。
    def __repr__(self):
        return ('{}(\n'
                '    edge_model={},\n'
                '    node_model={},\n'
                '    global_model={}\n'
                ')').format(self.__class__.__name__, self.edge_model,
                            self.node_model, self.global_model)
