"""VAMS 式 (4)-(10)。新训练直接导入本模块，避免旧模型文件漂移。

MLP 宽度沿用作者代码，论文未给出精确结构。辅助邻居保留为外部信息源，
论文没有其独立更新公式；邻居边仍随每层更新，输入特征可以回传梯度。
"""
import torch
from torch import nn
from torch_geometric.data import Data
from paper_graph import Graph_Generator
from conv_layer import ConvLayer

def mlp(inputs,outputs):
    return nn.Sequential(nn.Linear(inputs,256),nn.ReLU(),nn.LayerNorm(256),
                         nn.Linear(256,outputs),nn.ReLU(),nn.LayerNorm(outputs))

def aggregate(messages,index,count,mean=True):
    result=messages.new_zeros(count,messages.shape[-1]).index_add(0,index,messages)
    if mean:
        degree=messages.new_zeros(count).index_add(0,index,messages.new_ones(len(index)))
        result=result/degree.clamp_min(1)[:,None]
    return result

class EdgeModel(nn.Module):
    def __init__(self,insN_dim,insE_dim,labelN_dim,labelE_dim,crossE_dim):
        super().__init__()
        # 式 (7) 跨视图边与邻居边共享 rho 和 phi；三类边各自独立。
        self.edge_rho_ins=mlp(2*insN_dim,insE_dim)
        self.edge_mlp_ins=mlp(2*insE_dim,insE_dim)
        self.edge_rho_label=mlp(2*labelN_dim,labelE_dim)
        self.edge_mlp_label=mlp(2*labelE_dim,labelE_dim)
        self.edge_rho_cross=mlp(insN_dim+labelN_dim,crossE_dim)
        self.edge_mlp_cross=mlp(2*crossE_dim,crossE_dim)

    def forward(self,si,sl,sc):
        vi,ii,ei=si
        vl,il,el=sl
        im,em=sc
        # 式 (7)-(9)：rho 聚合两个端点，phi 融合旧边与聚合结果。
        ei=self.edge_mlp_ins(torch.cat([ei,self.edge_rho_ins(torch.cat([vi[ii[0]],vi[ii[1]]],-1))],-1))
        el=self.edge_mlp_label(torch.cat([el,self.edge_rho_label(torch.cat([vl[il[0]],vl[il[1]]],-1))],-1))
        em=self.edge_mlp_cross(torch.cat([em,self.edge_rho_cross(torch.cat([vi[im[0]],vl[im[1]]],-1))],-1))
        return (vi,ii,ei),(vl,il,el),(im,em)

class NodeModel(nn.Module):
    def __init__(self,insN_dim,insE_dim,labelN_dim,labelE_dim,crossE_dim):
        super().__init__()
        # 式 (4) 三路独立消息网络，更新网络输入为自身 + 三路消息。
        self.node_mlp_ins_inner=mlp(insE_dim+insN_dim,insN_dim)
        self.node_mlp_ins_neighbor=mlp(insE_dim+insN_dim,insN_dim)
        self.node_mlp_ins_inter=mlp(crossE_dim+labelN_dim,insN_dim)
        self.node_mlp_ins=mlp(4*insN_dim,insN_dim)
        self.node_mlp_label_inner=mlp(labelE_dim+labelN_dim,labelN_dim)
        self.node_mlp_label_inter=mlp(crossE_dim+insN_dim,labelN_dim)
        self.node_mlp_label=mlp(3*labelN_dim,labelN_dim)

    def forward(self,si,sl,sc):
        vi,ii,ei=si
        vl,il,el=sl
        im,em=sc
        views,k=self.num_views,self.num_neighbors
        stride=views*(1+k)
        count=len(vi)//stride*views
        compact=lambda ids: ids//stride*views+ids%stride
        # 图内边索引为 [接收节点,发送节点]；完整视图/标签图已有双向消息。
        current=(ii[0]%stride<views)&(ii[1]%stride<views)
        dst,src=ii[:,current]
        edges=ei[current]
        other=aggregate(self.node_mlp_ins_inner(torch.cat([edges,vi[src]],-1)),compact(dst),count)
        dst,src=ii[:,~current]
        neighbor=aggregate(self.node_mlp_ins_neighbor(torch.cat([ei[~current],vi[src]],-1)),compact(dst),count)
        # 式 (4) 标签通道没有 1/Q：按字面求和；其余带分母的通道取均值。
        matched=aggregate(self.node_mlp_ins_inter(torch.cat([em,vl[im[1]]],-1)),compact(im[0]),count,mean=False)
        ids=(torch.arange(len(vi)//stride,device=vi.device)[:,None]*stride+
             torch.arange(views,device=vi.device)[None]).reshape(-1)
        updated=self.node_mlp_ins(torch.cat([vi[ids],other,neighbor,matched],-1))
        dst,src=il
        edges=el
        # 式 (5)-(6)：标签邻接 Q-1 个标签和 V 个视图，两路分别取均值。
        other_labels=aggregate(self.node_mlp_label_inner(torch.cat([edges,vl[src]],-1)),dst,len(vl))
        views_to_labels=aggregate(self.node_mlp_label_inter(torch.cat([em,vi[im[0]]],-1)),im[1],len(vl))
        updated_labels=self.node_mlp_label(torch.cat([vl,other_labels,views_to_labels],-1))
        return (vi.index_copy(0,ids,updated),ii,ei),(updated_labels,il,el),sc

class _Model(nn.Module):
    def __init__(self,dim_node_ins=128,dim_node_label=128,dim_edge_ins=256,dim_edge_label=256,conv_layers=3):
        super().__init__()
        if conv_layers<1: raise ValueError('conv_layers 必须大于零')
        self.graph_generator=Graph_Generator()
        h=128
        self.red_edge_ins=mlp(dim_edge_ins,h)
        self.red_edge_label=mlp(dim_edge_label,h)
        self.red_edge_cross=mlp(dim_node_ins+dim_node_label,h)
        self.layers=nn.ModuleList([ConvLayer(EdgeModel(h,h,h,h,h),NodeModel(h,h,h,h,h)) for _ in range(conv_layers)])
        self.decoder_edge_cross=nn.Sequential(nn.Linear(h,h),nn.ReLU(),nn.LayerNorm(h),nn.Linear(h,1),nn.Sigmoid())

    def generate_cross_graph(self,graph_label,counts,num_neighbors=5):
        edges=graph_label.kwargs
        views,labels=map(int,counts[0])
        batch=len(counts)
        index=edges[:,:2].long().reshape(batch,views*labels,2).clone()
        offsets=torch.arange(batch,device=edges.device)
        index[:,:,0]+=offsets[:,None]*views*(1+num_neighbors)
        index[:,:,1]+=offsets[:,None]*labels
        return Data(edge_index=index.reshape(-1,2).T,edge_attr=edges[:,2:])

    def forward(self,samples,inp,target,args,indices_views,reshaped_train_view,batch_num):
        # 向量化构图直接在输入设备执行，索引操作保留输入/邻居的梯度。
        neighbors=indices_views[0]
        gi,gl=self.graph_generator(samples,inp[0],target.transpose(0,1),args,neighbors,reshaped_train_view[0],batch_num)
        counts=torch.tensor(gi.kwargs)
        gc=self.generate_cross_graph(gl,counts,neighbors.shape[-1])
        si=(gi.x,gi.edge_index,self.red_edge_ins(gi.edge_attr))
        sl=(gl.x,gl.edge_index,self.red_edge_label(gl.edge_attr))
        sc=(gc.edge_index,self.red_edge_cross(gc.edge_attr))
        for layer in self.layers:
            layer.node_model.num_views=samples.shape[1]
            layer.node_model.num_neighbors=neighbors.shape[-1]
            # Algorithm 1：节点卷积在前，边卷积在后。
            si,sl,sc=layer(si,sl,sc)
        return self.decoder_edge_cross(sc[1]),counts
