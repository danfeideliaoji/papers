import torch
import copy
from torch.nn import Sequential as Seq, Linear as Lin,LayerNorm, ReLU,Sigmoid
try:
    from torch_scatter import scatter_mean
except Exception:
    def scatter_mean(src, index, dim=0, dim_size=None):
        if dim_size is None:
            dim_size = int(index.max().item()) + 1
        out_shape = list(src.shape)
        out_shape[dim] = dim_size
        out = torch.zeros(out_shape, dtype=src.dtype, device=src.device)
        count = torch.zeros(out_shape, dtype=src.dtype, device=src.device)
        index_expanded = index.unsqueeze(-1).expand_as(src) if src.dim() > 1 else index
        out.scatter_add_(dim, index_expanded, src)
        count.scatter_add_(dim, index_expanded, torch.ones_like(src))
        return out / count.clamp(min=1)

try:
    from torchvision.models import resnet101
except Exception:
    resnet101 = None
from conv_layer import ConvLayer
from build_graph import Graph_Generator
import torch.nn as nn
from torch_geometric.data import Dataset,Data
# latent_dim = 512
# final_latent_dim = 256
latent_dim = 256
final_latent_dim = 128
node_ins_num = 2
k_neighbor = 5
batch_size = 64

class EdgeModel(torch.nn.Module):
    def __init__(self, insN_dim, insE_dim, labelN_dim, labelE_dim, crossE_dim):
        super(EdgeModel, self).__init__()
        self.latent_insN_dim = final_latent_dim
        self.latent_insE_dim = final_latent_dim
        self.latent_labelN_dim = final_latent_dim
        self.latent_labelE_dim = final_latent_dim
        self.latent_crossE_dim = final_latent_dim
        self.node_encoder_ins = Seq(Lin(insN_dim, latent_dim), ReLU(), LayerNorm(latent_dim), Lin(latent_dim, self.latent_insN_dim), ReLU(), LayerNorm(self.latent_insN_dim))
        self.edge_encoder_ins = Seq(Lin(insE_dim, latent_dim), ReLU(), LayerNorm(latent_dim), Lin(latent_dim, self.latent_insE_dim), ReLU(), LayerNorm(self.latent_insE_dim))
        self.node_encoder_label = Seq(Lin(labelN_dim, latent_dim), ReLU(), LayerNorm(latent_dim), Lin(latent_dim, self.latent_labelN_dim), ReLU(), LayerNorm(self.latent_labelN_dim))
        self.edge_encoder_label = Seq(Lin(labelE_dim, latent_dim), ReLU(), LayerNorm(latent_dim), Lin(latent_dim, self.latent_labelE_dim), ReLU(), LayerNorm(self.latent_labelE_dim))
        self.edge_encoder_cross = Seq(Lin(crossE_dim, latent_dim), ReLU(), LayerNorm(latent_dim), Lin(latent_dim, self.latent_crossE_dim), ReLU(), LayerNorm(self.latent_crossE_dim))

        self.edge_mlp_ins = Seq(Lin(self.latent_insN_dim*2 + self.latent_insE_dim, latent_dim), ReLU(), LayerNorm(latent_dim),
                                  Lin(latent_dim, insE_dim), ReLU(), LayerNorm(insE_dim))
        self.edge_mlp_label = Seq(Lin(self.latent_labelN_dim *2+self.latent_insE_dim, latent_dim), ReLU(), LayerNorm(latent_dim),
                                  Lin(latent_dim, labelE_dim),ReLU(), LayerNorm(labelE_dim))
        self.edge_mlp_cross = Seq(Lin(self.latent_insN_dim+self.latent_labelN_dim+self.latent_crossE_dim, latent_dim), ReLU(), LayerNorm(latent_dim),
                                  Lin(latent_dim, crossE_dim),ReLU(), LayerNorm(crossE_dim))

    def forward(self,state_ins,state_label,state_cross):

        node_ins, edge_index_ins, edge_attr_ins=state_ins

        node_label, edge_index_label, edge_attr_label=state_label

        edge_index_cross, edge_attr_cross=state_cross

        """mapping the attributes into a latent space"""
       
        # node_ins, edge_attr_ins =self.node_encoder_ins(node_ins),self.edge_encoder_ins(edge_attr_ins)
        # node_label, edge_attr_label = self.node_encoder_label(node_label), self.edge_encoder_label(edge_attr_label)
        # edge_attr_cross= self.edge_encoder_cross(edge_attr_cross)

        """instance edges convolution"""
        row_ins, col_ins = edge_index_ins
        src_ins,dest_ins=node_ins[row_ins],node_ins[col_ins]#[640,128]
        out_ins = torch.cat([src_ins, dest_ins, edge_attr_ins], 1)#[640,384]
        edge_attr_ins = self.edge_mlp_ins(out_ins)
        # ########################
        # first_n_indices_edge = torch.arange(node_ins_num*node_ins_num)
        # every_step_index_edge = torch.arange(0, edge_attr_ins.size(0), step=((node_ins_num*node_ins_num) + (k_neighbor * node_ins_num)))
        # update_indices_edge = (every_step_index_edge.view(-1, 1) + first_n_indices_edge).view(-1)
        # src_ins, dest_ins = node_ins[row_ins[update_indices_edge]], node_ins[col_ins[update_indices_edge]]
        # out_ins = torch.cat([src_ins, dest_ins, edge_attr_ins[update_indices_edge]], 1)
        # edge_attr_ins_to_update = self.edge_mlp_ins(out_ins)
        # edge_attr_ins[update_indices_edge] = edge_attr_ins_to_update
        # ########################

        """label edges convolution"""
        row_label, col_label = edge_index_label
        src_label, dest_label = node_label[row_label], node_label[col_label]
        out_label= torch.cat([src_label , dest_label, edge_attr_label], 1)
        edge_attr_label = self.edge_mlp_label(out_label)

        """cross edges convotion"""
        row_cross, col_corss = edge_index_cross
        src_cross, dest_cross = node_ins[row_cross], node_label[col_corss]
        out_cross = torch.cat([src_cross, dest_cross, edge_attr_cross], 1)
        edge_attr_cross = self.edge_mlp_cross(out_cross)

        state_ins = node_ins, edge_index_ins, edge_attr_ins
        state_label = node_label, edge_index_label, edge_attr_label
        state_cross = edge_index_cross, edge_attr_cross

        return state_ins, state_label, state_cross

class NodeModel(torch.nn.Module):
    def __init__(self,insN_dim,insE_dim,labelN_dim,labelE_dim,crossE_dim):
        super(NodeModel, self).__init__()
        self.latent_insN_dim=final_latent_dim
        self.latent_insE_dim = final_latent_dim
        self.latent_labelN_dim = final_latent_dim
        self.latent_labelE_dim = final_latent_dim
        self.latent_crossE_dim = final_latent_dim

        self.node_encoder_ins = Seq(Lin(insN_dim,latent_dim),ReLU(),LayerNorm(latent_dim), Lin(latent_dim, self.latent_insN_dim),ReLU(),LayerNorm(self.latent_insN_dim))
        self.edge_encoder_ins = Seq(Lin(insE_dim, latent_dim), ReLU(), LayerNorm(latent_dim), Lin(latent_dim, self.latent_insE_dim), ReLU(),LayerNorm(self.latent_insE_dim))
        self.node_encoder_label = Seq(Lin(labelN_dim, latent_dim), ReLU(), LayerNorm(latent_dim), Lin(latent_dim, self.latent_labelN_dim), ReLU(),LayerNorm(self.latent_labelN_dim))
        self.edge_encoder_label = Seq(Lin(labelE_dim, latent_dim), ReLU(), LayerNorm(latent_dim), Lin(latent_dim, self.latent_labelE_dim), ReLU(),LayerNorm(self.latent_labelE_dim))
        self.edge_encoder_cross = Seq(Lin(crossE_dim, latent_dim), ReLU(), LayerNorm(latent_dim), Lin(latent_dim, self.latent_crossE_dim), ReLU(),LayerNorm(self.latent_crossE_dim))

        """############### modify #####################"""
        self.node_mlp_ins_inner = Seq(Lin(self.latent_insN_dim + self.latent_insE_dim, latent_dim), ReLU(), LayerNorm(latent_dim),
                                  Lin(latent_dim, self.latent_insN_dim), ReLU(),LayerNorm(self.latent_insN_dim))
        self.node_mlp_ins_inter = Seq(Lin(self.latent_insN_dim + self.latent_insE_dim, latent_dim), ReLU(), LayerNorm(latent_dim),
                                  Lin(latent_dim, self.latent_insN_dim), ReLU(),LayerNorm(self.latent_insN_dim))
        self.node_mlp_ins = Seq(Lin(self.latent_insN_dim*3,latent_dim),ReLU(),LayerNorm(latent_dim),
                                Lin(latent_dim,insN_dim),ReLU(),LayerNorm(insN_dim))#

        """############### modify done #####################"""
        self.node_mlp_label_inner = Seq(Lin(self.latent_labelN_dim+self.latent_labelE_dim,latent_dim), ReLU(), LayerNorm(latent_dim),
                                        Lin(latent_dim, self.latent_labelN_dim), ReLU(), LayerNorm(self.latent_labelN_dim))
        self.node_mlp_label_inter = Seq(Lin(self.latent_labelN_dim+self.latent_labelE_dim, latent_dim), ReLU(), LayerNorm(latent_dim),
                                        Lin(latent_dim, self.latent_labelN_dim), ReLU(), LayerNorm(self.latent_labelN_dim))
        self.node_mlp_label = Seq(Lin(self.latent_labelN_dim*3, latent_dim), ReLU(), LayerNorm(latent_dim),
                                  Lin(latent_dim, labelN_dim), ReLU(), LayerNorm(labelN_dim))

    def forward(self, state_ins,state_label,state_cross):

        node_ins, edge_index_ins, edge_attr_ins=state_ins
        node_label, edge_index_label, edge_attr_label=state_label
        edge_index_cross, edge_attr_cross=state_cross

        """mapping the attributes into a latent space"""
        # node_ins, edge_attr_ins =self.node_encoder_ins(node_ins),self.edge_encoder_ins(edge_attr_ins)
        # node_label, edge_attr_label = self.node_encoder_label(node_label), self.edge_encoder_label(edge_attr_label)
        # edge_attr_cross= self.edge_encoder_cross(edge_attr_cross)

        """cross edges are directed from samples to labels"""

        """instance node attribute update"""
        """############################# modify #############################"""
        row_ins, col_ins = edge_index_ins
        # ################
        new_col_ins = []
        num = k_neighbor*node_ins_num#6
        for i in range(batch_size):
            temp = num * i
            for j in range((node_ins_num+(node_ins_num-1)*2)+node_ins_num*k_neighbor):#10
                new_col_ins.append(temp)
        new_col_ins = torch.tensor(new_col_ins, device=col_ins.device)
        new_col_ins = new_col_ins[:col_ins.shape[0]]
        new_col_ins = col_ins - new_col_ins

        # ################
        row_cross, col_cross = edge_index_cross
        out_ins_inner = torch.cat([node_ins[row_ins],edge_attr_ins],dim=1)
        out_ins_inner = self.node_mlp_ins_inner(out_ins_inner)#[640,128]

        out_ins_inner = scatter_mean(out_ins_inner,new_col_ins,dim=0,dim_size=new_col_ins[-1]+1)#

        out_ins_inter =torch.cat([node_label[col_cross],edge_attr_cross],dim=1)
        out_ins_inter = self.node_mlp_ins_inter(out_ins_inter)#[1792,128]
        # Map global view-node indices to the compact list, excluding neighbor nodes.
        sample_stride = node_ins_num * (1 + k_neighbor)
        compact_row_cross = (row_cross // sample_stride) * node_ins_num + row_cross % sample_stride
        out_ins_inter = scatter_mean(out_ins_inter,compact_row_cross,dim=0,dim_size=new_col_ins[-1]+1)

        first_n_indices = torch.arange(node_ins_num, device=node_ins.device)
        every_step_index = torch.arange(0, node_ins.size(0), step=(node_ins_num+(k_neighbor*node_ins_num)), device=node_ins.device)
        update_indices = (every_step_index.view(-1, 1) + first_n_indices).view(-1)
        node_ins_to_update = node_ins[update_indices]
        node_ins_to_update = torch.cat([node_ins_to_update, out_ins_inner, out_ins_inter], dim=1)
        node_ins_to_update = self.node_mlp_ins(node_ins_to_update)

        node_ins[update_indices] = node_ins_to_update
        # ########################

        # node_ins=torch.cat([node_ins,out_ins_inner,out_ins_inter],dim=1)
        # node_ins=self.node_mlp_ins(node_ins)
        """############################# modify done#############################"""

        """label node attribute update"""
        row_label, col_label = edge_index_label
        row_cross, col_cross = edge_index_cross
        out_label_inner = torch.cat([node_label[row_label], edge_attr_label], dim=1)
        out_label_inner = self.node_mlp_label_inner(out_label_inner)
        out_label_inner = scatter_mean(out_label_inner, col_label, dim=0, dim_size=node_label.size(0))
        
        out_label_inter = torch.cat([node_ins[row_cross], edge_attr_cross], dim=1)
        out_label_inter = self.node_mlp_label_inter(out_label_inter)
        out_label_inter = scatter_mean(out_label_inter, col_cross, dim=0, dim_size=node_label.size(0))

        node_label = torch.cat([node_label, out_label_inner, out_label_inter], dim=1)
        node_label=self.node_mlp_label(node_label)

        state_ins=node_ins, edge_index_ins, edge_attr_ins
        state_label=node_label, edge_index_label, edge_attr_label
        state_cross=edge_index_cross, edge_attr_cross

        return state_ins,state_label,state_cross

class _Model(torch.nn.Module):
    def __init__(self, dim_node_ins=128, dim_node_label=128, dim_edge_ins=256, dim_edge_label=256):
        super(_Model,self).__init__()
        self.graph_generator = Graph_Generator()
        self.red_dim_insN = final_latent_dim
        self.red_dim_insE = final_latent_dim
        self.red_dim_labelN = final_latent_dim
        self.red_dim_labelE = final_latent_dim
        self.red_dim_crossE = final_latent_dim

        self.red_node_ins=Seq(
            Lin(dim_node_ins,latent_dim),ReLU(),LayerNorm(latent_dim),Lin(latent_dim,self.red_dim_insN),ReLU(),LayerNorm(self.red_dim_insN)
        )
        self.red_edge_ins=Seq(
            Lin(dim_edge_ins, latent_dim), ReLU(), LayerNorm(latent_dim), Lin(latent_dim, self.red_dim_insE), ReLU(), LayerNorm(self.red_dim_insE)
        )
        self.red_node_label = Seq(
            Lin(dim_node_label, latent_dim), ReLU(), LayerNorm(latent_dim), Lin(latent_dim, self.red_dim_labelN), ReLU(), LayerNorm(self.red_dim_labelN)
        )
        self.red_edge_label= Seq(
            Lin(dim_edge_label, latent_dim), ReLU(), LayerNorm(latent_dim), Lin(latent_dim, self.red_dim_labelE), ReLU(), LayerNorm(self.red_dim_labelE)
        )
        self.red_edge_cross=Seq(
            Lin(dim_node_label+dim_node_ins, latent_dim), ReLU(), LayerNorm(latent_dim), Lin(latent_dim, self.red_dim_crossE), ReLU(), LayerNorm(self.red_dim_crossE)
        )
        self.pro_layer1=ConvLayer(EdgeModel(self.red_dim_insN,self.red_dim_insE,self.red_dim_labelN,self.red_dim_labelE,self.red_dim_crossE),
                                  NodeModel(self.red_dim_insN,self.red_dim_insE,self.red_dim_labelN,self.red_dim_labelE,self.red_dim_crossE))
        self.pro_layer2=ConvLayer(EdgeModel(self.red_dim_insN,self.red_dim_insE,self.red_dim_labelN,self.red_dim_labelE,self.red_dim_crossE),
                                  NodeModel(self.red_dim_insN,self.red_dim_insE,self.red_dim_labelN,self.red_dim_labelE,self.red_dim_crossE))
        self.pro_layer3 = ConvLayer(EdgeModel(self.red_dim_insN,self.red_dim_insE,self.red_dim_labelN,self.red_dim_labelE,self.red_dim_crossE),
                                  NodeModel(self.red_dim_insN,self.red_dim_insE,self.red_dim_labelN,self.red_dim_labelE,self.red_dim_crossE))
        self.decoder_edge_cross= Seq(
            Lin(self.red_dim_crossE, 128), ReLU(),LayerNorm(128), Lin(128, 1), Sigmoid()
        )
    def generate_cross_graph(self,graph_label, counts):
        cross_edges = graph_label["kwargs"]
        edge_index_cross = cross_edges[:, :2]
        edge_attr_cross = cross_edges[:, 2:]
        start_index, count_ins, count_label = 0, 0, 0
        for count in counts:
            count_i, count_l = count
            num_assignment = int(count_i * count_l)
            edge_index_cross[start_index:start_index + num_assignment, 0] += count_ins
            edge_index_cross[start_index:start_index + num_assignment, 1] += count_label
            start_index += num_assignment
            # Each sample also contains k neighbor nodes for each view.
            count_ins += count_i * (1 + k_neighbor)
            count_label += count_l

        edge_index_cross = edge_index_cross.type(torch.long).transpose(1, 0)

        graph_cross = Data(x=None, edge_index=edge_index_cross, edge_attr=edge_attr_cross, y=None)
        return graph_cross
    def forward(self,samples,inp,target,args,indices_views,reshaped_train_view,batch_num):

        inp = inp[0]
        indices_views = indices_views[0]
        reshaped_train_view = reshaped_train_view[0]
        # target = torch.transpose(target, 0, 1)  # torch.size(2,32,6)
        target = torch.transpose(copy.deepcopy(target), 0, 1)  # [2,32,6]

        # Python graph construction stays on CPU; transfer the complete graph once.
        # This avoids thousands of small CUDA copies/synchronizations per batch.
        graph_ins, graph_label = self.graph_generator(
            samples.cpu(), inp.cpu(), target.cpu(), args,
            indices_views.cpu(), reshaped_train_view.cpu(), batch_num)
        counts = torch.tensor(graph_ins["kwargs"])
        graph_cross = self.generate_cross_graph(graph_label, counts)
        device = next(self.parameters()).device
        graph_ins = graph_ins.to(device)
        graph_label = graph_label.to(device)
        graph_cross = graph_cross.to(device)

        node_ins, edge_index_ins, edge_attr_ins, num_node_ins = graph_ins["x"], graph_ins["edge_index"], \
                                                                graph_ins["edge_attr"], graph_ins["kwargs"]
        node_label, edge_index_label, edge_attr_label, num_node_label = graph_label["x"], graph_label["edge_index"], \
                                                                graph_label["edge_attr"], graph_label["kwargs"]

        edge_index_cross, edge_attr_cross = graph_cross["edge_index"],graph_cross["edge_attr"]
        """dimension reduction"""

        # node_ins, edge_attr_ins = self.red_node_ins(node_ins),self.red_edge_ins(edge_attr_ins)
        # node_label, edge_attr_label = self.red_node_label(node_label), self.red_edge_label(edge_attr_label)
        edge_attr_ins = self.red_edge_ins(edge_attr_ins)
        edge_attr_label = self.red_edge_label(edge_attr_label)
        edge_attr_cross = self.red_edge_cross(edge_attr_cross)

        """convolution layers"""
        state_ins, state_label, state_cross = [node_ins, edge_index_ins, edge_attr_ins],\
                                              [node_label, edge_index_label, edge_attr_label],\
                                              [edge_index_cross, edge_attr_cross]

        state_ins, state_label, state_cross = self.pro_layer1(state_ins, state_label, state_cross)
        state_ins, state_label, state_cross = self.pro_layer2(state_ins, state_label, state_cross)
        # state_ins, state_label, state_cross = self.pro_layer3(state_ins, state_label, state_cross)

        _,edge_attr_cross=state_cross
        prediction=self.decoder_edge_cross(edge_attr_cross)
        return prediction, counts
