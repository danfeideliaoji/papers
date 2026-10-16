"""式 (1)-(3) 的批图：无向边只存一份，双向消息共用其属性。"""
import torch
from torch_geometric.data import Data

class Graph_Generator(torch.nn.Module):
    def generate_samples(self,features):
        return torch.stack(features,1)

    def forward(self,samples,label_feats,target,args,indices_views,train_view,batch_num):
        # target 不参与构图；各样本保留全部标签节点。
        batch,views,hidden=samples.shape
        labels,k=len(label_feats),indices_views.shape[-1]
        start=args.batch_size*batch_num
        ids=indices_views[:,start:start+batch].to(samples.device)
        if ids.shape[1]!=batch: raise ValueError('邻居索引与批次行号不一致')
        neighbor=torch.stack([train_view[ids[v],v] for v in range(views)],1)
        stride=views*(1+k)
        nodes=torch.cat([samples,neighbor.reshape(batch,views*k,hidden)],1).reshape(-1,hidden)
        device=samples.device
        # 完全视图图无自环，式 (4) 只读取 V-1 个其他视图。
        view_edges=torch.triu_indices(views,views,offset=1,device=device)
        neighbor_edges=torch.stack([torch.arange(views*k,device=device)+views,
                                    torch.arange(views,device=device).repeat_interleave(k)])
        local=torch.cat([view_edges,neighbor_edges],1)
        index=(local[None]+torch.arange(batch,device=device)[:,None,None]*stride).permute(1,0,2).reshape(2,-1)
        attrs=torch.cat([nodes[index[0]],nodes[index[1]]],-1)
        label_nodes=label_feats[None].expand(batch,-1,-1).reshape(-1,hidden)
        local_label=torch.triu_indices(labels,labels,offset=1,device=device)
        label_index=(local_label[None]+torch.arange(batch,device=device)[:,None,None]*labels).permute(1,0,2).reshape(2,-1)
        label_attrs=torch.cat([label_nodes[label_index[0]],label_nodes[label_index[1]]],-1)
        src=torch.arange(views,device=device).repeat_interleave(labels)
        dst=torch.arange(labels,device=device).repeat(views)
        local_cross=torch.stack([src,dst],-1)
        cross_attrs=torch.cat([samples[:,:,None,:].expand(-1,-1,labels,-1),
                               label_feats[None,None].expand(batch,views,-1,-1)],-1).reshape(-1,2*hidden)
        cross=torch.cat([local_cross.repeat(batch,1).to(samples.dtype),cross_attrs],-1)
        return (Data(x=nodes,edge_index=index,edge_attr=attrs,kwargs=[[views,labels]]*batch),
                Data(x=label_nodes,edge_index=label_index,edge_attr=label_attrs,kwargs=cross))
