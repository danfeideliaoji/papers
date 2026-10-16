import numpy as np
import torch
import pickle
import heapq
from sklearn.neighbors import KNeighborsClassifier,NearestNeighbors

from torch_geometric.data import Data,Dataset,DataLoader
class Graph_Generator(torch.nn.Module):
	def __init__(self):
		super(Graph_Generator, self).__init__()

	def generate_samples(self,sample_feature):
		num_rows = sample_feature[0].shape[0]
		samples = torch.stack([torch.stack([view[i] for view in sample_feature]) for i in range(num_rows)])
		return samples

	def generate_labels(self, label_feats, target):
		samples = []
		for i in range(target.shape[0]):
			sample = [label_feats[j] for j in range(target.shape[1]) if 1]
			if sample:
				sample = torch.stack(sample)
				samples.append(sample)
		return samples

	def build_graph(self,sample_feature,label_feat,sample_features,indices_views,node_idx):
		sample_features = torch.transpose(sample_features,0,1)
		num_ins,num_label=sample_feature.shape[0],label_feat.shape[0]
		num_views = sample_feature.shape[0]
		k_neighbor = indices_views.shape[2]
		"""build object graph"""
		self_ins_edges = torch.eye(num_ins, dtype=torch.long)
		adjacency_ins = torch.zeros(num_ins, num_ins, dtype=torch.long)
		for i in range(num_ins - 1):
			adjacency_ins[i][i + 1] = 1
			adjacency_ins[i + 1][i] = 1
		ins_edges = self_ins_edges + adjacency_ins
		edge_idx_ins = torch.nonzero(ins_edges, as_tuple=False).transpose(1, 0).cuda()
		edge_attr_ins = torch.cat((sample_feature[edge_idx_ins[0, :]], sample_feature[edge_idx_ins[1, :]]), dim=1)
		for i in range(num_ins):
			for v in range(num_views):
				if v != i % num_views:
					continue
				temp = num_ins+(k_neighbor*i)
				for m in range(k_neighbor):
					edge_idx_ins = torch.cat(
						(edge_idx_ins, torch.tensor([[(temp+m)],[v]], dtype=torch.long).cuda()), dim=1)
				neighbors_view = indices_views[v, node_idx]
				for neighbor in neighbors_view:
					neighbor_idx = neighbor
					edge_attr_ins = torch.cat((edge_attr_ins, torch.cat((sample_features[v][neighbor_idx],sample_feature[i]),dim=0).unsqueeze(0)),dim=0)
		kwargs = [num_ins, num_label]
		"""build cross edge"""
		bbox_idx_C = torch.from_numpy(np.array(list(range(num_ins))).repeat(num_label).reshape(-1, 1)).type(torch.long)
		label_idx_C = torch.from_numpy(np.array(list(range(num_label))).reshape(-1, 1)).repeat(num_ins, 1).type(torch.long)
		edge_index_C = torch.cat((bbox_idx_C, label_idx_C), dim=1).cuda()
		#[[0,0],[0,1],[0,2],[0,3],[0,4],[0,5],[1,0],[1,1],[1,2],[1,3],[1,4],[1,5]]
		edge_attr_C = torch.cat((sample_feature[edge_index_C[:,0]], label_feat[edge_index_C[:,1]]), dim=1)
		cross_edge=torch.cat((edge_index_C.type(torch.float),edge_attr_C.type(torch.float)),dim=1)

		"""build label graph"""
		self_edges = torch.eye(num_label, dtype=torch.long)
		adjacency = torch.zeros(num_label, num_label, dtype=torch.long)
		for i in range(num_label - 1):
			adjacency[i][i + 1] = 1
			adjacency[i + 1][i] = 1
		edges = self_edges + adjacency
		edge_idx_label = torch.nonzero(edges, as_tuple=False).transpose(1, 0).cuda()
		edge_attr_label = torch.cat((label_feat[edge_idx_label[0, :]], label_feat[edge_idx_label[1, :]]), dim=1)
		return edge_idx_ins, edge_attr_ins, kwargs, edge_idx_label, edge_attr_label, cross_edge,
	def forward(self,sample_features,label_feats,target,args,indices_views,train_view,batch_num):
		target = target[0]  # [32,6]
		x_ins_batch, edge_idx_ins_batch, edge_attr_ins_batch, x_label_batch, edge_idx_label_batch, edge_attr_label_batch, cross_edge_batch= [], [], [], [], [], [],[]
		node_sum = 0
		labe_sum = 0
		kwargs_ins = []
		label_feats = self.generate_labels(label_feats,target)

		reshape_sample_features = torch.transpose(sample_features,0,1)#[2,32,128]
		reshape_train_view = torch.transpose(train_view,0,1)#[2,1924,128]
		for i in range(sample_features.shape[0]):
			edge_idx_ins, edge_attr_ins, kwargs, edge_idx_label, edge_attr_label, cross_edge = self.build_graph(sample_features[i], label_feats[i], train_view,indices_views,i+((args.batch_size)*batch_num))
			for view in range(sample_features.shape[1]):
				x_ins_batch.append(reshape_sample_features[view, i])
			for v in range(indices_views.shape[0]):
				for m in range(indices_views.shape[2]):
					x_ins_batch.append(reshape_train_view[v,indices_views[v,i+((reshape_sample_features.shape[0]+args.k*reshape_sample_features.shape[0])*batch_num),m]])
			edge_idx_ins_batch.append(edge_idx_ins+node_sum)
			edge_attr_ins_batch.append(edge_attr_ins)
			kwargs_ins.append(kwargs)
			x_label_batch.append(label_feats[i])
			edge_idx_label_batch.append(edge_idx_label+labe_sum)
			edge_attr_label_batch.append(edge_attr_label)
			cross_edge_batch.append(cross_edge)
			node_sum += (sample_features[i].shape[0]+(indices_views.shape[0]*indices_views.shape[2]))
			labe_sum += label_feats[i].shape[0]
		x_ins_batch = torch.stack(x_ins_batch)
		edge_idx_ins_batch = torch.cat(edge_idx_ins_batch, dim=1)
		edge_attr_ins_batch = torch.cat(edge_attr_ins_batch)
		x_label_batch = torch.cat(x_label_batch)
		edge_idx_label_batch = torch.cat(edge_idx_label_batch, dim=1)
		edge_attr_label_batch = torch.cat(edge_attr_label_batch)
		cross_edge_batch = torch.cat(cross_edge_batch)
		graph_ins = Data(x=x_ins_batch, edge_index=edge_idx_ins_batch,
						 edge_attr=edge_attr_ins_batch, kwargs=kwargs_ins)
		graph_label = Data(x=x_label_batch, edge_index=edge_idx_label_batch,
						   edge_attr=edge_attr_label_batch, kwargs=cross_edge_batch)
		return graph_ins, graph_label


