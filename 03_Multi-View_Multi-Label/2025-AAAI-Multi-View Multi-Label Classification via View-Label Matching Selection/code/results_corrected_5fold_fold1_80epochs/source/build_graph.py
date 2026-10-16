# 把一个批次的多视图样本构造成三类关系：视图/邻居边、标签边、视图—标签边。
# 约定：N 为当前折样本数，B 为批大小，V 为视图数，L 为标签数，H 为统一特征维度。
# 节点按样本顺序排列：先该样本的 V 个视图节点，再各视图的 k 个邻居节点。

import numpy as np
import torch
import pickle
import heapq
from sklearn.neighbors import KNeighborsClassifier,NearestNeighbors

from torch_geometric.data import Data,Dataset,DataLoader
# 构图器负责节点/边及局部索引，神经网络的参数更新在 GNBlock_model.py 中完成。
class Graph_Generator(torch.nn.Module):
	# 构图本身不含可学习参数。
	def __init__(self):
		super(Graph_Generator, self).__init__()

	# 将按视图存放的特征列表转换为按样本排列的 [B, V, H] 张量。
	def generate_samples(self,sample_feature):
		num_rows = sample_feature[0].shape[0]
		samples = torch.stack([torch.stack([view[i] for view in sample_feature]) for i in range(num_rows)])
		return samples

	# 为每个样本复制全部 L 个标签节点特征，而不是只创建正标签节点。
	# target 在这里仅提供样本/标签数；if 1 表示不会按真实标签值筛选节点。
	def generate_labels(self, label_feats, target):
		samples = []
		for i in range(target.shape[0]):
			sample = [label_feats[j] for j in range(target.shape[1]) if 1]
			if sample:
				sample = torch.stack(sample)
				samples.append(sample)
		return samples

	# 为单个样本建图；sample_feature 为 [V, H]，label_feat 为 [L, H]。
	# node_idx 是该样本在当前训练折或测试折中的行号，用它取预先计算的 KNN 邻居。
	def build_graph(self,sample_feature,label_feat,sample_features,indices_views,node_idx):
		sample_features = torch.transpose(sample_features,0,1)
		num_ins,num_label=sample_feature.shape[0],label_feat.shape[0]
		num_views = sample_feature.shape[0]
		k_neighbor = indices_views.shape[2]
		device = sample_feature.device
		"""build object graph"""
		# 视图节点先加自环，再连接相邻视图；当前两视图设置中形成双向连边。
		self_ins_edges = torch.eye(num_ins, dtype=torch.long)
		adjacency_ins = torch.zeros(num_ins, num_ins, dtype=torch.long)
		for i in range(num_ins - 1):
			adjacency_ins[i][i + 1] = 1
			adjacency_ins[i + 1][i] = 1
		ins_edges = self_ins_edges + adjacency_ins
		edge_idx_ins = torch.nonzero(ins_edges, as_tuple=False).transpose(1, 0).to(device)
		edge_attr_ins = torch.cat((sample_feature[edge_idx_ins[0, :]], sample_feature[edge_idx_ins[1, :]]), dim=1)
		# 邻居节点作为信息源，边方向是邻居 → 当前样本的对应视图节点。
		for i in range(num_ins):
			for v in range(num_views):
				if v != i % num_views:
					continue
				temp = num_ins+(k_neighbor*i)
				for m in range(k_neighbor):
					edge_idx_ins = torch.cat(
						(edge_idx_ins, torch.tensor([[(temp+m)],[v]], dtype=torch.long, device=device)), dim=1)
				neighbors_view = indices_views[v, node_idx]
				for neighbor in neighbors_view:
					neighbor_idx = neighbor
					edge_attr_ins = torch.cat((edge_attr_ins, torch.cat((sample_features[v][neighbor_idx],sample_feature[i]),dim=0).unsqueeze(0)),dim=0)
		# 记录视图数与标签数，解码时据此把跨图边恢复成每样本的预测矩阵。
		kwargs = [num_ins, num_label]
		"""build cross edge"""
		# 每个视图连接全部标签，构造 V×L 条边；边属性拼接视图特征和标签特征。
		bbox_idx_C = torch.from_numpy(np.array(list(range(num_ins))).repeat(num_label).reshape(-1, 1)).type(torch.long)
		label_idx_C = torch.from_numpy(np.array(list(range(num_label))).reshape(-1, 1)).repeat(num_ins, 1).type(torch.long)
		edge_index_C = torch.cat((bbox_idx_C, label_idx_C), dim=1).to(device)
		#[[0,0],[0,1],[0,2],[0,3],[0,4],[0,5],[1,0],[1,1],[1,2],[1,3],[1,4],[1,5]]
		edge_attr_C = torch.cat((sample_feature[edge_index_C[:,0]], label_feat[edge_index_C[:,1]]), dim=1)
		cross_edge=torch.cat((edge_index_C.type(torch.float),edge_attr_C.type(torch.float)),dim=1)

		"""build label graph"""
		# 论文式 (5) 聚合 Q-1 个其他标签：完全图，不含自环。
		edges = torch.ones(num_label, num_label, dtype=torch.long) - torch.eye(num_label, dtype=torch.long)
		edge_idx_label = torch.nonzero(edges, as_tuple=False).transpose(1, 0).to(device)
		edge_attr_label = torch.cat((label_feat[edge_idx_label[0, :]], label_feat[edge_idx_label[1, :]]), dim=1)
		return edge_idx_ins, edge_attr_ins, kwargs, edge_idx_label, edge_attr_label, cross_edge,
	# 把各样本的小图合并为批图，标签输入先取第一视图的副本。
	# 输入 sample_features 为 [B, V, H]，train_view 为当前折完整特征 [N, V, H]。
	def forward(self,sample_features,label_feats,target,args,indices_views,train_view,batch_num):
		target = target[0]  # [32,6]
		x_ins_batch, edge_idx_ins_batch, edge_attr_ins_batch, x_label_batch, edge_idx_label_batch, edge_attr_label_batch, cross_edge_batch= [], [], [], [], [], [],[]
		# 合并批图时，视图图和标签图各自维护全局节点偏移。
		node_sum = 0
		labe_sum = 0
		kwargs_ins = []
		label_feats = self.generate_labels(label_feats,target)

		reshape_sample_features = torch.transpose(sample_features,0,1)#[2,32,128]
		reshape_train_view = torch.transpose(train_view,0,1)#[2,1924,128]
		for i in range(sample_features.shape[0]):
			edge_idx_ins, edge_attr_ins, kwargs, edge_idx_label, edge_attr_label, cross_edge = self.build_graph(sample_features[i], label_feats[i], train_view,indices_views,i+((args.batch_size)*batch_num))
			# 同一样本先放视图节点，再放邻居节点；这个顺序必须与全局边索引一致。
			for view in range(sample_features.shape[1]):
				x_ins_batch.append(reshape_sample_features[view, i])
			for v in range(indices_views.shape[0]):
				for m in range(indices_views.shape[2]):
					# 跨批次行号为 i + batch_size×batch_num，需与邻居边属性取样的行号相同。
					x_ins_batch.append(reshape_train_view[v,indices_views[v,i+args.batch_size*batch_num,m]])
			edge_idx_ins_batch.append(edge_idx_ins+node_sum)
			edge_attr_ins_batch.append(edge_attr_ins)
			kwargs_ins.append(kwargs)
			x_label_batch.append(label_feats[i])
			edge_idx_label_batch.append(edge_idx_label+labe_sum)
			edge_attr_label_batch.append(edge_attr_label)
			cross_edge_batch.append(cross_edge)
			# 视图图的偏移必须包含 V 个当前节点和 V×k 个邻居节点。
			node_sum += (sample_features[i].shape[0]+(indices_views.shape[0]*indices_views.shape[2]))
			labe_sum += label_feats[i].shape[0]
		x_ins_batch = torch.stack(x_ins_batch)
		edge_idx_ins_batch = torch.cat(edge_idx_ins_batch, dim=1)
		edge_attr_ins_batch = torch.cat(edge_attr_ins_batch)
		x_label_batch = torch.cat(x_label_batch)
		edge_idx_label_batch = torch.cat(edge_idx_label_batch, dim=1)
		edge_attr_label_batch = torch.cat(edge_attr_label_batch)
		cross_edge_batch = torch.cat(cross_edge_batch)
		# Data 存储节点特征 x、边索引 edge_index、边特征 edge_attr 和额外计数信息。
		graph_ins = Data(x=x_ins_batch, edge_index=edge_idx_ins_batch,
						 edge_attr=edge_attr_ins_batch, kwargs=kwargs_ins)
		graph_label = Data(x=x_label_batch, edge_index=edge_idx_label_batch,
						   edge_attr=edge_attr_label_batch, kwargs=cross_edge_batch)
		return graph_ins, graph_label


