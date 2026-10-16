import numpy as np
import torch
import scipy.io as scio
from sklearn.metrics.pairwise import pairwise_distances

from torch_geometric.data import Data,Dataset,DataLoader
from torch_geometric.nn import MetaLayer
import itertools




class Graph_Generator(torch.nn.Module):
	def __init__(self):
		super(Graph_Generator, self).__init__()

	def build_graph(self, train_feats, label_feats, target, k):
		num_view, num_ins, num_label=len(train_feats), target.shape[0], label_feats.shape[0]  #5   #20

		"""build instance graph 同一视图下不同实例之间的图"""
		view_graph_list = []
		for i in range(num_view):
			view_feats = train_feats[i] #(5011, 100)
			data_dist = pairwise_distances(view_feats, metric='euclidean') #(5011, 5011)
			topk_dist, topk_index = torch.from_numpy(data_dist).topk(dim=1, k=k, largest=False, sorted=True) #(5011,5)

			knn_idx=topk_index.reshape(-1, 1).type(torch.long) #5011x5=25055, 1
			bbox_idx=torch.from_numpy(np.array(list(range(num_ins))).repeat(k).reshape(-1,1)).type(torch.long) #(25055,1)
			edge_idx_ins=torch.cat((bbox_idx, knn_idx),dim=1).transpose(1,0) #(2,25055)
			# edge_attr_ins=torch.cat((view_feats[edge_idx_ins[0,:]],view_feats[edge_idx_ins[1,:]]),dim=1) # (25055,200)

			graph_ins = Data(x=view_feats, edge_index=edge_idx_ins, kwargs=[num_ins, num_label])
			view_graph_list.append(graph_ins)

		"""build instance graph 不同视图下同一实例之间的图"""
		ins_idx_C = torch.from_numpy(np.array(list(range(num_ins))).repeat(num_view).reshape(-1, 1)).type(torch.long)  # （25055,1） 0,0,0,0, 40个0,0,0,1,1,1,40个1,1,1...
		view_idx_C = torch.from_numpy(np.array(list(range(num_view))).reshape(-1, 1)).repeat(num_ins, 1).type(torch.long)  # （25055，1）0, 1, 2, ..., 19, 0, 1,2,...,19,1,2,..
		view_edge_index_C = torch.cat((ins_idx_C, view_idx_C), dim=1).cuda()  # (25055, 2)

		"""build global-inst-label cross edge"""
		gins_idx_C = torch.from_numpy(np.array(list(range(num_ins))).repeat(num_label).reshape(-1, 1)).type(torch.long)  # （800,1） 0,0,0,0, 40个0,0,0,1,1,1,40个1,1,1...
		label_idx_C = torch.from_numpy(np.array(list(range(num_label))).reshape(-1, 1)).repeat(num_ins, 1).type(torch.long)  # （800，1）1, 2, ..., 19, 1,2,...,19,1,2,..
		gedge_index_C = torch.cat((gins_idx_C, label_idx_C), dim=1).cuda()  # (800, 2)

		"""build label graph"""
		if torch.cuda.is_available():
			graph_label = Data(x=label_feats.cuda(), kwargs1=view_edge_index_C.cuda(), kwargs2=gedge_index_C.cuda())
		else:
			graph_label = Data(x=label_feats, kwargs1=view_edge_index_C, kwargs2=gedge_index_C)

		return view_graph_list, graph_label


	def forward(self,ins_feats,label_feats, target,k):
		graph_ins,graph_label=self.build_graph(ins_feats,label_feats, target, k)

		return graph_ins,graph_label

class GraphData(Dataset):
	def __init__(self):
		super(GraphData, self).__init__()
		self.graph_generator=Graph_Generator()
	def __len__(self):
		'Denotes the total number of samples'
		return 10000

	def __getitem__(self, index):
		'Generates one sample of data'
		dataset = scio.loadmat('N:/Dataset/mat/lyu_data/lost.mat')

		ins_feature = torch.from_numpy(dataset['data'])
		partial_target = torch.from_numpy(dataset['partial_target'])
		print("-" * 66)
		graph_ins, graph_label, graph_cross = self.graph_generator(ins_feature, partial_target)
		return graph_ins,graph_label,graph_cross

# dataset=GraphData()
# loader=DataLoader(dataset,batch_size=2)
#
# for idx ,data in enumerate(loader):
# 	graph_ins,graph_label,graph_cross=data
# 	print(graph_ins["x"])
# 	print(graph_ins["edge_index"])
# 	print(graph_ins["edge_attr"])
# 	print("*"*66)
# 	print(graph_label["x"])
# 	print(graph_label["edge_index"])
# 	print(graph_label["edge_attr"])
# 	print("*" * 66)
# 	print(graph_cross["x"])
# 	print(graph_cross["edge_index"])
# 	print(graph_cross["edge_attr"])
# 	print("*" * 66)
# 	input()

# print(bbox_info)
# print(lable_feats)
# print("-"*66)
# for graph in graph_generator(bbox_info,lable_feats):
# 	print(graph["x"])
# 	print(graph["edge_index"])
# 	print(graph["edge_attr"])
# 	print("*" * 66)
