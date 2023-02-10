from torch.utils.data import Dataset
import numpy as np
from zinc.utils import *
import json

class GraphDataset(Dataset):

	def __init__(self, args, split):
		self.args = args

		if split == 'train':
			self.dataset = json.load(open(self.args.train_file))
		elif split == 'dev':
			self.dataset = json.load(open(self.args.dev_file))
		elif split == 'test':
			self.dataset = json.load(open(self.args.test_file))

	def __len__(self):
		return len(self.dataset)

	def __getitem__(self, idx):
		graph = self.dataset[idx]
		node_num = len(graph['node_features'])
		# add self connection and a virtual node
		virtual_weight = self.args.edge_type - 1 if hasattr(self.args, 'edge_type') else 1
		adj_mat = [[i, node_num] for i in range(node_num)]  # 虚拟节点
		weight = [[1, virtual_weight] for _ in range(node_num)]  # i与虚拟虚拟节点
		adj_mat.append([i for i in range(node_num + 1)])
		weight.append([virtual_weight for i in range(node_num + 1)])
		for src, w, dst in graph['graph']:
			adj_mat[src].append(dst)
			weight[src].append(w)
			adj_mat[dst].append(src)
			weight[dst].append(w)
		node_feature = graph['node_features']
		if isinstance(node_feature[0], int):
			new_node_feature = np.zeros((len(node_feature), self.args.num_feature))
			for i in range(len(node_feature)):
				new_node_feature[i][node_feature[i]] = 1
			node_feature = new_node_feature.tolist()
		if len(node_feature[0]) < self.args.num_feature:
			zeros = np.zeros((len(node_feature), self.args.num_feature - len(node_feature[0])))
			node_feature = np.concatenate((node_feature, zeros), axis=1).tolist()
		node_feature.append(one_hot_vec(self.args.num_feature, -1)) # virtual node

		return  {
		          'node': node_feature,
		          'adj_mat': adj_mat,  # [[与i节点有邻接关系的节点序号]]
		          'weight': weight,  # [[边类型]]
		          'label': graph['targets']
		        }
