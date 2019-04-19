import numpy as np
from typing import List
from classifier import Classifier

class DecisionTree(Classifier):
	def __init__(self):
		self.clf_name = "DecisionTree"
		self.root_node = None

	def train(self, features: List[List[float]], labels: List[int]):
		# init.
		assert(len(features) > 0)
		self.feautre_dim = len(features[0])
		num_cls = np.max(labels)+1

		# build the tree
		self.root_node = TreeNode(features, labels, num_cls)
		if self.root_node.splittable:
			self.root_node.split()

		return
		
	def predict(self, features: List[List[float]]) -> List[int]:
		y_pred = []
		for feature in features:
			y_pred.append(self.root_node.predict(feature))
		return y_pred

	def print_tree(self, node=None, name='node 0', indent=''):
		if node is None:
			node = self.root_node
		print(name + '{')
		
		string = ''
		for idx_cls in range(node.num_cls):
			string += str(node.labels.count(idx_cls)) + ' '
		print(indent + ' num of sample / cls: ' + string)

		if node.splittable:
			print(indent + '  split by dim {:d}'.format(node.dim_split))
			for idx_child, child in enumerate(node.children):
				self.print_tree(node=child, name= '  '+name+'/'+str(idx_child), indent=indent+'  ')
		else:
			print(indent + '  cls', node.cls_max)
		print(indent+'}')


class TreeNode(object):
	def __init__(self, features: List[List[float]], labels: List[int], num_cls: int):
		self.features = features
		self.labels = labels
		self.children = []
		self.num_cls = num_cls

		count_max = 0
		for label in np.unique(labels):
			if self.labels.count(label) > count_max:
				count_max = labels.count(label)
				self.cls_max = label # majority of current node

		if len(np.unique(labels)) < 2:
			self.splittable = False
		else:
			self.splittable = True

		self.dim_split = None # the index of the feature to be split

		self.feature_uniq_split = None # the possible unique values of the feature to be split


	def split(self):
		def conditional_entropy(branches: List[List[int]]) -> float:
			'''
			branches: C x B array, 
					  C is the number of classes,
					  B is the number of branches
					  it stores the number of 
					  corresponding training samples 
					  e.g.
					              ○ ○ ○ ○
					              ● ● ● ●
					            ┏━━━━┻━━━━┓
				               ○ ○       ○ ○
				               ● ● ● ●
				               
				      branches = [[2,2], [4,0]]
			'''
			########################################################
			# TODO: compute the conditional entropy
			########################################################
			branches = np.array(branches)
			leaftotal = np.sum(branches, axis=0)	#[1x2]
			layertotal = np.sum(leaftotal)	#scalar
			leafweight = leaftotal / layertotal
			P = branches / leaftotal
			entropy=np.zeros((P.shape[0],P.shape[1]))
			for i in range(P.shape[0]):
				for j in range(P.shape[1]):
					if P[i][j]>0:
						entropy[i][j]=-P[i][j]*np.log2(P[i][j])
			entropy = np.sum(entropy, axis=0)
			entropy = np.sum(entropy * leafweight)
			return entropy
		
		for idx_dim in range(len(self.features[0])):
		############################################################
		# TODO: compare each split using conditional entropy
		#       find the best split
		############################################################
			if not 'min_entropy' in locals():
				min_entropy = float('inf')
			xi = np.array(self.features)[:, idx_dim]
			if None in xi:
				continue
			branch_values = np.unique(xi)
			branches = np.zeros((self.num_cls, len(branch_values)))
			for i, val in enumerate(branch_values):
				y = np.array(self.labels)[np.where(xi == val)]
				for j in range(len(y)):
					branches[y[j], i] += 1
			entropy = conditional_entropy(branches)
			if entropy < min_entropy:
				min_entropy = entropy
				self.dim_split = idx_dim
				self.feature_uniq_split = branch_values.tolist()


		############################################################
		# TODO: split the node, add child nodes
		############################################################
		xi = np.array(self.features)[:, self.dim_split]
		x = np.array(self.features, dtype=object)
		x[:, self.dim_split] = None
		for val in self.feature_uniq_split:
			indexes = np.where(xi == val)
			x_new = x[indexes].tolist()
			y_new = np.array(self.labels)[indexes].tolist()
			child = TreeNode(x_new, y_new, self.num_cls)
			if np.array(x_new).size == 0 or all(v is None for v in x_new[0]):
				child.splittable = False
			self.children.append(child)



		# split the child nodes
		for child in self.children:
			if child.splittable:
				child.split()

		return

	def predict(self, feature: List[int]) -> int:
		if self.splittable:
			# print(feature)
			idx_child = self.feature_uniq_split.index(feature[self.dim_split])
			return self.children[idx_child].predict(feature)
		else:
			return self.cls_max



