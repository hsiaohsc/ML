import numpy as np
from typing import List, Set

from classifier import Classifier
from decision_stump import DecisionStump
from abc import abstractmethod

class Boosting(Classifier):
  # Boosting from pre-defined classifiers
	def __init__(self, clfs: Set[Classifier], T=0):
		self.clfs = clfs      # set of weak classifiers to be considered
		self.num_clf = len(clfs)
		if T < 1:
			self.T = self.num_clf
		else:
			self.T = T
	
		self.clfs_picked = [] # list of classifiers h_t for t=0,...,T-1
		self.betas = []       # list of weights beta_t for t=0,...,T-1
		return

	@abstractmethod
	def train(self, features: List[List[float]], labels: List[int]):
		return

	def predict(self, features: List[List[float]]) -> List[int]:
		'''
		Inputs:
		- features: the features of all test examples
   
		Returns:
		- the prediction (-1 or +1) for each example (in a list)
		'''
		########################################################
		# TODO: implement "predict"
		########################################################
		Ht = np.zeros(len(features))
		H=[]
		for i in range(self.T):
			Ht+=self.betas[i]*np.array(self.clfs_picked[i].predict(features))
		for i in range(len(Ht)):
			if Ht[i]<=0:
				H.append(-1)
			else:
				H.append(1)
		return H

class AdaBoost(Boosting):
	def __init__(self, clfs: Set[Classifier], T=0):
		Boosting.__init__(self, clfs, T)
		self.clf_name = "AdaBoost"
		return
		
	def train(self, features: List[List[float]], labels: List[int]):
		'''
		Inputs:
		- features: the features of all examples
		- labels: the label of all examples
   
		Require:
		- store what you learn in self.clfs_picked and self.betas
		'''
		############################################################
		# TODO: implement "train"
		############################################################
		N = len(labels)
		w = np.full(N,1/N)
		for t in range(self.T):
			epsilon = float("inf")
			for i in range(len(self.clfs)):
				h = self.clfs[i].predict(features)
				error = np.sum(w * (np.array(labels) != np.array(h)))
				if error < epsilon:
					ht = self.clfs[i]
					epsilon = error
					htx = h
			self.clfs_picked.append(ht)

			beta = 1 / 2 * np.log((1 - epsilon) / epsilon)
			self.betas.append(beta)

			for n in range(N):
				if labels[n] == htx[n]:
					w[n] *= np.exp(-beta)
				else:
					w[n] *= np.exp(beta)

			w /= np.sum(w)
		
	def predict(self, features: List[List[float]]) -> List[int]:
		return Boosting.predict(self, features)



	