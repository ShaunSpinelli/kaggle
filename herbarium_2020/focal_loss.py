# --- 100 characters ------------------------------------------------------------------------------
# Created by: Shaun 2019/01/01
"""Focal Loss"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class FocalLoss(nn.Module):
	def __init__(self, alpha=0.25, gamma=2.0):
		super(FocalLoss, self).__init__()
		self.alpha = alpha
		self.gamma = gamma

	def forward(self, logits, labels):
		# probs = torch.sigmoid(logits)
		probs = F.softmax(logits)
		labels = labels.view(-1, 1)
		pt = probs.gather(1, labels) #  probablities true , get only the true probilities

		# Calculate Cross Entropy
		# Why do we not use the labels in cross entropy , it it cause we already have the probs we want ?
		log_p = pt.log()

		# Calculate Focal Loss
		batch_loss = -self.alpha * (1 - pt).pow(self.gamma) * log_p

		return batch_loss.mean()
