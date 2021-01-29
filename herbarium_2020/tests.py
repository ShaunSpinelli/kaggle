# --- 100 characters ------------------------------------------------------------------------------
# Created by: Shaun 2019/01/01

import torch
import triplet_loss as fl

embeddings = torch.randn((4, 256, 256))
dist = fl._pairwise_distances(embeddings)
print(dist)
