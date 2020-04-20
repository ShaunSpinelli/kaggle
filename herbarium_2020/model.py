# --- 100 characters ------------------------------------------------------------------------------
# Created by: Shaun 2020/04/11

import torch.nn as nn
import torchvision.models as models

NUM_CLASSES = 32093


def get_head(input_size, p1=0.5, p2=0.5):
    # NOTE: how are we initialise this
    # potential do use batch norm after each drop out
    return nn.Sequential(
        nn.Dropout(p=p1),
        nn.Linear(in_features=input_size, out_features=input_size//2),
        nn.ReLU(),
        nn.Dropout(p=p2),
        nn.Linear(in_features=input_size//2, out_features=NUM_CLASSES)
    )


def get_model(base, p1=0.5, p2=0.5, device="cpu"):
    base.fc = get_head(base.fc.in_features, p1, p2)

    if device == "gpu":
        return base.cuda()
    return base
