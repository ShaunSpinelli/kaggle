# --- 100 characters ------------------------------------------------------------------------------
# Created by: Shaun 2020/04/11

import torch.nn as nn
import torchvision.models as models

NUM_CLASSES = 32094


def get_head(input_size, p=0.5):
    # NOTE: how are we initialise this
    # potential do use batch norm after each drop out
    return nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(in_features=input_size, out_features=input_size),
        nn.ReLU(),
        nn.Dropout(p=0.5),
        nn.Linear(in_features=input_size, out_features=NUM_CLASSES)
    )


def get_model(base, p=0.5, device="cpu"):
    base.fc = get_head(base.fc.in_features, p)

    if device == "gpu":
        return base.cuda()
    return base
