# --- 100 characters ------------------------------------------------------------------------------
# Created by: Shaun 2020/04/11

import torch.nn as nn
import torchvision.models as models

NUM_CLASSES = 32093


class Identity(nn.Module):
    def __init__(self, in_features):
        super(Identity, self).__init__()
        self.in_features = in_features

    def forward(self, x):
        return x


def get_head(input_size, p1=0.5, p2=0.5):
    # NOTE: how are we initialise this
    # potential do use batch norm after each drop out
    return nn.Sequential(
        nn.Dropout(p=p1),
        # nn.BatchNorm1d(input_size),
        nn.Linear(in_features=input_size, out_features=input_size * 2),
        nn.ReLU(),
        nn.Dropout(p=p2),
        # nn.BatchNorm1d(input_size * 2),
        nn.Linear(in_features=input_size * 2, out_features=NUM_CLASSES)
    )


def get_model(base, p1=0.5, p2=0.5, device="cpu"):
    base.fc = get_head(base.fc.in_features, p1, p2)

    if device == "gpu":
        return base.cuda()
    return base

def get_triplet_model(base, p1=0.5, p2=0.5, device="cpu"):
    base.fc = Identity(base.fc.in_features)
    model = TripletsModel(base, p1=p1, p2=p2)

    if device == "gpu":
        return model.cuda()
    return model

class TripletsModel(nn.Module):
    """Model fro triplet and classifies training"""

    def __init__(self, base,  p1=0.5, p2=0.5):
        super(TripletsModel, self).__init__()
        self.base = base
        self.embeddings = nn.Sequential(
            nn.Dropout(p=p1),
            nn.BatchNorm1d(base.fc.in_features),
            nn.Linear(in_features=base.fc.in_features, out_features=base.fc.in_features * 2),
            nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(p=p2),
            nn.BatchNorm1d(base.fc.in_features * 2),
            nn.Linear(in_features=base.fc.in_features * 2, out_features=NUM_CLASSES)
        )

    def forward(self, input):
        features = self.base(input)
        embeddings = self.embeddings(features)
        logits = self.classifier(embeddings)
        if self.training:
            return embeddings, logits
        return logits



