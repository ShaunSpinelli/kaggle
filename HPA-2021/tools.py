from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from torchvision import transforms
import torch
from PIL import Image


def get_one_hot(labels):
    max_classes = 18 + 1

    one_hotted_labels = np.empty((len(labels), max_classes))

    for i, label in enumerate(labels):
        lbls_idxs = list(map(int, label.split("|")))
        lbl = np.zeros(max_classes)
        lbl[lbls_idxs] = 1
        one_hotted_labels[i] = lbl

    return torch.tensor(one_hotted_labels).long()



class HPADataSet(Dataset):
    def __init__(self, image_dir, images, labels):
        self.image_dir = image_dir
        self.images =  images
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def build_image(self, img_id):
        r = np.array(Image.open(f"{self.image_dir}/{img_id}_red.png")) # mitochondria
        b = np.array(Image.open(f"{self.image_dir}/{img_id}_blue.png")) # er
        y = np.array(Image.open(f"{self.image_dir}/{img_id}_yellow.png")) # nuclei
        g = np.array(Image.open(f"{self.image_dir}/{img_id}_green.png")) # protein of interest

        img = torch.tensor(np.stack([r,b,g])/255, dtype=torch.float)

        return img


    def __getitem__(self, idx):
        image_id = self.images[idx]
        label = self.labels[idx] #NOTE: this will prob have to be one hot
        image = self.build_image(image_id)

        return image, label

    @classmethod
    def from_csv(cls, csv, image_dir):
        df = pd.read_csv(csv)
        images_id = df["ID"].to_numpy()
        one_hot_labels = get_one_hot(df["Label"])

        return cls(image_dir, images_id, one_hot_labels )


