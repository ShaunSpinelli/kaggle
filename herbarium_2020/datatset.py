# --- 100 characters ------------------------------------------------------------------------------
# Created by: Shaun 2019/01/01

import numpy as np
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torchvision import transforms
import torch

import imgaug.augmenters as iaa

from PIL import Image

augment = iaa.Sequential([
    iaa.Affine(rotate=(-45, 45)),
])




class HerbDataSet(Dataset):
    def __init__(self, data_frame, location, size, file_name="file_name", label="label"):
        """

        Args:
            data_frame (Dataframe): Dataframe containing label and data
            location (str): data location directory
            size (int): size of image
            x_name (str): column name of file
            label (str): column name of label
        """
        self.df = data_frame
        self.location = location
        self.size = (size, size)
        self.file_name = file_name
        self.label = label

        self.normalise = transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Taken from torchvision
                                              std=[0.229, 0.224, 0.225])

    def __len__(self):
        return len(self.df)

    def load_image(self, item):
        """Loads, reshapes and converts images from file name to np.array"""
        return np.array(Image.open("{}/{}".format(self.location, item[self.file_name])).resize(self.size)).reshape((3, *self.size))

    def process_images(self, image):
        """Normalises and converts to tensor"""
        return self.normalise(torch.tensor(image/255, dtype=torch.float)).cuda()

    def __getitem__(self, idx):
        item = self.df.iloc[idx]
        image = self.load_image(item)
        if item.augment:
            image = augment(images=image)

        label = torch.tensor(item[self.label]).cuda()
        image_tensor = self.process_images(image)

        return image_tensor, label


class HerbTripletDataSet(Dataset):
    def __init__(self, data_frame, location, size, file_name="file_name", label="label"):
        """Triplet Herbarvarium Dataset

        Args:
            data_frame (Dataframe): Dataframe containing label and data
            location (str): data location directory
            size (int): size of image
            x_name (str): column name of file
            label (str): column name of label
        """
        self.df = data_frame
        self.location = location
        self.size = (size, size)
        self.file_name = file_name
        self.label = label

        self.normalise = transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Taken from torchvision
                                              std=[0.229, 0.224, 0.225])

    def __len__(self):
        return len(self.df)

    def load_image(self, item):
        """Loads, reshapes and converts images from file name to np.array"""
        return np.array(Image.open("{}/{}".format(self.location, item[self.file_name])).resize(self.size)).reshape((3, *self.size))

    def _process_images(self, image):
        """Normalises and converts to tensor"""
        return self.normalise(torch.tensor(image/255, dtype=torch.float)).cuda()



    def __getitem__(self, idx):
        item = self.df.iloc[idx]
        image = 1
        label = torch.tensor(item[self.label]).cuda()
        image_tensor = 1
        return image_tensor, image_tensor, image_tensor
