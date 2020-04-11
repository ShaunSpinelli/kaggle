# --- 100 characters ------------------------------------------------------------------------------
# Created by: Shaun 2019/01/01

import numpy as np
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch

from PIL import Image


class HerbDataSet(Dataset):
    def __init__(self, data_frame, location, size, x_name="file_name", y_name="label"):
        """

        Args:
            data_frame (Dataframe): Dataframe containing label and data
            location (str): data location directory
            size (int): size of image
            x_name (str): column name of file
            y_name (str): column name of label
        """
        self.df = data_frame
        self.location = location
        self.size = (size, size)
        self.x_name = x_name
        self.y_name = y_name

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        item = self.df.iloc[idx]
        image = np.array(Image.open("{}/{}".format(self.location, item[self.x_name])).resize(self.size)).reshape((3, *self.size))
        image_tensor = torch.tensor(image/255, dtype=torch.float).cuda()
        return image_tensor, torch.tensor(item[self.y_name]).cuda(), item[self.y_name]

