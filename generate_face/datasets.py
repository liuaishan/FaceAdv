import glob
import random
import os
import numpy as np

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms


"""
   this file is to load the dataset
"""
class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None, mode=[""]):
        self.transform = transforms.Compose(transforms_)

        # load all files in to a list
        self.files = []
        for mod in mode:
            self.files += glob.glob(os.path.join(root, "%s" % mod) + "/*.*")
        self.files = sorted(self.files)

    def __getitem__(self, index):
        """
          @index: which picture will be loaded
        """
        img = self.transform(Image.open(self.files[index % len(self.files)]))
        
        # this is some mistake in the ImageNet dataset, which is gray
        # so when load a gray picture and exchange it to next RGB pic
        while img.size()[0] != 3:
            index += 1
            img = self.transform(Image.open(self.files[index % len(self.files)]))
        return img

    def __len__(self):
        return len(self.files)
