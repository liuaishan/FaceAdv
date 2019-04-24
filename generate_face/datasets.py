import glob
import random
import os
import numpy as np

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None, mode=[""]):
        self.transform = transforms.Compose(transforms_)

        self.files = []
        for mod in mode:
            self.files += glob.glob(os.path.join(root, "%s" % mod) + "/*.*")
        self.files = sorted(self.files)
        print("the length of files read is ", len(self.files))

    def __getitem__(self, index):
        img = self.transform(Image.open(self.files[index % len(self.files)]))
        while img.size()[0] != 3:
            index += 1
            img = self.transform(Image.open(self.files[index % len(self.files)]))
        return img

    def __len__(self):
        return len(self.files)
