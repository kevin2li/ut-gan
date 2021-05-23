
import glob
import os
from typing import List

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

class ImageDataset(Dataset):
    def __init__(self, data_dirs:List[str], transforms=None, mode: str ='train', seed: int = 2021):
        super(ImageDataset, self).__init__()
        self.data_dirs = data_dirs
        self.mode = mode
        self.transforms = transforms
        self.cover_path_list = glob.glob(os.path.join(data_dirs[-1],'*.png'))
        np.random.seed(seed)
        np.random.shuffle(self.cover_path_list)

    def __getitem__(self, idx):
        cover_img = np.array(Image.open(self.cover_path_list[idx]))
        
        if self.transforms:
            data = self.transforms(cover_img)

        label = torch.tensor(0, dtype=torch.int64)

        return data, label

    def __len__(self):
        return len(self.cover_path_list)