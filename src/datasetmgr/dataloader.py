from typing import List
import numpy as np
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader
from .dataset import ImageDataset

class AugData():
    def __call__(self, data):
        # Rotation
        rot = np.random.randint(0, 3)
        data = np.rot90(data, rot, axes=[1, 2]).copy()

        # Mirroring
        if np.random.random() < 0.5:
            data = np.flip(data, axis=2).copy()

        return data

class ToTensor():
    def __call__(self, data):
        data = data.astype(np.float32)
        # data = np.expand_dims(data, 1)
        data = data / 255.0
        return torch.from_numpy(data)

train_transform = T.Compose([
    # AugData(),
    ToTensor(),
])

eval_transform = T.Compose([
    ToTensor()
])

def get_dataloader(data_dirs: List[str], batch_size: int = 32):
    dataset = ImageDataset(data_dirs=data_dirs, transforms=train_transform)
    dataloader = DataLoader(dataset, batch_size=batch_size)
    return dataloader