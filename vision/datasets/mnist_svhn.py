import numpy as np
import os
import torch

from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import SVHN


class MNIST_SVHN_Dataset(Dataset):
    def __init__(self, root_dir, transform=transforms.Compose([
                                               transforms.ToTensor()])):
        super().__init__()
        self.root_dir = root_dir
        # Transform is ignored
        self.transform = transform
        # MNIST_transform
        self.mnist_transform = transforms.Compose([
            transforms.CenterCrop(28),
            transforms.Resize(32),
            transforms.ToTensor()])
        self.svhn_transform = transforms.Compose([
            transforms.CenterCrop(32),
            transforms.Resize(32),
            transforms.ToTensor()])
        self.mnist_dataset = MNIST(root_dir, transform=mnist_transform)
        self.svhn_dataset = SVHN(os.path.join(root_dir, "SVHN"), transform=svhn_transform)

    def __getitem__(self, index):
        y_onehot = [0.]*2
        y_class_onehot = [0.]*10
        if np.random.sample() > 0.5:
            x, y = self.mnist_dataset.__getitem__(index)
            y_onehot[0] = 1.
        else:
            x, y = self.svhn_dataset.__getitem__(index)
            y_onehot[1] = 1.
        y_class_onehot[y] = 1.
        return {
            "x": x,
            "y_onehot": np.asarray(y_onehot, dtype=np.float32)
            "y_class_onehot": np.asarray(y_class_onehot, dtype=np.float32)
        }

    def __len__(self):
        return len(self.mnist_dataset) + len(self.svhn_dataset)
