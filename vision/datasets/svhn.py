import numpy as np
import torch

from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import SVHN


class SVHNDataset(Dataset):
    def __init__(self, root_dir, transform=transforms.Compose([
                                               transforms.ToTensor()])):
        super().__init__()
        self.root_dir = root_dir
        self.transform = transform
        self.dataset = SVHN(root_dir, transform=transform)
        self.num_classes = 10

    def __getitem__(self, index):
        x, y = self.dataset.__getitem__(index)
        # # Add instance noise ~ U(0,1)
        # x = (x*255. + torch.rand(x.size()))/256.
        y_onehot = [0.]*self.num_classes
        y_onehot[y] = 1.
        return {
            "x": x,
            "y_onehot": np.asarray(y_onehot, dtype=np.float32)
        }

    def __len__(self):
        return len(self.dataset)
