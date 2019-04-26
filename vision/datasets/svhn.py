import numpy as np
import torch

from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import SVHN


class SVHNDataset(Dataset):
    def __init__(self, root_dir, transform=transforms.Compose([
                                               transforms.ToTensor()]),
                 rot=False):
        super().__init__()
        self.root_dir = root_dir
        self.transform = transform
        self.rot = rot
        self.dataset = SVHN(root_dir, transform=transform)
        self.num_classes = 10

    def __getitem__(self, index):
        x, y = self.dataset.__getitem__(index)
        # # Add instance noise ~ U(0,1)
        # x = (x*255. + torch.rand(x.size()))/256.
        y_onehot = [0.]*self.num_classes
        y_onehot[y] = 1.
        y_onehot = np.asarray(y_onehot, dtype=np.float32)
        if self.rot:
            x_0 = x
            x_90 = x.transpose(1, 2).flip(1)
            x_180 = x.flip(1).flip(2)
            x_270 = x.transpose(1, 2).flip(2)
            x = torch.stack([x_0, x_90, x_180, x_270])
            y_rot = np.zeros((4, 4))
            y_rot[np.arange(4), np.arange(4)] = 1.
            y_onehot = y_onehot.reshape(1, -1).repeat(4, 0)
            return {
                "x": x,
                "y_rot": y_rot,
                "y_onehot": y_onehot
            }
        else:
            return {
                "x": x,
                "y_onehot": y_onehot
            }

    def __len__(self):
        return len(self.dataset)
