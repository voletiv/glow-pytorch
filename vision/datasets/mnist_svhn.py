import numpy as np
import os
import torch

from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import MNIST, SVHN


class MNIST_SVHN_Dataset(Dataset):
    def __init__(self, root_dir, transform=transforms.Compose([
                                               transforms.ToTensor()]),
                 rot=False):
        super().__init__()
        self.root_dir = root_dir
        # Transform is ignored
        self.transform = transform
        self.rot = rot
        # MNIST
        self.mnist_transform = transforms.Compose([
            transforms.CenterCrop(28),
            transforms.Resize(32),
            transforms.ToTensor()])
        self.mnist_dataset = MNIST(root_dir, transform=self.mnist_transform)
        self.mnist_len = len(self.mnist_dataset)
        # SVHN
        self.svhn_transform = transforms.Compose([
            transforms.CenterCrop(32),
            transforms.Resize(32),
            transforms.ToTensor()])
        self.svhn_dataset = SVHN(os.path.join(root_dir, "SVHN"), transform=self.svhn_transform)
        self.svhn_len = len(self.svhn_dataset)

    def __getitem__(self, index):
        y_onehot = [0.]*2
        y_class_onehot = [0.]*10
        if np.random.sample() < 0.3:
            x, y = self.mnist_dataset.__getitem__(index % self.mnist_len)
            x = torch.cat((x, x, x), dim=0)
            y_onehot[0] = 1.
        else:
            x, y = self.svhn_dataset.__getitem__(index % self.svhn_len)
            y_onehot[1] = 1.
        # # Add instance noise ~ U(0,1)
        # x = (x*255. + torch.rand(x.size()))/256.
        y_class_onehot[y] = 1.
        y_onehot = np.asarray(y_onehot, dtype=np.float32)
        y_class_onehot = np.asarray(y_class_onehot, dtype=np.float32)
        if self.rot:
            x_0 = x
            x_90 = x.transpose(1, 2).flip(1)
            x_180 = x.flip(1).flip(2)
            x_270 = x.transpose(1, 2).flip(2)
            x = torch.stack([x_0, x_90, x_180, x_270])
            y_rot = np.zeros((4, 4))
            y_rot[np.arange(4), np.arange(4)] = 1.
            y_onehot = y_onehot.reshape(1, -1).repeat(4, 0)
            y_class_onehot = y_class_onehot.reshape(1, -1).repeat(4, 0)
            return {
                "x": x,
                "y_rot": y_rot,
                "y_onehot": y_onehot,
                "y_class_onehot": y_class_onehot
            }
        else:
            return {
                "x": x,
                "y_onehot": y_onehot,
                "y_class_onehot": y_class_onehot
            }

    def __len__(self):
        return max([self.mnist_len, self.svhn_len])
