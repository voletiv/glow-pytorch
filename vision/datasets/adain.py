import numpy as np
import torch

from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder


class AdaINDataset(Dataset):
    def __init__(self, data1_dir, data2_dir, transform=transforms.Compose([
                                                transforms.ToTensor()])):
        super().__init__()
        self.data1_dir = data1_dir
        self.data2_dir = data2_dir
        self.transform = transform
        self.dataset1 = ImageFolder(self.data1_dir, transform=transform)
        self.dataset2 = ImageFolder(self.data2_dir, transform=transform)
        self.len_dataset1 = len(self.dataset1)
        self.len_dataset2 = len(self.dataset2)

    def __getitem__(self, index):
        dataset1_index = index
        dataset2_index = np.random.randint(self.len_dataset2)
        x1, _ = self.dataset1.__getitem__(dataset1_index)
        x2, _ = self.dataset2.__getitem__(dataset2_index)
        # Add instance noise ~ U(0,1)
        x1 = (x1*255. + torch.rand(x1.size()))/256.
        x2 = (x2*255. + torch.rand(x2.size()))/256.
        return {
            "x1": x1,
            "x2": x2,
        }

    def __len__(self):
        return self.len_dataset1
