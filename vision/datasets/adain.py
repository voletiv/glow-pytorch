import numpy as np
import torch

from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder


class AdaINDataset(Dataset):
    def __init__(self, content_data_class, content_data_dir, style_data_class, style_data_dir,
                 transform=transforms.Compose([transforms.ToTensor()])):
        super().__init__()
        self.data_c_class = content_data_class
        self.data_c_dir = content_data_dir
        self.data_s_class = style_data_class
        self.data_s_dir = style_data_dir
        self.transform = transform
        self.ds_c = self.data_c_class(self.data_c_dir, transform=transform)
        self.ds_s = self.data_s_class(self.data_s_dir, transform=transform)
        self.ds_c_len = len(self.ds_c)
        self.ds_s_len = len(self.ds_s)

    def __getitem__(self, index):
        ds_c_index = index
        ds_s_index = np.random.randint(self.ds_s_len)
        data_c = self.ds_c.__getitem__(ds_c_index)
        data_s = self.ds_s.__getitem__(ds_s_index)
        # Content
        if isinstance(self.data_c_class, ImageFolder):
            x_c, y_c = data_c
            y_onehot_c = [0.]*self.ds_c.num_classes
            y_onehot_c[y_c] = 1.
            y_onehot_c = np.asarray(y_onehot_c, dtype=np.float32)
        else:
            x_c = data_c["x"]
            y_onehot_c = data_c["y_onehot"]
        # Style
        if isinstance(self.data_s_class, ImageFolder):
            x_s, y_s = data_s
            y_onehot_s = [0.]*self.ds_s.num_classes
            y_onehot_s[y_s] = 1.
            y_onehot_s = np.asarray(y_onehot_s, dtype=np.float32)
        else:
            x_s = data_s["x"]
            y_onehot_s = data_s["y_onehot"]
        # Return
        return {
            "x_c": x_c,
            "x_s": x_s,
            "y_onehot_c": y_onehot_c,
            "y_onehot_s": y_onehot_s
        }

    def __len__(self):
        return self.ds_c_len
