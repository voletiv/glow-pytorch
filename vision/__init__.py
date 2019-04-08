from .datasets import CelebADataset, MNISTDataset, SVHNDataset, MNIST_SVHN_Dataset

Datasets = {
    "celeba": CelebADataset,
    "mnist": MNISTDataset,
    "svhn": SVHNDataset,
    "mnist_svhn": MNIST_SVHN_Dataset
}
