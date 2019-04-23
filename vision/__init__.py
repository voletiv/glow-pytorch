from .datasets import AdaINDataset, CelebADataset, ImageFolderDataset, MNISTDataset, SVHNDataset, MNIST_SVHN_Dataset, CIFAR10Dataset

Datasets = {
    "adain": AdaINDataset,
    "celeba": CelebADataset,
    "image_folder": ImageFolderDataset,
    "mnist": MNISTDataset,
    "svhn": SVHNDataset,
    "mnist_svhn": MNIST_SVHN_Dataset,
    "cifar10": CIFAR10Dataset
}
