"""Train script.

Usage:
    train.py <hparams> <dataset1_root> <dataset2_root> <name>
"""
from comet_ml import Experiment

import os
import vision
from docopt import docopt
from torchvision import transforms
from glow.builder import build
from glow.config import JsonConfig
from glow.trainer_adain_cometml import Trainer


if __name__ == "__main__":
    # Args
    args = docopt(__doc__)
    hparams = args["<hparams>"]
    dataset1_root = args["<dataset1_root>"]
    dataset2_root = args["<dataset2_root>"]
    name = args["<name>"]

    # # Assert dataset is in vision.Datasets
    # assert dataset_name in vision.Datasets, (
    #     "`{}` is not supported, use `{}`".format(dataset_name, vision.Datasets.keys()))

    # Check
    assert os.path.exists(hparams), (
        "Failed to find hparams json `{}`".format(hparams))
    assert os.path.exists(dataset1_root), (
        "Failed to find root dir of dataset1: `{}`".format(dataset1_root))
    assert os.path.exists(dataset1_root), (
        "Failed to find root dir of dataset2: `{}`".format(dataset2_root))

    # Build graph
    hparams = JsonConfig(hparams)
    built = build(hparams, True)

    # Set transform of dataset
    transform = transforms.Compose([
        transforms.Resize(hparams.Data.resize),
        transforms.CenterCrop(hparams.Data.center_crop),
        transforms.ToTensor()])

    # Build dataset
    dataset_class = vision.Datasets['adain']
    dataset = dataset_class(dataset1_root, dataset2_root, transform=transform)

    # begin to train
    trainer = Trainer(**built, dataset=dataset, hparams=hparams, name=name, dataset_name='adain')
    trainer.train()
