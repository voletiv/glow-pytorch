"""Train script.

Usage:
    train.py <hparams> <dataset_name> <dataset_root> <name>
"""
from comet_ml import Experiment

import os
import vision
from docopt import docopt
from torchvision import transforms
from glow.builder import build
from glow.config import JsonConfig
from glow.trainer_cometml import Trainer


if __name__ == "__main__":
    # Read params
    args = docopt(__doc__)
    hparams = args["<hparams>"]
    dataset_name = args["<dataset_name>"]
    dataset_root = args["<dataset_root>"]
    name = args["<name>"]

    # Checks
    assert dataset_name in vision.Datasets, (
        "`{}` is not supported, use `{}`".format(dataset_name, vision.Datasets.keys()))
    assert os.path.exists(dataset_root), (
        "Failed to find root dir `{}` of dataset.".format(dataset_root))
    assert os.path.exists(hparams), (
        "Failed to find hparams json `{}`".format(hparams))

    # Build graph
    hparams = JsonConfig(hparams)
    built = build(hparams, True)

    # Build dataset
    transform = transforms.Compose([
        transforms.Resize(hparams.Data.resize),
        transforms.CenterCrop(hparams.Data.center_crop),
        transforms.ToTensor()])
    dataset_class = vision.Datasets[dataset_name]
    dataset = dataset_class(dataset_root, transform=transform)

    # Begin to train
    trainer = Trainer(**built, dataset=dataset, hparams=hparams, name=name, dataset_name=dataset_name)
    trainer.train()
