"""Train script.

Usage:
    train_adain_cometml.py <hparams> <dataset1_type> <dataset1_root> <dataset2_type> <dataset2_root> <name>
"""
from comet_ml import Experiment

import os
import vision
from docopt import docopt
from torchvision import transforms
from glow.builder_adain import build
from glow.config import JsonConfig
from glow.trainer_adain_cometml import Trainer


if __name__ == "__main__":
    # Args
    args = docopt(__doc__)
    hparams = args["<hparams>"]
    dataset1_type = args["<dataset1_type>"]
    dataset1_root = args["<dataset1_root>"]
    dataset2_type = args["<dataset2_type>"]
    dataset2_root = args["<dataset2_root>"]
    name = args["<name>"]

    # # Assert dataset is in vision.Datasets
    # assert dataset_name in vision.Datasets, (
    #     "`{}` is not supported, use `{}`".format(dataset_name, vision.Datasets.keys()))

    # Check
    assert os.path.exists(hparams), (
        "Failed to find hparams json `{}`".format(hparams))
    assert dataset1_type in vision.Datasets, (
        "dataset1_type `{}` is not supported, use one of `{}`".format(dataset1_type, vision.Datasets.keys()))
    assert os.path.exists(dataset1_root), (
        "Failed to find dataset1_root: `{}`".format(dataset1_root))
    assert dataset2_type in vision.Datasets, (
        "dataset2_type `{}` is not supported, use one of `{}`".format(dataset2_type, vision.Datasets.keys()))
    assert os.path.exists(dataset2_root), (
        "Failed to find dataset2_root: `{}`".format(dataset2_root))

    # Build graph
    hparams = JsonConfig(hparams)
    built = build(hparams, True)

    # Set transform of dataset
    transform = transforms.Compose([
        transforms.Resize(hparams.Data.resize),
        transforms.CenterCrop(hparams.Data.center_crop),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()])

    # Build dataset
    # Photoshop problem : https://github.com/python-pillow/Pillow/pull/3771
    adain_class = vision.Datasets['adain']
    dataset = adain_class(vision.Datasets[dataset1_type], dataset1_root,
                          vision.Datasets[dataset2_type], dataset2_root,
                          transform=transform)

    # begin to train
    trainer = Trainer(**built, dataset=dataset, hparams=hparams, name=name, dataset_name='adain')
    trainer.train()
