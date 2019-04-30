"""Train script.

Usage:
    train_adain_cometml.py <hparams> <content_dataset_type> <content_dataset_dir> <style_dataset_type> <style_dataset_dir> <name>
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
    ds_c_type = args["<content_dataset_type>"]
    ds_c_root = args["<content_dataset_dir>"]
    ds_s_type = args["<style_dataset_type>"]
    ds_s_root = args["<style_dataset_dir>"]
    name = args["<name>"]

    # # Assert dataset is in vision.Datasets
    # assert dataset_name in vision.Datasets, (
    #     "`{}` is not supported, use `{}`".format(dataset_name, vision.Datasets.keys()))

    # Check
    assert os.path.exists(hparams), (
        "Failed to find hparams json `{}`".format(hparams))
    assert ds_c_type in vision.Datasets, (
        "ds_c_type `{}` is not supported, use one of `{}`".format(ds_c_type, vision.Datasets.keys()))
    assert os.path.exists(ds_c_root), (
        "Failed to find dataset_content_root: `{}`".format(ds_c_root))
    assert ds_s_type in vision.Datasets, (
        "ds_c_type `{}` is not supported, use one of `{}`".format(ds_s_type, vision.Datasets.keys()))
    assert os.path.exists(ds_s_root), (
        "Failed to find dataset_style_root: `{}`".format(ds_s_root))

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
    ds_c = vision.Datasets[ds_c_type](ds_c_root, transform=transform)
    ds_s = vision.Datasets[ds_s_type](ds_s_root, transform=transform)

    # begin to train
    trainer = Trainer(**built, content_dataset=ds_c, style_dataset=ds_s, hparams=hparams, name=name, dataset_name='adain')
    trainer.train()
