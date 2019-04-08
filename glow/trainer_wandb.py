import re
import os
import torch
import torch.nn.functional as F
import datetime
import numpy as np
import subprocess

from tqdm import tqdm
# from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from .utils import save, load, plot_prob
from .config import JsonConfig
from .models import Glow
from . import thops

import wandb


class Trainer(object):
    def __init__(self, graph, optim, lrschedule, loaded_step,
                 devices, data_device,
                 dataset, hparams, name, dataset_name):
        self.hparams = hparams
        self.name = name
        if isinstance(hparams, str):
            hparams = JsonConfig(hparams)
        # set members
        # append date info
        date = str(datetime.datetime.now())
        self.date = date[:date.rfind(".")].replace("-", "")\
                                     .replace(":", "")\
                                     .replace(" ", "_")
        self.log_dir = os.path.join(hparams.Dir.log_root, self.date + "_" + self.name)
        self.checkpoints_dir = os.path.join(self.log_dir, "checkpoints")
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        # write hparams
        hparams.dump(self.log_dir)
        if not os.path.exists(self.checkpoints_dir):
            os.makedirs(self.checkpoints_dir)
        # Checkpoints
        self.checkpoints_gap = hparams.Train.checkpoints_gap
        self.max_checkpoints = hparams.Train.max_checkpoints
        # model relative
        self.graph = graph
        self.optim = optim
        self.weight_y = hparams.Train.weight_y
        # grad operation
        self.max_grad_clip = hparams.Train.max_grad_clip
        self.max_grad_norm = hparams.Train.max_grad_norm
        # copy devices from built graph
        self.devices = devices
        self.data_device = data_device
        # number of training batches
        self.batch_size = hparams.Train.batch_size
        self.data_loader = DataLoader(dataset,
                                      batch_size=self.batch_size,
                                    #   num_workers=8,
                                      shuffle=True,
                                      drop_last=True)
        self.n_epoches = (hparams.Train.num_batches+len(self.data_loader)-1) // len(self.data_loader)
        self.global_step = 0
        # lr schedule
        self.lrschedule = lrschedule
        self.loaded_step = loaded_step
        # data relative
        self.y_classes = hparams.Glow.y_classes
        self.y_condition = hparams.Glow.y_condition
        self.y_criterion = hparams.Criterion.y_condition
        # Checkpoints
        assert self.y_criterion in ["multi-classes", "single-class"]

        # log relative
        # tensorboard
        # self.writer = SummaryWriter(log_dir=self.log_dir)
        self.scalar_log_gaps = hparams.Train.scalar_log_gap
        self.plot_gaps = hparams.Train.plot_gap
        self.inference_gap = hparams.Train.inference_gap
        self.n_image_samples = hparams.Train.n_image_samples

    def hparams_dict(self):
        hparams_dict = {}
        hparams = self.hparams.to_dict()
        for key in hparams:
            for in_key in hparams[key]:
                hparams_dict[in_key] = hparams[key][in_key]
                if isinstance(hparams[key][in_key], dict):
                    for in_in_key in hparams[key][in_key]:
                        hparams_dict[in_in_key] = hparams[key][in_key][in_in_key]
        # Also
        hparams_dict['name'] = self.name
        hparams_dict['date'] = self.date
        hparams_dict['run_name'] = self.date + "_" + self.name
        hparams_dict['log_dir'] = self.log_dir
        hparams_dict['n_epoches'] = self.n_epoches
        return hparams_dict

    def train(self):

        # wandb
        wandb.init(project="glow-mnist")
        wandb.run.description = self.date + "_" + self.name
        wandb.run.save()
        hparams_dict = self.hparams_dict()
        wandb.config.update(hparams_dict)
        wandb.watch(self.graph)

        # set to training state
        self.graph.train()
        self.global_step = self.loaded_step

        # begin to train
        for epoch in range(self.n_epoches):
            print("epoch", epoch)
            progress = tqdm(self.data_loader)
            for i_batch, batch in enumerate(progress):

                # update learning rate
                lr = self.lrschedule["func"](global_step=self.global_step,
                                             **self.lrschedule["args"])
                for param_group in self.optim.param_groups:
                    param_group['lr'] = lr
                self.optim.zero_grad()

                # log
                if self.global_step % self.scalar_log_gaps == 0:
                    # self.writer.add_scalar("lr/lr", lr, self.global_step)
                    wandb.log({"lr": lr, "epoch": epoch+i_batch/len(self.data_loader)}, step=self.global_step)

                # get batch data
                for k in batch:
                    batch[k] = batch[k].to(self.data_device)
                x = batch["x"]
                y = None
                y_onehot = None
                if self.y_condition:
                    if self.y_criterion == "multi-classes":
                        assert "y_onehot" in batch, "multi-classes ask for `y_onehot` (torch.FloatTensor onehot)"
                        y_onehot = batch["y_onehot"]
                    elif self.y_criterion == "single-class":
                        assert "y" in batch, "single-class ask for `y` (torch.LongTensor indexes)"
                        y = batch["y"]
                        y_onehot = thops.onehot(y, num_classes=self.y_classes)

                # at first time, initialize ActNorm
                if self.global_step == 0:
                    self.graph(x[:self.batch_size // len(self.devices), ...],
                               y_onehot[:self.batch_size // len(self.devices), ...] if y_onehot is not None else None)

                # parallel
                if len(self.devices) > 1 and not hasattr(self.graph, "module"):
                    print("[Parallel] move to {}".format(self.devices))
                    self.graph = torch.nn.parallel.DataParallel(self.graph, self.devices, self.devices[0])

                # forward phase
                z, nll, y_logits = self.graph(x=x, y_onehot=y_onehot)

                # loss_generative
                loss_generative = Glow.loss_generative(nll)

                # loss_classes
                loss_classes = 0
                if self.y_condition:
                    loss_classes = (Glow.loss_multi_classes(y_logits, y_onehot)
                                    if self.y_criterion == "multi-classes" else
                                    Glow.loss_class(y_logits, y))

                # total loss
                loss = loss_generative + loss_classes * self.weight_y

                # log
                if self.global_step % self.scalar_log_gaps == 0:
                    # self.writer.add_scalar("loss/loss_generative", loss_generative, self.global_step)
                    wandb.log({"loss_generative": loss_generative}, step=self.global_step)
                    if self.y_condition:
                        # self.writer.add_scalar("loss/loss_classes", loss_classes, self.global_step)
                        wandb.log({"loss_classes": loss_classes}, step=self.global_step)
                        wandb.log({"total_loss": loss}, step=self.global_step)

                # backward
                self.graph.zero_grad()
                self.optim.zero_grad()
                loss.backward()

                # operate grad
                if self.max_grad_clip is not None and self.max_grad_clip > 0:
                    torch.nn.utils.clip_grad_value_(self.graph.parameters(), self.max_grad_clip)
                if self.max_grad_norm is not None and self.max_grad_norm > 0:
                    grad_norm = torch.nn.utils.clip_grad_norm_(self.graph.parameters(), self.max_grad_norm)
                    if self.global_step % self.scalar_log_gaps == 0:
                        # self.writer.add_scalar("grad_norm/grad_norm", grad_norm, self.global_step)
                        wandb.log({"grad_norm": grad_norm}, step=self.global_step)

                # step
                self.optim.step()

                # checkpoints
                if self.global_step % self.checkpoints_gap == 0 and self.global_step > 0:
                    save(global_step=self.global_step,
                         graph=self.graph,
                         optim=self.optim,
                         pkg_dir=self.checkpoints_dir,
                         is_best=True,
                         max_checkpoints=self.max_checkpoints)

                # plot images
                if self.global_step % self.plot_gaps == 0:
                    img = self.graph(z=z, y_onehot=y_onehot, reverse=True)
                    # img = torch.clamp(img, min=0, max=1.0)

                    if self.y_condition:
                        if self.y_criterion == "multi-classes":
                            y_pred = torch.sigmoid(y_logits)
                        elif self.y_criterion == "single-class":
                            y_pred = thops.onehot(torch.argmax(F.softmax(y_logits, dim=1), dim=1, keepdim=True),
                                                  self.y_classes)
                        y_true = y_onehot

                    # plot images
                    # self.writer.add_image("0_reverse/{}".format(bi), torch.cat((img[bi], batch["x"][bi]), dim=1), self.global_step)
                    gen_images = make_grid(torch.stack([torch.cat((img[bi], batch["x"][bi]), dim=1) for bi in range(min([len(img), self.n_image_samples]))]), nrow=10)
                    wandb.log({"0_decode": [wandb.Image(gen_images, caption="0_decode")]}, step=self.global_step)

                    # plot preds
                    for bi in range(min([len(img), self.n_image_samples])):
                        # wandb.log({"0_reverse_{}".format(bi): [wandb.Image(torch.cat((img[bi], batch["x"][bi]), dim=1), caption="0_reverse/{}".format(bi))]}, step=self.global_step)
                        if self.y_condition:
                            # self.writer.add_image("1_prob/{}".format(bi), plot_prob([y_pred[bi], y_true[bi]], ["pred", "true"]), self.global_step)
                            wandb.log({"1_prob_{}".format(bi): [wandb.Image(plot_prob([y_pred[bi], y_true[bi]], ["pred", "true"]))]}, step=self.global_step)

                # inference
                if hasattr(self, "inference_gap"):
                    if self.global_step % self.inference_gap == 0:
                        try:
                            img = self.graph(z=None, y_onehot=inference_y_onehot, eps_std=0.5, reverse=True)
                        except NameError:
                            inference_y_onehot = torch.zeros_like(y_onehot, device=torch.device('cpu'))
                            for i in range(inference_y_onehot.size(0)):
                                inference_y_onehot[i, (i % inference_y_onehot.size(1))] = 1.
                            # now
                            inference_y_onehot = inference_y_onehot.to(y_onehot.device)
                            img = self.graph(z=None, y_onehot=inference_y_onehot, eps_std=0.5, reverse=True)
                        # grid
                        sampled_images = make_grid(img[:min([len(img), self.n_image_samples])], nrow=10)
                        wandb.log({"2_sample": [wandb.Image(sampled_images, caption="2_sample")]}, step=self.global_step)
                        # img = torch.clamp(img, min=0, max=1.0)
                        # for bi in range(min([len(img), n_images])):
                        #     # self.writer.add_image("2_sample/{}".format(bi), img[bi], self.global_step)
                        #     wandb.log({"2_sample_{}".format(bi): [wandb.Image(img[bi])]}, step=self.global_step)

                if self.global_step == 0:
                    subprocess.run('nvidia-smi')

                # global step
                self.global_step += 1

        # self.writer.export_scalars_to_json(os.path.join(self.log_dir, "all_scalars.json"))
        # self.writer.close()
