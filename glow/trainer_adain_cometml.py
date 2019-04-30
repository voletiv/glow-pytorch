from comet_ml import Experiment

import datetime
import math
import numpy as np
import os
import re
import subprocess
import torch
import torch.nn.functional as F
import torchvision.utils as vutils

from tqdm import tqdm
# from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from .utils import save, load, plot_prob
from .config import JsonConfig
from .models_adain import Glow
from . import thops


class Trainer(object):
    def __init__(self, graph, optim, lrschedule, loaded_step,
                 devices, data_device,
                 content_dataset, style_dataset, hparams, name, dataset_name):
        self.hparams = hparams
        self.name = name
        self.dataset_name = dataset_name
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
        self.output_shapes = self.graph.flow.output_shapes
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
        self.content_data_loader = DataLoader(content_dataset,
                                      batch_size=self.batch_size,
                                      num_workers=8,
                                      shuffle=True,
                                      drop_last=True)
        self.style_data_loader = DataLoader(style_dataset,
                                      batch_size=self.batch_size,
                                      num_workers=8,
                                      shuffle=True,
                                      drop_last=True)
        self.style_iter = iter(self.style_data_loader)
        self.n_epoches = (hparams.Train.num_batches+len(self.content_data_loader)-1) // len(self.content_data_loader)
        self.global_step = 0
        # lr schedule
        self.lrschedule = lrschedule
        self.loaded_step = loaded_step
        # data relative
        self.y_classes = hparams.Glow.y_classes
        self.y_condition = hparams.Glow.y_condition
        self.y_criterion = hparams.Criterion.y_criterion
        # Checkpoints
        assert self.y_criterion in ["multi-classes", "single-class"]
        # style freq
        self.style_freq = hparams.Train.style_freq

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
        hparams_dict['dataset_name'] = self.dataset_name
        hparams_dict['date'] = self.date
        hparams_dict['run_name'] = self.date + "_" + self.name
        hparams_dict['log_dir'] = self.log_dir
        hparams_dict['n_epoches'] = self.n_epoches
        return hparams_dict

    def calc_mean_std(self, feat, eps=1e-5):
        # eps is a small value added to the variance to avoid divide-by-zero.
        size = feat.size()
        assert (len(size) == 4)
        N, C = size[:2]
        feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
        feat_var = feat.view(N, C, -1).var(dim=2) + eps
        feat_std = feat_var.sqrt().view(N, C, 1, 1)
        return feat_mean, feat_std

    def adaIN(self, content, style):
        content_mean, content_std = self.calc_mean_std(content)
        style_mean, style_std = self.calc_mean_std(style)
        z_new = (content - content_mean)/content_std*style_std + style_mean
        return z_new

    def style_loss(self, intermediate_zs_style, intermediate_zs_new):
        loss = 0
        for iz_s, iz_new in zip(intermediate_zs_style, intermediate_zs_new):
            # assert (iz_s.requires_grad is False)
            iz_s_mean, iz_s_std = self.calc_mean_std(iz_s)
            iz_new_mean, iz_new_std = self.calc_mean_std(iz_new)
            loss += F.mse_loss(iz_s_mean, iz_new_mean) + F.mse_loss(iz_s_std, iz_new_std)
        return loss

    def get_style_samples(self):
        try:
            yield_values = next(self.style_iter)
        except:
            self.style_iter = iter(self.style_data_loader)
            yield_values = next(self.style_iter)

        x_s = yield_values["x"]
        x_s = x_s.to(self.data_device)
        return x_s

    def train(self, cometml_project_name="glow-adain"):

        # comet_ml
        # Create an experiment
        experiment = Experiment(api_key="B6hzNydshIpZSG2Xi9BDG9gdG",
                                project_name=cometml_project_name, workspace="voletiv")
        hparams_dict = self.hparams_dict()
        experiment.log_parameters(hparams_dict)

        # set to training state
        self.graph.train()
        self.global_step = self.loaded_step

        # begin to train
        for epoch in range(self.n_epoches):
            print("epoch", epoch)
            progress = tqdm(self.content_data_loader)
            for i_batch, batch in enumerate(progress):

                experiment.set_step(self.global_step)

                # update learning rate
                lr = self.lrschedule["func"](global_step=self.global_step,
                                             **self.lrschedule["args"])
                for param_group in self.optim.param_groups:
                    param_group['lr'] = lr
                self.optim.zero_grad()

                # log
                if self.global_step % self.scalar_log_gaps == 0:
                    # self.writer.add_scalar("lr/lr", lr, self.global_step)
                    experiment.log_metrics({"lr": lr, "epoch": epoch+i_batch/len(self.content_data_loader)})

                # get batch data
                for k in batch:
                    batch[k] = batch[k].to(self.data_device)

                x_c = batch["x"]
                y_c = None
                y_onehot_c = None
                if self.y_condition:
                    if self.y_criterion == "multi-classes":
                        assert "y_onehot_c" in batch, "multi-classes ask for `y_onehot_c` (torch.FloatTensor onehot)"
                        y_onehot_c = batch["y_onehot_c"]
                    elif self.y_criterion == "single-class":
                        assert "y_c" in batch, "single-class ask for `y_c` (torch.LongTensor indexes)"
                        y_c = batch["y_c"]
                        y_onehot_c = thops.onehot(y_c, num_classes=self.y_classes)

                # at first time, initialize ActNorm
                if self.global_step == 0:
                    self.graph(x_c[:self.batch_size // len(self.devices), ...],
                               y_onehot_c[:self.batch_size // len(self.devices), ...] if y_onehot_c is not None else None)

                # parallel
                if len(self.devices) > 1 and not hasattr(self.graph, "module"):
                    print("[Parallel] move to {}".format(self.devices))
                    self.graph = torch.nn.parallel.DataParallel(self.graph, self.devices, self.devices[0])

                # forward phase
                z_c, nll_c, y_logits_c, _ = self.graph(x=x_c, y_onehot=y_onehot_c)

                # loss_generative
                loss_generative = Glow.loss_generative(nll_c)

                # loss_classes
                loss_classes = 0
                if self.y_condition:
                    loss_classes = (Glow.loss_multi_classes(y_logits_c, y_onehot_c)
                                    if self.y_criterion == "multi-classes" else
                                    Glow.loss_class(y_logits_c, y_c))

                # total loss
                loss = loss_generative + loss_classes * self.weight_y

                # AdaIN
                if (self.global_step + 1) % self.style_freq == 0:
                    # Content-1, Style-2
                    # content, style = z1.detach().clone(), z2.detach().clone()

                    # Get style samples
                    x_s = self.get_style_samples()

                    # Forward style
                    z_s, _, _, style_feats = self.graph(x=x_s, y_onehot=None)

                    # Forward content + style AdaIN
                    z_new, _, _, _ = self.graph(x=x_c, y_onehot=y_onehot_c, style_feats=style_feats)

                    # Reverse with new z
                    x_new, new_feats = self.graph(z=z_new, y_onehot=y_onehot_c, reverse=True)

                    # Style loss
                    loss_style = self.style_loss(style_feats, new_feats)
                    loss += loss_style
                    # print(loss_style.item())

                    experiment.log_metrics({"loss_style": loss_style})

                # log
                if self.global_step % self.scalar_log_gaps == 0:
                    # self.writer.add_scalar("loss/loss_generative", loss_generative, self.global_step)
                    experiment.log_metrics({"loss_generative": loss_generative,
                                            "total_loss": loss})
                    if self.y_condition:
                        # self.writer.add_scalar("loss/loss_classes", loss_classes, self.global_step)
                        experiment.log_metrics({"loss_classes": loss_classes})

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
                        experiment.log_metrics({"grad_norm": grad_norm})

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
                    rev_c, _ = self.graph(z=z_c, y_onehot=y_onehot_c, reverse=True)
                    rev_c = torch.clamp(rev_c, min=0, max=1.0)

                    if self.y_condition:
                        if self.y_criterion == "multi-classes":
                            y_pred_c = torch.sigmoid(y_logits_c)
                        elif self.y_criterion == "single-class":
                            y_pred_c = thops.onehot(torch.argmax(F.softmax(y_logits_c, dim=1), dim=1, keepdim=True),
                                                  self.y_classes)
                        y_true_c = y_onehot_c

                    # plot images
                    # self.writer.add_image("0_reverse/{}".format(bi), torch.cat((img[bi], batch["x"][bi]), dim=1), self.global_step)
                    vutils.save_image(torch.stack([torch.cat((rev_c[i], x_c[i]), dim=1) for i in range(min([len(rev_c), self.n_image_samples]))]), '/tmp/vikramvoleti_rev1.png', nrow=10)
                    experiment.log_image('/tmp/vikramvoleti_rev1.png', name="0_reverse1")

                    try:
                        vutils.save_image(torch.stack([torch.cat((x_new[i], x_c[i], x_s[i]), dim=1) for i in range(min([len(x_new), self.n_image_samples]))]), '/tmp/vikramvoleti_new1.png', nrow=10)
                        experiment.log_image('/tmp/vikramvoleti_new1.png', name="1_new1")
                    except:
                        pass

                # # inference
                # if hasattr(self, "inference_gap"):
                #     if self.global_step % self.inference_gap == 0:
                #         if self.global_step == 0:
                #             if y_onehot is not None:
                #                 inference_y_onehot = torch.zeros_like(y_onehot, device=torch.device('cpu'))
                #                 for i in range(inference_y_onehot.size(0)):
                #                     inference_y_onehot[i, (i % inference_y_onehot.size(1))] = 1.
                #                 # now
                #                 inference_y_onehot = inference_y_onehot.to(y_onehot.device)
                #             else:
                #                 inference_y_onehot = None
                #         # infer
                #         img = self.graph(z=None, y_onehot=inference_y_onehot, eps_std=0.5, reverse=True)
                #         # grid
                #         vutils.save_image(img[:min([len(img), self.n_image_samples])], '/tmp/vikramvoleti_sam.png', nrow=10)
                #         experiment.log_image('/tmp/vikramvoleti_sam.png', name="2_samples")
                #         # img = torch.clamp(img, min=0, max=1.0)
                #         # for bi in range(min([len(img), n_images])):
                #         #     # self.writer.add_image("2_sample/{}".format(bi), img[bi], self.global_step)
                #         #     wandb.log({"2_sample_{}".format(bi): [wandb.Image(img[bi])]}, step=self.global_step)

                if self.global_step == 0 or self.global_step == 1:
                    subprocess.run('nvidia-smi')

                # global step
                self.global_step += 1

        # self.writer.export_scalars_to_json(os.path.join(self.log_dir, "all_scalars.json"))
        # self.writer.close()
