import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
import torchvision
import torchvision.transforms as T
import torchvision.transforms.functional as F
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torchvision.utils import make_grid

from pytorch_lightning import LightningModule
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger

from ..common.lit_basemodel import LitBaseModel
from .glow_64x64_v0 import Glow64x64V0
from .glow_256x256_v0 import Glow256x256V0
from loss import NLLLoss #, TripletLoss, MSELoss, L1Loss, PerceptualLoss, IDLoss, GANLoss
from metric import L1, PSNR, SSIM

import os
import numbers
import numpy as np
from PIL import Image
from collections import OrderedDict

import cv2

# NLL + quant_randomness
class LitGlowV0(LitBaseModel):
    def __init__(self,
                 opt: dict,
                 pretrained=None):

        super().__init__()

        # opt
        if pretrained is True:
            opt['flow_net']['args']['pretrained'] = True
        self.opt = opt
        
        # network
        flow_nets = {
            'Glow64x64V0': Glow64x64V0,
            'Glow256x256V0': Glow256x256V0,
        }

        self.flow_net = flow_nets[self.opt['flow_net']['type']](**self.opt['flow_net']['args'])
        self.in_size = self.opt['in_size']
        self.n_bits = self.opt['n_bits']
        self.n_bins = 2.0**self.n_bits

        self.norm_mean = [0.5, 0.5, 0.5]
        self.norm_std = [1.0, 1.0, 1.0] #[0.5, 0.5, 0.5]
        # self.norm_mean = [0.485, 0.456, 0.406]
        # self.norm_std = [0.229, 0.224, 0.225]

        self.preprocess = transforms.Normalize(
            mean=self.norm_mean, 
            std=self.norm_std)
        self.reverse_preprocess = transforms.Normalize(
            mean=[-m/s for m,s in zip(self.norm_mean, self.norm_std)],
            std=[1/s for s in self.norm_std])

        # loss
        self._create_loss()
        
        # metric
        self.sampled_images = []
        
        # log
        self.save_hyperparameters(ignore=[])

    def forward(self, x):
        pass

    def preprocess_batch(self, batch):
        # Data Preprocess
        im = batch        
        im = im * 255

        if self.n_bits < 8:
            im = torch.floor(im / 2 ** (8 - self.n_bits))
        im = im / self.n_bins
        im = self.preprocess(im)
        
        conditions = [None] * len(self.flow_net.blocks)

        return im, conditions

    def training_step(self, batch, batch_idx):
        im, conditions = self.preprocess_batch(batch)

        # Forward
        # quant_randomness = torch.zeros_like(im)
        quant_randomness = self.preprocess(torch.rand_like(im)/self.n_bins) - self.preprocess(torch.zeros_like(im)) # x = (0~1)/n_bins, \ (im-m)/s + (x-m)/s - (0-m)/s = (im+x-m)/s
        w, log_p, log_det, _ = self.flow_net.forward(im + quant_randomness, conditions)
        
        # Loss
        losses = dict()
        losses['loss_nll'], log_nll = self.loss_nll(log_p, log_det, n_pixel=3*self.in_size*self.in_size)
        loss_total_common = sum(losses.values())
        
        # Log
        log_train = {
            'train/loss_nll': log_nll,
            'train/loss_total_common': loss_total_common,
        }        
        self.log_dict(log_train, logger=True, prog_bar=True)        
        return loss_total_common

    def validation_step(self, batch, batch_idx):
        im, conditions = self.preprocess_batch(batch)

        # Forward
        w, log_p, log_det, splits = self.flow_net.forward(im, conditions)

        # Reverse - Latent to Image
        w_rand = torch.randn_like(w)
        w_rand_temp = w_rand * 0.7
        splits_random = [torch.randn_like(split) for split in splits[:-1]] + [None]
        splits_random_temp = [0.7*split for split in splits_random[:-1]] + [None]

        im_recs = self.flow_net.reverse(w, conditions, splits=splits)
        im_recr = self.flow_net.reverse(w, conditions, splits=splits_random)
        im_recr_temp = self.flow_net.reverse(w, conditions, splits=splits_random_temp)
        im_gen = self.flow_net.reverse(w_rand, conditions, splits=splits_random)
        im_gen_temp = self.flow_net.reverse(w_rand_temp, conditions, splits=splits_random_temp)
        
        # Format - range (0~1)
        im = torch.clamp(self.reverse_preprocess(im), 0, 1)
        im_recs = torch.clamp(self.reverse_preprocess(im_recs), 0, 1)
        im_recr = torch.clamp(self.reverse_preprocess(im_recr), 0, 1)
        im_recr_temp = torch.clamp(self.reverse_preprocess(im_recr_temp), 0, 1)
        im_gen = torch.clamp(self.reverse_preprocess(im_gen), 0, 1)
        im_gen_temp = torch.clamp(self.reverse_preprocess(im_gen_temp), 0, 1)
        
        # Metric
        ## Metric - Image, CHW
        if batch_idx < 10:
            self.sampled_images.append(im[0].cpu())
            self.sampled_images.append(im_recs[0].cpu())
            self.sampled_images.append(im_recr[0].cpu())
            self.sampled_images.append(im_recr_temp[0].cpu())
            self.sampled_images.append(im_gen[0].cpu())
            self.sampled_images.append(im_gen_temp[0].cpu())
            
        ## Metric - PSNR, SSIM
        im = im[0].cpu().numpy().transpose(1,2,0)
        im_recs = im_recs[0].cpu().numpy().transpose(1,2,0)
        im_recr = im_recr[0].cpu().numpy().transpose(1,2,0)
        im_recr_temp = im_recr_temp[0].cpu().numpy().transpose(1,2,0)
        im_gen = im_gen[0].cpu().numpy().transpose(1,2,0)
        metric_psnr_r = PSNR(im_recr*255, im*255) 
        metric_ssim_r = SSIM(im_recr*255, im*255)
        metric_psnr_r_temp = PSNR(im_recr_temp*255, im*255) 
        metric_ssim_r_temp = SSIM(im_recr_temp*255, im*255)

        ## Metric - NLL
        loss_nll, metric_nll = self.loss_nll(log_p, log_det, n_pixel=3*self.in_size*self.in_size)

        # Log
        log_valid = {
            'val/metric/nll': metric_nll,
            'val/metric/psnr_r': metric_psnr_r,
            'val/metric/ssim_r': metric_ssim_r,
            'val/metric/psnr_r_temp': metric_psnr_r_temp,
            'val/metric/ssim_r_temp': metric_ssim_r_temp,}
        self.log_dict(log_valid)
        return log_valid

    def test_step(self, batch, batch_idx):
        # if batch_idx == 0:
        #     torch.save(self.flow_net.state_dict(), 'glow.ckpt')
        return self.validation_step(batch, batch_idx)

    def validation_epoch_end(self, outputs):
        grid = make_grid(self.sampled_images, nrow=6)
        if isinstance(self.logger, TensorBoardLogger):
            self.logger.experiment.add_image(
                f'val/visualization',
                grid, self.global_step+1, dataformats='CHW')
        self.sampled_images = []

    def test_epoch_end(self, outputs):
        grid = make_grid(self.sampled_images, nrow=6)
        if isinstance(self.logger, TensorBoardLogger):
            self.logger.experiment.add_image(
                f'test/visualization',
                grid, self.global_step+1, dataformats='CHW')
        self.sampled_images = []
        
    def configure_optimizers(self):
        trainable_parameters = [*self.flow_net.parameters()]

        optimizer = Adam(
            trainable_parameters, 
            lr=self.opt['optim']['lr'], 
            betas=self.opt['optim']['betas'])
    
        scheduler = {
            'scheduler': CosineAnnealingLR(
                optimizer, 
                T_max=self.opt['scheduler']['T_max'], 
                eta_min=self.opt['scheduler']['eta_min']),
            'name': 'learning_rate'}

        return [optimizer], [scheduler]
    
    def _create_loss(self):
        losses = {
            'NLLLoss': NLLLoss,
            # 'TripletLoss': TripletLoss,
            # 'MSELoss': MSELoss,
            # 'L1Loss': L1Loss,
            # 'PerceptualLoss': PerceptualLoss,
            # 'IDLoss': IDLoss,
            # 'GANLoss': GANLoss
        }
        
        self.loss_nll = losses[self.opt['loss']['nll']['type']](**self.opt['loss']['nll']['args'])
       
