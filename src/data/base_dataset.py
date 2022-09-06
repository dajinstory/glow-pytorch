import os, sys
import numpy as np
import pandas as pd
import math, random
from PIL import Image, ImageDraw

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from torchvision.transforms import ToPILImage, PILToTensor

tt = T.ToTensor()
tti = ToPILImage()
itt = PILToTensor()

class BaseDataset(Dataset):
    def __init__(self, opt):
        self.root = opt['root_path']
        self.frames = self._prepare_frames(opt)
        self.img_size = opt['in_size']
        self.norm = False
        self.noise = False
        self.to_tensor = T.ToTensor()

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        return self.load_image(idx)

    def load_image(self, idx):
        # Load images
        im = self.frames[idx]['name']
        im = Image.open(os.path.join(self.root, im))

        # Resize image if necessary
        im = im.resize((self.img_size, self.img_size))
        im = self.to_tensor(im)

        # Grayscale Exception
        if im.shape[0] != 3:
            im = torch.stack([im[0,:,:]]*3, dim=0)

        # Norm, Noise
        if self.norm:
            norm = T.Normalize(mean, std)
            im = norm(im)
        if self.noise:
            im += 0.005 * torch.rand_like(im)

        return im
                 
    def _prepare_frames(self, opt):
        meta = pd.read_csv(opt['meta_path'], index_col=0).to_dict(orient='records')
        if opt['abbreviation']:
            meta = self._create_small_meta(meta, opt['batch_size_per_gpu'])
        return meta
    
    def _create_small_meta(self, meta, n_batch):
        n_valid = n_batch * torch.cuda.device_count()
        return meta[:n_valid]


