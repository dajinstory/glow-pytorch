import torch
import torch.nn as nn
import torchvision

def get_vgg_header(ch_in, ch_hidden, ch_out, kernel=3):
    pad = kernel // 2
    header = nn.Sequential(
        nn.Conv2d(ch_in, ch_hidden, kernel, padding=pad),
        nn.ReLU(),
        nn.Conv2d(ch_hidden, ch_hidden, kernel, padding=pad),
        nn.ReLU(),
        nn.Conv2d(ch_hidden, ch_out, kernel, padding=pad),
        nn.Tanh()
    )
    return header