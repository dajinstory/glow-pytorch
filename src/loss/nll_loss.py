import torch
import torch.nn as nn
import torch.nn.functional as F

from math import log, sqrt, pi, exp, cos, sin

class NLLLoss(nn.Module):
    def __init__(self, weight=1.0, n_bits=8):
        super().__init__()

        self.weight = weight        
        self.n_bits = n_bits
        self.n_bins = 2.0 ** n_bits # 256.0
        

    def forward(self, log_p, log_det, n_pixel, **kwargs):
        """Forward Function.

        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        ll = log_p + log_det -log(self.n_bins) * n_pixel
        loss_nll = (-ll / (log(2) * n_pixel)).mean()
        
        return self.weight * loss_nll, loss_nll

