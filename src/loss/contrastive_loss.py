import torch
import torch.nn as nn
import torch.nn.functional as F

class TripletLoss(nn.Module):
    def __init__(self, weight=1.0, n_bits=8, margin_pos=0.1, margin_neg=1.0):
        super().__init__()
        
        self.weight = weight
        self.margin_pos = margin_pos
        self.margin_neg = margin_neg

    def forward(self, anchor, pos, neg, metric_type='L2', margin_pos=None, margin_neg=None):
        metrics = {
            'L1': torch.nn.functional.l1_loss,
            'L2': torch.nn.functional.mse_loss
        }
        margin_pos = margin_pos if margin_pos else self.margin_pos
        margin_neg = margin_neg if margin_neg else self.margin_neg
        margin_pos = torch.Tensor([margin_pos]).to(anchor.get_device())
        margin_neg = torch.Tensor([margin_neg]).to(anchor.get_device())
        
        d_pos = metrics[metric_type](anchor, pos, reduction='none').mean(1)
        d_neg = metrics[metric_type](anchor, neg, reduction='none').mean(1)
        loss_cvg = torch.max(d_pos, margin_pos) - torch.min(d_neg, margin_neg)
        d_pos = d_pos.mean()
        d_neg = d_neg.mean()
        loss_cvg = loss_cvg.mean()
        
        return self.weight * loss_cvg, (loss_cvg, d_pos, d_neg)
