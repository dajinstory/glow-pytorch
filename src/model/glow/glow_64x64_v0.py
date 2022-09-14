import torch
import torch.nn as nn

from math import log, sqrt, pi, exp, cos, sin

from ..common.flow_module import gaussian_log_p
from ..common.flow_module import Block, FakeBlock, ZeroConv2d

def sub_conv(ch_hidden, kernel):
    pad = kernel // 2
    kernel_center = 1
    pad_center = kernel_center // 2
    return lambda ch_in, ch_out: nn.Sequential(
                                    nn.Conv2d(ch_in, ch_hidden, kernel, padding=pad),
                                    nn.ReLU(),
                                    nn.Conv2d(ch_hidden, ch_hidden, kernel_center, padding=pad_center),
                                    nn.ReLU(),
                                    nn.Conv2d(ch_hidden, ch_out, kernel, padding=pad),)

class Glow64x64V0(nn.Module):
    def __init__(self, pretrained=None):
        super().__init__()

        # configs
        self.img_size = 64
        self.w_size = 4
        self.inter_temp = 1.0
        self.final_temp = 1.0

        # Blocks (3,64,64) -> (96,4,4)
        self.blocks = nn.Sequential(
            Block(squeeze=True, # (12,32,32)
                  flow_type='InvConvFlow', n_flows=48, coupling_type= 'SingleAffine', ch_in=12, ch_c=0, n_chunk=2, subnet=sub_conv(512,3), clamp=1.0, clamp_activation='GLOW',
                  split=True),
            Block(squeeze=True, # (24,16,16)
                  flow_type='InvConvFlow', n_flows=48, coupling_type= 'SingleAffine', ch_in=24, ch_c=0, n_chunk=2, subnet=sub_conv(512,3), clamp=1.0, clamp_activation='GLOW',
                  split=True),
            Block(squeeze=True, # (48,8,8)
                  flow_type='InvConvFlow', n_flows=48, coupling_type= 'SingleAffine', ch_in=48, ch_c=0, n_chunk=2, subnet=sub_conv(512,3), clamp=1.0, clamp_activation='GLOW',
                  split=True),
            Block(squeeze=True, # (96,4,4)
                  flow_type='InvConvFlow', n_flows=48, coupling_type= 'SingleAffine', ch_in=96, ch_c=0, n_chunk=2, subnet=sub_conv(512,3), clamp=1.0, clamp_activation='GLOW',
                  split=False),
        )

        # checkpoint
        if pretrained is not None:
            ckpt_path = pretrained['ckpt_path']
            print("Load flownet - Checkpoint : ", ckpt_path, flush=True)
            self.init_weights(ckpt_path)
        else:
            print("Load flownet -  Initial Random N(0,0.01)", flush=True)
            for p in self.parameters():
                p.data = 0.01 * torch.randn_like(p)
    
    def init_weights(self, ckpt_path):
        self.load_state_dict(torch.load(ckpt_path), strict=True)
        for block in [*self.blocks]:
            for flow in block.flows:
                flow.actnorm.inited=True

    def forward(self, x, conditions):
        output = x        
        log_p = 0
        log_det = 0  
        splits = []
        
        # Blocks (3,64,64) -> (96,4,4)
        for block, condition in zip(self.blocks, conditions[:len(self.blocks)]):
            output, _log_det, _split = block(output, condition)
            log_det = log_det + _log_det
            splits.append(_split)

            if _split is not None:
                split = _split
                split = split.view(split.shape[0], -1)
                _m = torch.zeros_like(split)
                _log_sd = torch.ones_like(split) * log(self.inter_temp)
                _log_p = gaussian_log_p(split, _m, _log_sd)
                log_p += _log_p.sum(1)
        w = output
            
        # Calculate log_p for final Z
        z = output.view(output.shape[0], -1)
        _m = torch.zeros_like(z)
        _log_sd = torch.ones_like(z) * log(self.final_temp)
        _log_p = gaussian_log_p(z, _m, _log_sd)
        log_p += _log_p.sum(1)
          
        return w, log_p, log_det, splits

    def reverse(self, w, conditions, splits=None):
        input = w.view(w.shape[0],-1,self.w_size,self.w_size)
  
        # Blocks
        for block, condition, split in zip(self.blocks[::-1], conditions[::-1], splits[::-1]):
            input = block.reverse(input, condition, split)
            
        return input
        
    
    