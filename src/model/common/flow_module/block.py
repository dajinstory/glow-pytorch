import torch
import torch.nn as nn

from .flow import PermuteFlow, InvConvFlow

class Block(nn.Module):
    def __init__(self, 
                 squeeze,
                 flow_type, n_flows, ch_in, ch_c, n_chunk, subnet, clamp, clamp_activation,
                 split):
        super().__init__()

        # Squeeze
        self.squeeze = squeeze
        
        # Flows
        flows = {
            'PermuteFlow': PermuteFlow,
            'InvConvFlow': InvConvFlow}
        self.flows = nn.ModuleList()
        for _ in range(n_flows):
            self.flows.append(flows[flow_type](ch_in, ch_c, subnet, n_chunk, clamp, clamp_activation))

        # Split
        self.split = split
        
    def forward(self, input, c=None):
        output = input
        log_det = 0
        
        # Downsample
        if self.squeeze:
            b_in, ch_in, h_in, w_in = output.shape
            output = output.view(b_in, ch_in, h_in//2, 2, w_in//2, 2)
            output = output.permute(0, 1, 3, 5, 2, 4)
            output = output.contiguous().view(b_in, ch_in*4, h_in//2, w_in//2)

        # Flows
        for flow in self.flows:
            output, _log_det = flow(output, c)
            log_det = log_det + _log_det
        
        # Split
        if self.split:
            output, split = output.chunk(2, 1)
        else:
            split = None
        
        return output, log_det, split

    def reverse(self, output, c=None, split=None):
        input = output
            
        # Split Reversed
        if self.split:
            split = split.view_as(input)
            input = torch.cat([input, split], 1)

        # Flows
        for flow in self.flows[::-1]:
            input = flow.reverse(input, c)

        # Upsample
        if self.squeeze:
            b_in, ch_in, h_in, w_in = input.shape
            input = input.view(b_in, ch_in//4, 2, 2, h_in, w_in)
            input = input.permute(0, 1, 4, 2, 5, 3)
            input = input.contiguous().view(b_in, ch_in//4, h_in*2, w_in*2)

        return input

    
class FakeBlock(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, input, c=None):
        log_det = 0
        split = None
        output = input
        return output, log_det, split

    def reverse(self, output, c=None, split=None):
        input = output
        return input