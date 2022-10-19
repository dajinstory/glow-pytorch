import torch
from torch import nn
from torch.nn import functional as F
from math import log, sqrt, pi, exp, cos, sin
import numpy as np
from scipy import linalg as la

logabs = lambda x: torch.log(torch.abs(x))
    

class _ActNorm(nn.Module):
    """
    REFERENCE : https://github.com/y0ast/Glow-PyTorch/blob/master/modules.py
    
    Activation Normalization
    Initialize the bias and scale with a given minibatch,
    so that the output per-channel have zero mean and unit variance for that.
    After initialization, `bias` and `logs` will be trained as parameters.
    """

    def __init__(self, num_features, scale=1.0):
        super().__init__()
        # register mean and scale
        size = [1, num_features, 1, 1]
        self.bias = nn.Parameter(torch.zeros(*size))
        self.logs = nn.Parameter(torch.zeros(*size))
        self.num_features = num_features
        self.scale = scale
        self.inited = False

    def initialize_parameters(self, input):
        with torch.no_grad():
            bias = -torch.mean(input.clone(), dim=[0, 2, 3], keepdim=True)
            vars = torch.mean((input.clone() + bias) ** 2, dim=[0, 2, 3], keepdim=True)
            logs = torch.log(self.scale / (torch.sqrt(vars) + 1e-6))

            self.bias.data.copy_(bias.data)
            self.logs.data.copy_(logs.data)
        
        if self.training:
            self.inited = True

    def _center(self, input, reverse=False):
        if reverse:
            return input - self.bias
        else:
            return input + self.bias

    def _scale(self, input, logdet=None, reverse=False):

        if reverse:
            input = input * torch.exp(-self.logs)
        else:
            input = input * torch.exp(self.logs)

        if logdet is not None:
            """
            logs is log_std of `mean of channels`
            so we need to multiply by number of pixels
            """
            b, c, h, w = input.shape

            dlogdet = torch.sum(self.logs) * h * w

            if reverse:
                dlogdet *= -1

            logdet = logdet + dlogdet

        return input, logdet

    def forward(self, input, logdet=None, reverse=False):
        self._check_input_dim(input)

        if not self.inited:
            self.initialize_parameters(input)

        if reverse:
            input, logdet = self._scale(input, logdet, reverse)
            input = self._center(input, reverse)
        else:
            input = self._center(input, reverse)
            input, logdet = self._scale(input, logdet, reverse)

        return input, logdet


class ActNorm2d(_ActNorm):
    '''
    REFERENCE : https://github.com/y0ast/Glow-PyTorch/blob/master/modules.py
    '''
    def __init__(self, num_features, scale=1.0):
        super().__init__(num_features, scale)

    def _check_input_dim(self, input):
        assert len(input.size()) == 4
        assert input.size(1) == self.num_features, (
            "[ActNorm]: input should be in shape as `BCHW`,"
            " channels should be {} rather than {}".format(
                self.num_features, input.size()
            )
        )
        
        
class InvConv2dLU(nn.Module):
    def __init__(self, ch_in):
        super().__init__()

        weight = np.random.randn(ch_in, ch_in)
        q, _ = la.qr(weight)
        w_p, w_l, w_u = la.lu(q.astype(np.float32))
        w_s = np.diag(w_u)
        w_u = np.triu(w_u, 1)
        u_mask = np.triu(np.ones_like(w_u), 1)
        l_mask = u_mask.T

        w_p = torch.from_numpy(w_p)
        w_l = torch.from_numpy(w_l)
        w_s = torch.from_numpy(w_s)
        w_u = torch.from_numpy(w_u)

        self.register_buffer("w_p", w_p)
        self.register_buffer("u_mask", torch.from_numpy(u_mask))
        self.register_buffer("l_mask", torch.from_numpy(l_mask))
        self.register_buffer("s_sign", torch.sign(w_s))
        self.register_buffer("l_eye", torch.eye(l_mask.shape[0]))
        self.w_l = nn.Parameter(w_l)
        self.w_s = nn.Parameter(logabs(w_s))
        self.w_u = nn.Parameter(w_u)

    def forward(self, input):
        _, _, height, width = input.shape

        weight = self.calc_weight()

        output = F.conv2d(input, weight)
        log_det = height * width * torch.sum(self.w_s)

        return output, log_det

    def calc_weight(self):
        weight = (
            self.w_p
            @ (self.w_l * self.l_mask + self.l_eye)
            @ ((self.w_u * self.u_mask) + torch.diag(self.s_sign * torch.exp(self.w_s)))
        )

        return weight.unsqueeze(2).unsqueeze(3)

    def reverse(self, output):
        weight = self.calc_weight()

        return F.conv2d(output, weight.squeeze().inverse().unsqueeze(2).unsqueeze(3))


class ZeroConv2d(nn.Module):
    def __init__(self, ch_in, ch_out, padding=1):
        super().__init__()

        self.conv = nn.Conv2d(ch_in, ch_out, 3, padding=0)
        self.conv.weight.data.zero_()
        self.conv.bias.data.zero_()
        self.scale = nn.Parameter(torch.zeros(1, ch_out, 1, 1))
        self.inited = False
    
    def initialize_parameters(self):
        self.conv.weight.data.zero_()
        self.conv.bias.data.zero_()
        self.scale = nn.Parameter(torch.zeros_like(self.scale))
        
        if self.training:
            self.inited = True

    def forward(self, input):
        if not self.inited:
            self.initialize_parameters()

        output = F.pad(input, [1, 1, 1, 1], value=1)
        output = self.conv(output)
        output = output * torch.exp(self.scale * 3)

        return output
    

class RandomPermute(nn.Module):
    '''permutes input vector in a random but fixed way'''
    def __init__(self, ch_in, seed=None):
        super().__init__()

        self.ch_in = ch_in

        if seed is not None:
            np.random.seed(seed)
        self.perm = np.random.permutation(self.ch_in)

        self.perm_inv = np.zeros_like(self.perm)
        for i, p in enumerate(self.perm):
            self.perm_inv[p] = i

        self.perm = torch.LongTensor(self.perm)
        self.perm_inv = torch.LongTensor(self.perm_inv)


    def forward(self, input):
        output = input[:, self.perm, ...]
        return output
    
    def reverse(self, output):
        input = output[:, self.perm_inv, ...]
        return input


class AdditiveCoupling(nn.Module):
    def __init__(self, ch_in, ch_c, subnet, n_chunk=2, clamp=2.0, clamp_activation='ATAN'):
        super().__init__()
        self.clamp = clamp
        self.ch_in = ch_in
        self.ch_c = ch_c
        
        self.n_chunk = n_chunk
        self.ch_chunks = []
        for _ in range(n_chunk):
            self.ch_chunks.append((ch_in - sum(self.ch_chunks) - 1) // (n_chunk - len(self.ch_chunks)) + 1)
        
        # NN
        self.nets = nn.ModuleList()
        for ch_chunk in self.ch_chunks:
            self.nets.append(subnet(ch_in-ch_chunk+ch_c, ch_chunk))
            for idx, layer in enumerate(self.nets[-1]):
                if type(layer).__name__ == 'Conv2d':
                    self.nets[-1][idx].weight.data.normal_(0, 0.05)
                    self.nets[-1][idx].bias.data.zero_()
                
        # Activation
        f_clamps = {
            'ATAN': (lambda u: 0.636 * torch.atan(u)),
            'TANH': torch.tanh,
            'SIGMOID': (lambda u: 2. * (torch.sigmoid(u) - 0.5)),
            'GLOW': (lambda u: torch.sigmoid(u + 2))}
        self.f_clamp = f_clamps[clamp_activation]

    def forward(self, input, c=None):
        chunks = list(torch.split(input, self.ch_chunks, dim=1))
        log_det = 0
        
        # Nets
        for idx in range(self.n_chunk):
            u = torch.cat(chunks[:idx]+chunks[idx+1:], dim=1)
            u_c = torch.cat([u, c], dim=1) if self.ch_c > 0 else u
            chunks[idx] = chunks[idx] + self.nets[idx](u_c)
                             
        output = torch.cat(chunks, dim=1)
    
        return output, log_det

    def reverse(self, output, c=None):
        chunks = list(torch.split(output, self.ch_chunks, dim=1))
        
        # Nets
        for idx in reversed(range(self.n_chunk)):
            u = torch.cat(chunks[:idx]+chunks[idx+1:], dim=1)
            u_c = torch.cat([u, c], dim=1) if self.ch_c > 0 else u
            chunks[idx] = chunks[idx] - self.nets[idx](u_c)
         
        input = torch.cat(chunks, dim=1)
    
        return input


class AffineCoupling(nn.Module):
    def __init__(self, ch_in, ch_c, subnet, n_chunk=2, clamp=2.0, clamp_activation='ATAN'):
        super().__init__()
        self.clamp = clamp
        self.ch_in = ch_in
        self.ch_c = ch_c
        
        self.n_chunk = n_chunk
        self.ch_chunks = []
        for _ in range(n_chunk):
            self.ch_chunks.append((ch_in - sum(self.ch_chunks) - 1) // (n_chunk - len(self.ch_chunks)) + 1)
        
        # NN
        self.nets = nn.ModuleList()
        for ch_chunk in self.ch_chunks:
            self.nets.append(subnet(ch_in-ch_chunk+ch_c, ch_chunk*2))
            for idx, layer in enumerate(self.nets[-1]):
                if type(layer).__name__ == 'Conv2d':
                    self.nets[-1][idx].weight.data.normal_(0, 0.05)
                    self.nets[-1][idx].bias.data.zero_()
                
        # Activation
        f_clamps = {
            'ATAN': (lambda u: 0.636 * torch.atan(u)),
            'TANH': torch.tanh,
            'SIGMOID': (lambda u: 2. * (torch.sigmoid(u) - 0.5)),
            'GLOW': (lambda u: torch.sigmoid(u + 2))}
        self.f_clamp = f_clamps[clamp_activation]

    def forward(self, input, c=None):
        chunks = list(torch.split(input, self.ch_chunks, dim=1))
        log_det = 0
        
        # Nets
        for idx in range(self.n_chunk):
            u = torch.cat(chunks[:idx]+chunks[idx+1:], dim=1)
            u_c = torch.cat([u, c], dim=1) if self.ch_c > 0 else u
            log_s, t = self.nets[idx](u_c).chunk(2, 1)
            log_s = self.clamp * self.f_clamp(log_s)
            s = torch.exp(log_s)
            chunks[idx] = (chunks[idx] + t) * s
            log_det = log_det + torch.sum(log_s.view(log_s.shape[0], -1), 1)
                             
        output = torch.cat(chunks, dim=1)
    
        return output, log_det

    def reverse(self, output, c=None):
        chunks = list(torch.split(output, self.ch_chunks, dim=1))
        
        # Nets
        for idx in reversed(range(self.n_chunk)):
            u = torch.cat(chunks[:idx]+chunks[idx+1:], dim=1)
            u_c = torch.cat([u, c], dim=1) if self.ch_c > 0 else u
            log_s, t = self.nets[idx](u_c).chunk(2, 1)
            log_s = self.clamp * self.f_clamp(log_s)
            s = torch.exp(log_s)
            chunks[idx] = chunks[idx] / s - t
         
        input = torch.cat(chunks, dim=1)
    
        return input
    

class SingleAdditiveCoupling(nn.Module):
    def __init__(self, ch_in, ch_c, subnet, n_chunk=2, clamp=2.0, clamp_activation='ATAN'):
        super().__init__()
        self.clamp = clamp
        self.ch_in = ch_in
        self.ch_c = ch_c
        
        self.n_chunk = n_chunk
        self.ch_chunks = []
        for _ in range(n_chunk):
            self.ch_chunks.append((ch_in - sum(self.ch_chunks) - 1) // (n_chunk - len(self.ch_chunks)) + 1)
        
        # NN
        self.nets = nn.ModuleList()
        for ch_chunk in self.ch_chunks:
            self.nets.append(subnet(ch_in-ch_chunk+ch_c, ch_chunk))
            for idx, layer in enumerate(self.nets[-1]):
                if type(layer).__name__ == 'Conv2d':
                    self.nets[-1][idx].weight.data.normal_(0, 0.05)
                    self.nets[-1][idx].bias.data.zero_()
                
        # Activation
        f_clamps = {
            'ATAN': (lambda u: 0.636 * torch.atan(u)),
            'TANH': torch.tanh,
            'SIGMOID': (lambda u: 2. * (torch.sigmoid(u) - 0.5)),
            'GLOW': (lambda u: torch.sigmoid(u + 2))}
        self.f_clamp = f_clamps[clamp_activation]

    def forward(self, input, c=None):
        chunks = list(torch.split(input, self.ch_chunks, dim=1))
        log_det = 0
        
        # Nets
        for idx in range(1):
            u = torch.cat(chunks[:idx]+chunks[idx+1:], dim=1)
            u_c = torch.cat([u, c], dim=1) if self.ch_c > 0 else u
            chunks[idx] = chunks[idx] + self.nets[idx](u_c)
                             
        output = torch.cat(chunks, dim=1)
    
        return output, log_det

    def reverse(self, output, c=None):
        chunks = list(torch.split(output, self.ch_chunks, dim=1))
        
        # Nets
        for idx in reversed(range(1)):
            u = torch.cat(chunks[:idx]+chunks[idx+1:], dim=1)
            u_c = torch.cat([u, c], dim=1) if self.ch_c > 0 else u
            chunks[idx] = chunks[idx] - self.nets[idx](u_c)
         
        input = torch.cat(chunks, dim=1)
    
        return input


class SingleAffineCoupling(nn.Module):
    def __init__(self, ch_in, ch_c, subnet, n_chunk=2, clamp=2.0, clamp_activation='ATAN'):
        super().__init__()
        self.clamp = clamp
        self.ch_in = ch_in
        self.ch_c = ch_c
        
        self.n_chunk = n_chunk
        self.ch_chunks = []
        for _ in range(n_chunk):
            self.ch_chunks.append((ch_in - sum(self.ch_chunks) - 1) // (n_chunk - len(self.ch_chunks)) + 1)
        
        # NN
        self.nets = nn.ModuleList()
        for ch_chunk in self.ch_chunks:
            self.nets.append(subnet(ch_in-ch_chunk+ch_c, ch_chunk*2))
            for idx, layer in enumerate(self.nets[-1]):
                if type(layer).__name__ == 'Conv2d':
                    self.nets[-1][idx].weight.data.normal_(0, 0.05)
                    self.nets[-1][idx].bias.data.zero_()
                
        # Activation
        f_clamps = {
            'ATAN': (lambda u: 0.636 * torch.atan(u)),
            'TANH': torch.tanh,
            'SIGMOID': (lambda u: 2. * (torch.sigmoid(u) - 0.5)),
            'GLOW': (lambda u: torch.sigmoid(u + 2))}
        self.f_clamp = f_clamps[clamp_activation]

    def forward(self, input, c=None):
        chunks = list(torch.split(input, self.ch_chunks, dim=1))
        log_det = 0
        
        # Nets
        for idx in range(1):
            u = torch.cat(chunks[:idx]+chunks[idx+1:], dim=1)
            u_c = torch.cat([u, c], dim=1) if self.ch_c > 0 else u
            log_s, t = self.nets[idx](u_c).chunk(2, 1)
            log_s = self.clamp * self.f_clamp(log_s)
            s = torch.exp(log_s)
            chunks[idx] = (chunks[idx] + t) * s
            log_det = log_det + torch.sum(log_s.view(log_s.shape[0], -1), 1)
                             
        output = torch.cat(chunks, dim=1)
    
        return output, log_det

    def reverse(self, output, c=None):
        chunks = list(torch.split(output, self.ch_chunks, dim=1))
        
        # Nets
        for idx in reversed(range(1)):
            u = torch.cat(chunks[:idx]+chunks[idx+1:], dim=1)
            u_c = torch.cat([u, c], dim=1) if self.ch_c > 0 else u
            log_s, t = self.nets[idx](u_c).chunk(2, 1)
            log_s = self.clamp * self.f_clamp(log_s)
            s = torch.exp(log_s)
            chunks[idx] = chunks[idx] / s - t
         
        input = torch.cat(chunks, dim=1)
    
        return input


class PermuteFlow(nn.Module):
    def __init__(self, coupling_type, ch_in, ch_c, subnet, n_chunk=2, clamp=2.0, clamp_activation="ATAN"):
        super().__init__()

        couplings = {
            'SingleAffine': SingleAffineCoupling,
            'SingleAdditive': SingleAdditiveCoupling,
            'Affine': AffineCoupling,
            'Additive': AdditiveCoupling,
        }

        # Flow Components
        self.permute = RandomPermute(ch_in)
        self.coupling = couplings[coupling_type](ch_in, ch_c, subnet, n_chunk, clamp, clamp_activation)
        
    def forward(self, input, c=None):
        output = input
        output = self.permute(output)
        output, log_det = self.coupling(output, c)
        return output, log_det

    def reverse(self, output, c=None):
        input = output
        input = self.coupling.reverse(input, c)
        input = self.permute.reverse(input)
        return input
    

class InvConvFlow(nn.Module):
    def __init__(self, coupling_type, ch_in, ch_c, subnet, n_chunk=2, clamp=2.0, clamp_activation="GLOW"):
        super().__init__()

        couplings = {
            'SingleAffine': SingleAffineCoupling,
            'SingleAdditive': SingleAdditiveCoupling,
            'Affine': AffineCoupling,
            'Additive': AdditiveCoupling,
        }

        # Flow Components
        self.actnorm = ActNorm2d(ch_in)
        self.invconv = InvConv2dLU(ch_in)
        self.coupling = couplings[coupling_type](ch_in, ch_c, subnet, n_chunk, clamp, clamp_activation)
        
    def forward(self, input, c=None):
        output = input
        output, log_det_0 = self.actnorm(output, logdet=0, reverse=False)
        output, log_det_1 = self.invconv(output)
        output, log_det_2 = self.coupling(output, c)
        log_det = log_det_0 + log_det_1 + log_det_2
        return output, log_det

    def reverse(self, output, c=None):
        input = output
        input = self.coupling.reverse(input, c)
        input = self.invconv.reverse(input,)
        input, _ = self.actnorm(input, reverse=True)
        return input

