# Load Library
from model.glow import Glow64x64V0, Glow256x256V0

def load_glow_64x64_imagenet():
    ckpt_path = '/home/dajinhan/nas_dajinhan/experiments/glow/result/glow_64x64_imagenet.ckpt'
    net = Glow64x64V0(ckpt_path)
    return net

def load_glow_64x64_celeba():
    ckpt_path = '/home/dajinhan/nas_dajinhan/experiments/glow/result/glow_64x64_celeba.ckpt'
    net = Glow64x64V0(ckpt_path)
    return net

def load_glow_256x256_celeba():
    ckpt_path = '/home/dajinhan/nas_dajinhan/experiments/glow/result/glow_256x256_celeba.ckpt'
    net = Glow256x256V0(ckpt_path)
    return net

net = load_glow_64x64_imagenet()

def sample(net, n_samples=1, temp=0.7):
    w = torch.randn((n_samples,96,4,4)) * self.final_temp
    s
def x_to_z(self, x):
    conditions = [None] * len(self.blocks)
    z, _, _, splits = self.forward(x, conditions)
    return z, splits

def z_to_x(self, z):
    
        
    
    