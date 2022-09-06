# https://github.com/orpatashnik/StyleCLIP/blob/5f50b4e998ac41f5a2802145827d81647b9f5e20/criteria/id_loss.py

import torch
from torch import nn

from .facial_recognition.model_irse import Backbone as Backbone_ID_Loss


class IDLoss(nn.Module):
    def __init__(self, weight, checkpoint_path, use_input_norm=True):
        super(IDLoss, self).__init__()
        print('Loading ResNet ArcFace')
        self.facenet = Backbone_ID_Loss(input_size=112, num_layers=50, drop_ratio=0.6, mode='ir_se')
        self.facenet.load_state_dict(torch.load(checkpoint_path))
        self.pool = torch.nn.AdaptiveAvgPool2d((256, 256))
        self.face_pool = torch.nn.AdaptiveAvgPool2d((112, 112))
        self.facenet.eval()
        self.facenet.cuda()
        self.loss_weight = weight

        self.use_input_norm = use_input_norm
        if self.use_input_norm:
            # the mean is for image with range [0, 1]
            self.register_buffer(
                'mean',
                torch.Tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1))
            # the std is for image with range [-1, 1]
            self.register_buffer(
                'std',
                torch.Tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1))

        for v in self.parameters():
            v.requires_grad = False

    def extract_feats(self, x):
        # For Iterative Image Optimization
        self.facenet.eval()
        
        if x.shape[2] != 256:
            x = self.pool(x)
        x = x[:, :, 35:223, 32:220]  # Crop interesting region
        x = self.face_pool(x)
        x_feats = self.facenet(x)
        return x_feats

    def forward(self, y_hat, y):
        if self.use_input_norm:
            y_hat = (y_hat - self.mean) / self.std
            y = (y - self.mean) / self.std

        n_samples = y.shape[0]
        y_feats = self.extract_feats(y)  # Otherwise use the feature from there
        y_hat_feats = self.extract_feats(y_hat)
        y_feats = y_feats.detach()
        loss = 0
        sim_improvement = 0
        count = 0
        for i in range(n_samples):
            diff_target = y_hat_feats[i].dot(y_feats[i])
            loss += 1 - diff_target
            count += 1

        loss_id = loss / count
        return self.loss_weight * loss_id, loss_id