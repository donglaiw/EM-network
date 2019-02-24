import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np

from em_net.model.block.basic import conv3dBlock, upsampleBlock

class DeepAutoencoder(nn.Module):
    """
    This net is inspired from https://arxiv.org/pdf/1612.00101.pdf and https://github.com/laurahanu/2D-and-3D-Deep-Autoencoder.
    It combines U-Net like skip connection into the autoencoder.
    """

    def __init__(self, in_planes=1, out_planes=1, filters=(32, 64, 128, 256), latent_feature_size=1024, upsample_mode='bilinear', pad_mode='zero', bn_mode='sync', relu_mode='elu', init_mode='kaiming_normal', bn_momentum=0.001):
        super(DeepAutoencoder, self).__init__()
        downLayers = [in_planes] + [elem for elem in filters]
        upLayers = [elem for elem in filters[::-1]] + [out_planes]
        self.depth = len(downLayers) - 1 # This refers to the depth of the encoder/decoder. 
        
        # The encoder part.
        self.downC = nn.ModuleList([conv3dBlock(in_planes=[downLayers[x]], out_planes=[downLayers[x+1]], kernel_size=[(1,3,3)], stride=[1], padding=[0], bias=[True], pad_mode=[pad_mode], bn_mode=[bn_mode], relu_mode=[relu_mode], init_mode=init_mode, bn_momentum=bn_momentum) for x in range(self.depth)])

        self.downS = nn.ModuleList([nn.MaxPool3d(kernel_size=(1,2,2), stride=(1,2,2))] * self.depth)

        # The latent feature(center) part.
        self.latent_feature_size = latent_feature_size

        # The decoder part.
        self.upS = nn.ModuleList([nn.Upsample(scale_factor=(1,2,2), mode='trilinear', align_corners=True)] * self.depth)

        self.upC = nn.ModuleList([nn.ConvTranspose3d(in_planes=upLayers[x], out_planes=upLayers[x+1], kernel_size=(1,3,3), stride=(1,1,1), bias=True) for x in range(self.depth)])        

    def forward(self, x):

        # The encoder part.
        down_u = [None] * (self.depth)
        for i in range(self.depth):
            down_u[i] = self.downC[i](x)
            x = self.downS[i](down_u[i])

        # The center layer
        # x has shape (N, C, D, H, W)
        x_size = x.size()
        # Want C * D * H * W
        num_neurons = np.prod(list(x_size())[-4:])
        x = x.view(-1, num_neurons)
        self.center_in = nn.Linear(num_neurons, self.latent_feature_size)
        self.center_out = nn.Linear(self.latent_feature_size, num_neurons)

        x = self.center_in(x)
        x = self.center_out(x)

        x = x.view(x_size)

        # The decoder part.
        for i in range(self.depth):
            x = down_u[self.depth-1-i] + self.upS[i](x)
            x = self.upC[i](x)

        return x

         
