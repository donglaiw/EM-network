# deployed model without much flexibility
# useful for stand-alone test, model translation, quantization
import torch.nn as nn
import torch.nn.functional as F
from em_net.model.block import merge_crop
import torch


class unet_vgg(nn.Module):  # deployed Toufiq model
    def __init__(self, in_num=1, out_num=3, filters=(24, 72, 216, 648), relu_slope=0.005, rescale_skip=0):
        super(UNet3DM1, self).__init__()
        self.filters = filters
        self.io_num = [in_num, out_num]
        self.relu_slope = relu_slope

        filters_in = [in_num] + filters[:-1]
        self.depth = len(filters) - 1
        self.seq_num = self.depth * 3 + 2

        self.downC = nn.ModuleList([nn.Sequential(
            nn.Conv3d(filters_in[x], filters_in[x + 1], kernel_size=3, stride=1, bias=True),
            nn.LeakyReLU(relu_slope),
            nn.Conv3d(filters_in[x + 1], filters_in[x + 1], kernel_size=3, stride=1, bias=True),
            nn.LeakyReLU(relu_slope))
            for x in range(self.depth)])
        self.downS = nn.ModuleList(
            [nn.MaxPool3d((1, 2, 2), (1, 2, 2))
             for x in range(self.depth)])
        self.center = nn.Sequential(
            nn.Conv3d(filters[-2], filters[-1], kernel_size=3, stride=1, bias=True),
            nn.LeakyReLU(relu_slope),
            nn.Conv3d(filters[-1], filters[-1], kernel_size=3, stride=1, bias=True),
            nn.LeakyReLU(relu_slope))
        self.upS = nn.ModuleList([nn.Sequential(
            nn.ConvTranspose3d(filters[3 - x], filters[3 - x], (1, 2, 2), (1, 2, 2), groups=filters[3 - x], bias=False),
            nn.Conv3d(filters[3 - x], filters[2 - x], kernel_size=1, stride=1, bias=True))
            for x in range(self.depth)])
        # initialize upsample
        for x in range(self.depth):
            self.upS[x]._modules['0'].weight.data.fill_(1.0)

        # double input channels: merge-crop
        self.upC = nn.ModuleList([nn.Sequential(
            nn.Conv3d(2 * filters[2 - x], filters[2 - x], kernel_size=3, stride=1, bias=True),
            nn.LeakyReLU(relu_slope),
            nn.Conv3d(filters[2 - x], filters[2 - x], kernel_size=3, stride=1, bias=True),
            nn.LeakyReLU(relu_slope))
            for x in range(self.depth)])

        self.final = nn.Sequential(nn.Conv3d(filters[0], out_num, kernel_size=1, stride=1, bias=True))

    def get_learnable_seq(self, seq_id):  # learnable variable
        if seq_id < self.depth:
            return self.downC[seq_id]
        elif seq_id == self.depth:
            return self.center
        elif seq_id != self.seq_num - 1:
            seq_id = seq_id - self.depth - 1
            if seq_id % 2 == 0:
                return self.upS[seq_id / 2]
            else:
                return self.upC[seq_id / 2]
        else:
            return self.final

    def set_learnable_seq(self, seq_id, seq):  # learnable variable
        if seq_id < self.depth:
            self.downC[seq_id] = seq
        elif seq_id == self.depth:
            self.center = seq
        elif seq_id != self.seq_num - 1:
            seq_id = seq_id - self.depth - 1
            if seq_id % 2 == 0:
                self.upS[seq_id / 2] = seq
            else:
                self.upC[seq_id / 2] = seq
        else:
            self.final = seq

    def forward(self, x):
        down_u = [None] * self.depth
        for i in range(self.depth):
            down_u[i] = self.downC[i](x)
            x = self.downS[i](down_u[i])
        x = self.center(x)
        for i in range(self.depth):
            x = merge_crop(down_u[self.depth - 1 - i], self.upS[i](x))
            x = self.upC[i](x)
        return torch.sigmoid(self.final(x))
