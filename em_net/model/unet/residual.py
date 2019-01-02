# deployed model without much flexibility
# useful for stand-alone test, model translation, quantization
import torch.nn as nn
import torch.nn.functional as F
import torch

from em_net.model.block.basic import conv3dBlock, upsampleBlock
from em_net.model.block.residual import resBlock_pni


class UNet_PNI(nn.Module):  # deployed PNI model
    # Superhuman Accuracy on the SNEMI3D Connectomics Challenge. Lee et al.
    # https://arxiv.org/abs/1706.00120
    def __init__(self, in_planes=1, out_planes=3, filters=(28, 36, 48, 64, 80), upsample_mode='bilinear', merge_mode='add', pad_mode='zero', bn_mode='sync', relu_mode='elu', init_mode='kaiming_normal', bn_momentum=0.001, do_embed=True):
        # filter_ratio: #filter_decode/#filter_encode
        super(UNet_PNI, self).__init__()
        filters2 = filters[:1] + filters
        self.merge_mode = merge_mode
        self.do_embed = do_embed
        self.depth = len(filters2) - 2
       
        if do_embed:
            # 2D conv for anisotropic
            self.embed_in = conv3dBlock([in_planes], [filters2[0]], [(1, 5, 5)], [1], [(0, 2, 2)], [True], [pad_mode], [''], [relu_mode], init_mode, bn_momentum)

        # downsample stream
        self.downC = nn.ModuleList(
            [resBlock_pni(filters2[x], filters2[x + 1], pad_mode, bn_mode, relu_mode, init_mode, bn_momentum)
               for x in range(self.depth+1)])
        self.downS = nn.ModuleList([nn.MaxPool3d((1, 2, 2), (1, 2, 2))] * (self.depth))

        # upsample stream
        self.upS = [None] * self.depth
        self.upC = [None] * self.depth
        self.upB = [None] * self.depth
        for x in range(self.depth):
            self.upS[x] = upsampleBlock(filters2[self.depth+1-x], 
                           filters2[self.depth-x], 
                           (1,2,2), upsample_mode, init_mode=init_mode)
            if merge_mode=='add': # merge-add
                self.upB[x] = conv3dBlock([0], [filters2[self.depth-x]], bn_mode=[bn_mode], relu_mode=[relu_mode], bn_momentum=bn_momentum)
                self.upC[x] = resBlock_pni(filters2[self.depth-x], filters2[self.depth-x], pad_mode, bn_mode, relu_mode, init_mode, bn_momentum)
            elif merge_mode=='cat': # merge-concat
                self.upB[x] = conv3dBlock([0], [filters2[self.depth-x]*2], bn_mode=[bn_mode], relu_mode=[relu_mode], bn_momentum=bn_momentum)
                self.upC[x] = resBlock_pni(filters2[self.depth-x]*2, filters2[self.depth-x], pad_mode, bn_mode, relu_mode, init_mode, bn_momentum)
        self.upS = nn.ModuleList(self.upS)
        self.upC = nn.ModuleList(self.upC)
        self.upB = nn.ModuleList(self.upB)
        
        if do_embed:
            self.embed_out = conv3dBlock([filters2[0]], [filters2[0]], [(1, 5, 5)], [1], [(0, 2, 2)], [True], [pad_mode], [''], [relu_mode], init_mode, bn_momentum)
            
        self.output = conv3dBlock([filters2[0]], [out_planes], [(1, 1, 1)], init_mode=init_mode)

    def forward(self, x):
        # embedding
        if self.do_embed:
            x = self.embed_in(x)
        down_u = [None] * (self.depth)
        for i in range(self.depth):
            down_u[i] = self.downC[i](x)
            x = self.downS[i](down_u[i])
        
        x = self.downC[-1](x)

        for i in range(self.depth):
            if self.merge_mode=='add':
                x = down_u[self.depth-1-i] + self.upS[i](x)
            elif self.merge_mode=='cat':
                x = torch.cat([down_u[self.depth-1-i], self.upS[i](x)])
            x = self.upB[i](x)
            x = self.upC[i](x)
        if self.do_embed:
            x = self.embed_out(x)
        x = self.output(x)
        return F.sigmoid(x)
