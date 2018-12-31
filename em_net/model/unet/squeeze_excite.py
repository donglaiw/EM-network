import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from em_net.model.residual import resBlock_seIso, resBlock_seAnisoDilation

# u-net with dilated convolution, ELU and synchronized BN
class unet_SE_synBN(nn.Module):
    # unet architecture with residual blocks
    def __init__(self, in_num=1, out_num=1, filters=[64,96,128,256], aniso_num=2):
        super(unet_SE_synBN, self).__init__()
        self.filters = filters 
        self.layer_num = len(filters) # 4
        self.aniso_num = aniso_num # the number of anisotropic conv layers

        self.downC = nn.ModuleList(
                  [resBlock_seAnisoDilation(in_num, filters[0])]
                + [resBlock_seAnisoDilation(filters[x], filters[x + 1])
                   for x in range(self.aniso_num-1)]
                + [resBlock_seIso(filters[x], filters[x + 1])
                   for x in range(self.aniso_num-1, self.layer_num-2)]
                      ) 

        self.downS = nn.ModuleList(
                [nn.MaxPool3d(kernel_size=(1,2,2), stride=(1,2,2))
                    for x in range(self.aniso_num)]
              + [nn.MaxPool3d(kernel_size=(2,2,2), stride=(2,2,2))
                    for x in range(self.aniso_num, self.layer_num-1)]
                )

        self.center = resBlock_seIso(filters[-2], filters[-1])

        self.upS = nn.ModuleList(
            [nn.Sequential(
                nn.Upsample(scale_factor=(2,2,2), mode='trilinear', align_corners=False),
                conv3d_bn_non(filters[self.layer_num-1-x], filters[self.layer_num-2-x], 
                              kernel_size=(3,3,3), stride=1, padding=(1,1,1), bias=True))
                for x in range(self.layer_num-self.aniso_num-1)]
          + [nn.Sequential(
                nn.Upsample(scale_factor=(1,2,2), mode='trilinear', align_corners=False),
                conv3d_bn_non(filters[self.layer_num-1-x], filters[self.layer_num-2-x], 
                              kernel_size=(3,3,3), stride=1, padding=(1,1,1), bias=True))
                for x in range(1, self.aniso_num+1)]
            )

        self.upC = nn.ModuleList(
            [resBlock_seIso(filters[self.layer_num - 2 - x], filters[self.layer_num - 2 - x])
             for x in range(self.layer_num-self.aniso_num-1)]
          + [resBlock_seAnisoDilation(filters[self.layer_num - 2 - x], filters[self.layer_num - 2 - x])
             for x in range(1, self.aniso_num)]
          + [resBlock_seAnisoDilation(filters[0], filters[0])]
            )

        self.fconv = conv3d_bn_non(filters[0], out_num, kernel_size=(3,3,3), stride=1, padding=(1,1,1), bias=True)
        self.elu = nn.ELU(inplace=True)
        self.softmax = nn.Softmax(dim=1) 
        self.sigmoid = nn.Sigmoid()     

        for m in self.modules():
            import pdb; pdb.set_trace()
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif 'BatchNorm' in m.name:
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        down_u = [None]*(self.layer_num-1)
        for i in range(self.layer_num-1):
            down_u[i] = self.downC[i](x)
            x = self.downS[i](down_u[i])

        x = self.center(x)

        for i in range(self.layer_num-1):
            x = down_u[self.layer_num-2-i] + self.upS[i](x)
            x = self.elu(x)
            x = self.upC[i](x)
        # convert to probability  
        x = self.fconv(x)  
        x = self.sigmoid(x)
        return x


# model for visualization purpose
class unet_SE_synBN_visualization(unet_SE_synBN):
    def __init__(self, in_num=1, out_num=3, filters=(64, 96, 128, 256), aniso_num=2):
        super(unet_SE_synBN_visualization, self).__init__(in_num, 
                                      out_num, filters, aniso_num)  

    def forward(self, x):
        down_u = [None]*(self.layer_num-1)
        for i in range(self.layer_num-1):
            down_u[i] = self.downC[i](x)
            x = self.downS[i](down_u[i])

        x = self.center(x)

        output_up0 = [None]*(self.layer_num-1)
        output_up1 = [None]*(self.layer_num-1)

        for i in range(self.layer_num-1):
            output_up0[i] = self.upS[i](x)
            x = down_u[self.layer_num-2-i] + output_up0[i]
            x = self.elu(x)
            x = self.upC[i](x)
            output_up1[i] = x
        # convert to probability
        x = self.fconv(x)    
        x = self.sigmoid(x)
        return x, down_u, output_up0, output_up1
