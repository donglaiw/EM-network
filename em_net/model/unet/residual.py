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
    def __init__(self, in_planes=1, out_planes=3, filters=(28, 36, 48, 64, 80), upsample_mode='bilinear', merge_mode='add', pad_mode='zero', bn_mode='async', relu_mode='elu', init_mode='km_normal'):
        # filter_ratio: #filter_decode/#filter_encode
        super(UNet_PNI, self).__init__()
        self.filters = filters
        self.io_num = [in_planes, out_planes]
        self.res_num = len(filters) - 2
        self.merge_mode = merge_mode

        self.downC = nn.ModuleList(
            [conv3dBlock([in_planes], [filters[0]], [(1, 5, 5)], [1], [(0, 2, 2)], [False], [pad_mode], [bn_mode], [relu_mode], init_mode)]
            + [resBlock_pni(filters[x], filters[x + 1], True, pad_mode, bn_mode, relu_mode, init_mode)
               for x in range(self.res_num)])
        self.downS = nn.ModuleList([nn.MaxPool3d((1, 2, 2), (1, 2, 2))] * (self.res_num + 1))
        self.center = resBlock_pni(filters[-2], filters[-1], True, pad_mode, bn_mode, relu_mode, init_mode)
       
        self.upS = nn.ModuleList(
            [upsampleBlock(filters[self.res_num + 1 - x], 
                           filters[self.res_num - x], 
                           up = (1,2,2), mode = upsample_mode, init_mode=init_mode)
                for x in range(self.res_num + 1)])
        
        if merge_mode=='add': # merge-add
            up_layers = [resBlock_pni(filters[self.res_num - x], filters[self.res_num - x], True, pad_mode, bn_mode, relu_mode, init_mode)
             for x in range(self.res_num)]
        elif merge_mode=='cat': # merge-concat
            up_layers = [resBlock_pni(filters[self.res_num - x]*2, filters[self.res_num - x], True, pad_mode, bn_mode, relu_mode, init_mode)
             for x in range(self.res_num)]

        self.upC = nn.ModuleList(
            up_layers
            + [conv3dBlock([filters[0]], [out_planes], [(1, 5, 5)], [1], [(0, 2, 2)], [True], [pad_mode], [''], [''], init_mode)])


    def forward(self, x):
        down_u = [None] * (self.res_num + 1)
        for i in range(self.res_num + 1):
            down_u[i] = self.downC[i](x)
            x = self.downS[i](down_u[i])
        x = self.center(x)
        for i in range(self.res_num + 1):
            if self.merge_mode=='add':
                x = down_u[self.res_num - i] + self.upS[i](x)
            elif self.merge_mode=='cat':
                x = torch.cat([down_u[self.res_num - i], self.upS[i](x)])
            x = self.upC[i](x)
        return torch.sigmoid(x)


class FFNBasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=(3, 3, 3), pad_size=1):
        super(FFNBasicBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, padding=pad_size, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_planes, out_planes, kernel_size=kernel_size, padding=pad_size, bias=True))

    def forward(self, x):
        return x + self.conv(x)


class FFN(nn.Module):  # deployed FFN model
    # https://github.com/google/ffn/blob/master/ffn/training/models/convstack_3d.py
    # https://github.com/google/ffn/blob/master/ffn/training/model.py
    def __init__(self, in_seed=1, in_patch=1, deltas=(4, 4, 4),
                 depth=11, filter_num=32, pad_size=1, kernel_size=(3, 3, 3)):
        super(FFN, self).__init__()
        self.build_rnn(in_seed, in_patch, deltas)
        self.build_conv(depth, filter_num, pad_size, kernel_size)
        self.pred_mask_size = None
        self.input_seed_size = None
        self.input_image_size = None
        self.in_seed = None
        self.in_patch = None
        self.deltas = None
        self.shifts = None
        self.depth = depth
        # build convolution model
        self.conv0 = None
        self.conv1 = None
        self.conv2 = None
        self.out = nn.Sigmoid()

    def set_uniform_io_size(self, patch_size):
            """Initializes unset input/output sizes to 'patch_size', sets input shapes.
            This assumes that the inputs and outputs are of equal size, and that exactly
            one step is executed in every direction during training.
            Args:
              patch_size: (x, y, z) specifying the input/output patch size
            Returns:
              None
            """
            if self.pred_mask_size is None:
                self.pred_mask_size = patch_size
            if self.input_seed_size is None:
                self.input_seed_size = patch_size
            if self.input_image_size is None:
                self.input_image_size = patch_size
            self.set_input_shapes()

    def set_input_shapes(self):
        """Sets the shape inference for input_seed and input_patches.
        Assumes input_seed_size and input_image_size are already set.
        """
        self.input_seed.set_shape([self.batch_size] +
                                  list(self.input_seed_size[::-1]) + [1])
        self.input_patches.set_shape([self.batch_size] +
                                     list(self.input_image_size[::-1]) + [1])

    def build_rnn(self, in_seed, in_patch, deltas):
        # parameters for recurrent
        self.in_seed = in_seed
        self.in_patch = in_patch
        self.deltas = deltas
        self.shifts = []
        for dx in (-self.deltas[0], 0, self.deltas[0]):
            for dy in (-self.deltas[1], 0, self.deltas[1]):
                for dz in (-self.deltas[2], 0, self.deltas[2]):
                    if dx == 0 and dy == 0 and dz == 0:
                        continue
                    self.shifts.append((dx, dy, dz))

    def build_conv(self, depth, filter_num, pad_size, kernel_size):
        self.depth = depth
        # build convolution model
        self.conv0 = nn.Sequential(
            nn.Conv3d(2, filter_num, kernel_size=kernel_size, stride=1, padding=pad_size, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv3d(filter_num, filter_num, kernel_size=kernel_size, stride=1, padding=pad_size, bias=True))

        self.conv1 = nn.ModuleList(
            [FFNBasicBlock(filter_num, filter_num, kernel_size, pad_size)
             for x in range(self.depth)])

        self.conv2 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv3d(filter_num, filter_num, kernel_size=1, bias=True))
        self.out = nn.Sigmoid()

    def predict_object_mask(self, x):
        out = self.conv0(x)
        for i in range(self.depth):
            out = self.conv1[i](out)
        return self.conv2(out)

    def update_seed(self, seed, update):
        """Updates the initial 'seed' with 'update'."""
        dx = self.input_seed_size[0] - self.pred_mask_size[0]
        dy = self.input_seed_size[1] - self.pred_mask_size[1]
        dz = self.input_seed_size[2] - self.pred_mask_size[2]

        if dx == 0 and dy == 0 and dz == 0:
            seed += update
        else:
            seed += F.pad(update, ([0, 0],
                                   [dz // 2, dz - dz // 2],
                                   [dy // 2, dy - dy // 2],
                                   [dx // 2, dx - dx // 2],
                                   [0, 0]))
        return seed

    def forward(self, x):
        # input
        logit_update = self.predict_object_mask(x)

        logit_seed = self.update_seed(self.input_seed, logit_update)

        self.logits = logit_seed
        self.logistic = self.out(logit_seed)

