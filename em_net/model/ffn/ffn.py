# deployed model without much flexibility
# useful for stand-alone test, model translation, quantization
import torch.nn as nn
import torch.nn.functional as F
import torch

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

