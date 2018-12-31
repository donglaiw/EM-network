from __future__ import print_function, division
import numpy as np
import random

import torch

from .dataset import BaseDataset
# dataset class for polarity input
class PolaritySynapseDataset(BaseDataset):
    def __init__(self,
                 volume, label=None,
                 sample_input_size=(8, 64, 64),
                 sample_label_size=None,
                 sample_stride=(1, 1, 1),
                 data_aug=False,
                 mode='train',
                 activation='sigmoid'):

        super(PolaritySynapseDataset, self).__init__(volume,
                                                     label,
                                                     sample_input_size,
                                                     sample_label_size,
                                                     sample_stride,
                                                     data_aug,
                                                     mode)

        self.activation = activation
        num_vol = len(label)
        self.label_pos = [None]*num_vol
        self.label_neg = [None]*num_vol
        self.label = [None]*num_vol

        for idx in range(num_vol):
            assert label[idx].ndim == 4
            self.label_pos[idx] = label[idx][0, :, :, :]
            self.label_neg[idx] = label[idx][1, :, :, :]
            self.label[idx] = self.label_pos[idx] + self.label_neg[idx]

    def __getitem__(self, index):
        if self.mode == 'train':
            # 1. get volume size
            vol_size = self.sample_input_size

            # reject no-synapse samples with a probability of p 
            seed = np.random.RandomState(index)
            while True:
                pos = self.get_pos_seed(vol_size, seed)
                out_label = crop_volume(self.label[pos[0]], vol_size, pos[1:])
                if np.sum(out_label) > 100:
                    break
                else:
                    if random.random() > 0.75:
                        break
                        # pos = self.getPos(vol_size, index)

            # 2. get input volume
            out_input = crop_volume(self.input[pos[0]], vol_size, pos[1:])
            out_label_pos = crop_volume(self.label_pos[pos[0]], vol_size, pos[1:])
            out_label_neg = crop_volume(self.label_neg[pos[0]], vol_size, pos[1:])

            # 3. augmentation
            if self.data_aug:  # augmentation
                # if random.random() > 0.5:
                #    out_input, out_label = apply_elastic_transform(out_input, out_label)    
                out_input, out_label, out_label_pos, out_label_neg = \
                    self.simple_aug.multi_mask([out_input, out_label, out_label_pos, out_label_neg])
                # if random.random() > 0.75: out_input = self.intensity_aug.augment(out_input)
                if random.random() > 0.5:
                    out_input = apply_deform(out_input)

            # 4. class weight
            # add weight to classes to handle data imbalance
            # match input tensor shape
            out_input = torch.from_numpy(out_input.copy())
            out_label = torch.from_numpy(out_label.copy())
            out_label_pos = torch.from_numpy(out_label_pos.copy())
            out_label_neg = torch.from_numpy(out_label_neg.copy())

            weight_factor = out_label.float().sum() / torch.prod(torch.tensor(out_label.size()).float())
            weight_factor = torch.clamp(weight_factor, min=1e-3)
            # the fraction of synaptic cleft pixels, can be 0
            weight = out_label*(1-weight_factor)/weight_factor + (1-out_label)
            ww = torch.Tensor(gaussian_blend(vol_size, 0.9))
            weight = weight * ww

            # include the channel dimension
            out_input = out_input.unsqueeze(0)
            weight = weight.unsqueeze(0)

            if self.activation == 'sigmoid':
                out_label_final = torch.stack([out_label_pos, out_label_neg, out_label])  # 3 channel output
            elif self.activation == 'tanh':
                out_label_final = out_label_pos - out_label_neg
                out_label_final = out_label_final.unsqueeze(0)
            elif self.activation == 'softmax':
                out_label_final = (1-out_label)*0 + out_label_pos*1 + out_label_neg*2
                out_label_final = out_label_final.long()
            else:
                raise ValueError("The following activation function is not supported: {}".format(self.activation))

            # class_weight = torch.Tensor([(1-weight_factor)/weight_factor, (1-weight_factor)/weight_factor, 1])

            return out_input, out_label_final, weight, weight_factor

        elif self.mode == 'test':
            # 1. get volume size
            vol_size = self.sample_input_size
            # test mode
            pos = self.get_pos_test(index)
            out_input = crop_volume(self.input[pos[0]], vol_size, pos[1:])
            out_input = torch.Tensor(out_input)
            out_input = out_input.unsqueeze(0)
            return pos, out_input
