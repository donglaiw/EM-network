from __future__ import print_function, division
import numpy as np
import random

import torch
import torch.utils.data

# use image augmentation
from ..augmentation import IntensityAugment, simpleaug_train_produce
from ..augmentation import apply_elastic_transform, apply_deform
#from em_dataLib import augmentor

from em_segLib.aff_util import seg_to_affgraph
from em_segLib.seg_util import mknhood3d, genSegMalis

from .dataset import BaseDataset
from .dataset import crop_volume
from em_net.util.blend import gaussian_blend


class AffinityDataset(BaseDataset):
    def __init__(self,
                 volume, label=None,
                 sample_input_size=(8, 64, 64),
                 sample_label_size=None,
                 sample_stride=(1, 1, 1),
                 augmentor=None,
                 mode='train'):

        super(AffinityDataset, self).__init__(volume,
                                              label,
                                              sample_input_size,
                                              sample_label_size,
                                              sample_stride,
                                              augmentor,
                                              mode)

    def __getitem__(self, index):
        vol_size = self.sample_input_size
        invalid_mask = None

        # Train Mode Specific Operations:
        if self.mode == 'train':
            # 2. get input volume
            seed = np.random.RandomState(index)
            # if elastic deformation: need different receptive field
            # change vol_size first
            pos = self.get_pos_seed(vol_size, seed)
            out_label = crop_volume(self.label[pos[0]], vol_size, pos[1:])
            out_input = crop_volume(self.input[pos[0]], vol_size, pos[1:])
            # 3. augmentation
            if self.augmentor is not None:  # augmentation
                #out_input, out_label = self.augmentor([out_input, out_label])
                out_input, out_label = self.simple_aug.multi_mask([out_input, out_label])
                """
                if random.random() > 0.5:
                    out_input, out_label = apply_elastic_transform(out_input, out_label)
                if random.random() > 0.75:
                    out_input = self.intensity_aug.augment(out_input)
                """

        # Test Mode Specific Operations:
        elif self.mode == 'test':
            # test mode
            pos = self.get_pos_test(index)
            out_input = crop_volume(self.input[pos[0]], vol_size, pos[1:])
            out_label = None if self.label is None else crop_volume(self.label[pos[0]], vol_size, pos[1:])
        # Turn segmentation label into affinity in Pytorch Tensor
        if out_label is not None:
            # check for invalid region (-1)
            seg_bad = np.array([-1]).astype(out_label.dtype)[0]
            invalid_mask = out_label==seg_bad
            out_label[out_label==seg_bad] = 0
            out_label = genSegMalis(out_label, 1)
            # replicate-pad the aff boundary
            out_label = seg_to_affgraph(out_label, mknhood3d(1), pad='replicate').astype(np.float32)
            out_label = torch.from_numpy(out_label.copy())

        # Turn input to Pytorch Tensor, unsqueeze once to include the channel dimension:
        out_input = torch.from_numpy(out_input.copy())
        out_input = out_input.unsqueeze(0)

        # Calculate Weight and Weight Factor
        weight_factor = None
        weight = None
        if out_label is not None:
            weight_factor = out_label.float().sum() / torch.prod(torch.tensor(out_label.size()).float())
            weight_factor = torch.clamp(weight_factor, min=1e-3)
            weight = out_label*(1-weight_factor)/weight_factor + (1-out_label)
            if invalid_mask is not None: # apply to all channel
                weight = weight * torch.Tensor(np.tile(invalid_mask[None].astype(np.uint8),(3,1,1,1)))
            weight = weight * torch.Tensor(gaussian_blend(vol_size, 0.9))
        return pos, out_input, out_label, weight, weight_factor
