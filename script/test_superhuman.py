import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import h5py
import time
import argparse

import torch
import torch.utils.data
from em_net.data.dataset import AffinityDataset, collate_fn_test
#from em_net.model.model_seg import UNet3DPniM2
from em_net.model.io import load_checkpoint
from em_net.model.unet import UNet_PNI
from em_net.libs.sync import DataParallelWithCallback
from em_net.util.options import *


def get_args():
    parser = argparse.ArgumentParser(
        description='Tests a previously trained PNI 3DUNet model for SNEMI affinity prediction.')
    # I/O
    optIO(parser, 'test')

    optModel(parser)
    # machine option
    parser.add_argument('-g', '--num-gpu', type=int, default=1,
                        help='Number of CUDA capable GPUs used for testing the model. Must be positive.')
    parser.add_argument('-j', '--num-procs', type=int, default=1,
                        help="Number of processes used with Pytorch's dataloader class.")
    parser.add_argument('-b', '--batch-size', type=int, default=1,
                        help='Batch size.')
    parser.add_argument('-ml', '--model', help='Path of the model used for testing.')

    args = parser.parse_args()
    optParse(args, mode='test')
    return args


def check_output(args):
    """
    Checks whether the output directory exists. If not, creates it.
    """
    # Create the output directory before writing into it.
    sn = args.output_dir + '/'
    if not os.path.isdir(sn):
        os.makedirs(sn)
        print('Output directory was created.')
    print('Output directory: {}.'.format(sn))


def get_preferred_device(args):
    if args.num_gpu < 0:
        raise ValueError("The number of GPUs must be greater than or equal to zero.")
    return torch.device("cuda" if torch.cuda.is_available() and args.num_gpu > 0 else "cpu")


def get_model_input_shape(args):
    #shape = np.array([int(x) for x in args.input_shape.split(',')])
    shape = args.data_shape
    print("Model Shape: {}.".format(shape))
    return shape


def load_volumes(args, model_input_size):
    in_path = args.test_volume.split('@')
    x_pad = args.x_pad
    y_pad = args.y_pad
    z_pad = args.z_pad
    # 1. load data
    print('Number of volumes passed: {}.'.format(len(in_path)))
    test_input = [None] * len(in_path)
    result = [None] * len(in_path)
    weight = [None] * len(in_path)

    # may use datasets from multiple folders
    # should be either one or the same as dir_name
    # original image is in [0, 255], normalize to [0, 1]
    print("Loading volumes...")
    for i in range(len(in_path)):
        test_input[i] = np.array(h5py.File(in_path[i], 'r')['main']).astype(np.float32) / 255.0
        test_input[i] = np.pad(test_input[i], [[z_pad, z_pad], [y_pad, y_pad], [x_pad, x_pad]], 'reflect')
        print("Loaded {}.".format(in_path[i]))
        print("Volume shape: {}".format(test_input[i].shape))
        result[i] = np.zeros(test_input[i].shape)
        weight[i] = np.zeros(test_input[i].shape)

        result[i] = np.stack([result[i]] * 3)
        print("Result shape: {}.".format(result[i].shape))
        print("Weight shape: {}.".format(weight[i].shape))

    dataset = AffinityDataset(volume=test_input, label=None, sample_input_size=model_input_size,
                              sample_label_size=None, sample_stride=model_input_size / 2,
                              augmentor=None, mode='test')

    img_loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn_test,
        num_workers=args.num_procs, pin_memory=True)
    return img_loader, result, weight


def gaussian_blend(sz):
    # Gaussian blending
    zz, yy, xx = np.meshgrid(np.linspace(-1, 1, sz[0], dtype=np.float32),
                             np.linspace(-1, 1, sz[1], dtype=np.float32),
                             np.linspace(-1, 1, sz[2], dtype=np.float32), indexing='ij')

    dd = np.sqrt(zz * zz + yy * yy + xx * xx)
    sigma, mu = 0.2, 0.0
    ww = 1e-6 + np.exp(-((dd - mu) ** 2 / (2.0 * sigma ** 2)))
    print('Gaussian Blending Weight Shape: {}.'.format(ww.shape))

    return ww


def load_model(args, device):
    print(args.model)
    #model = UNet_PNI(in_planes=1, out_planes=3)
    model = UNet_PNI(in_planes=1, out_planes=3, decode_ratio = args.decode_ratio,\
                     bn_mode=args.bn_mode, relu_mode=args.relu_mode, init_mode=args.init_mode)
    model = DataParallelWithCallback(model, device_ids=range(args.num_gpu))
    print("Loading model to device: {}.".format(device))
    model = model.to(device)
    print("Finished.")
    print("Loading model state from file {}...".format(args.model))
    #model.load_state_dict(torch.load(args.model))
    #model.load_state_dict(torch.load(args.model)['state_dict'])
    #import pdb;pdb.set_trace()
    model.load_state_dict(load_checkpoint(args.model, args.num_gpu)['state_dict'])
    print("Finished loading.")
    return model


def test(args, test_loader, result, weight, model, device, model_io_size):
    z_pad = args.z_pad
    y_pad = args.y_pad
    x_pad = args.x_pad
    # switching model to eval mode
    model.eval()
    volume_id = 0
    ww = gaussian_blend(model_io_size)

    print('Starting prediction...')
    start = time.time()
    with torch.no_grad():
        for i, (pos, volume, _, _, _) in enumerate(test_loader):
            volume = volume.to(device)
            output = model(volume)

            if i == 0:
                print("I/O information:")
                print("Input Volume size: {}.".format(volume.size()))
                print("Output Volume size: {}.".format(output.size()))

            volume_id += args.batch_size
            print('Performing prediction on volume_id: {}...'.format(volume_id))
            print('Statistics:')
            print('Input Volume min: {}.'.format(volume.min()))
            print('Input Volume max: {}.'.format(volume.max()))
            print('Output Volume min: {}.'.format(output.min()))
            print('Output Volume max: {}.'.format(output.max()))
            sz = tuple([3] + list(model_io_size))
            for idx in range(output.size()[0]):
                st = pos[idx]
                result[st[0]][:, st[1]:st[1] + sz[1], st[2]:st[2] + sz[2], st[3]:st[3] + sz[3]] += \
                    output[idx].cpu().detach().numpy().reshape(sz) * np.expand_dims(ww, axis=0)
                weight[st[0]][st[1]:st[1] + sz[1], st[2]:st[2] + sz[2], st[3]:st[3] + sz[3]] += ww

    end = time.time()
    print("Finished.")
    print("Prediction time: {}.".format(end - start))
    print("Saving results...")
    for vol_id in range(len(result)):
        print("Result with vol_id: {}...".format(vol_id))
        result[vol_id] = result[vol_id] / np.expand_dims(weight[vol_id], axis=0)
        print('Statistics:')
        print('Input Volume min: {}.'.format(result[vol_id].min()))
        print('Input Volume max: {}.'.format(result[vol_id].max()))
        # data = (result[vol_id]*255).astype(np.uint8)
        # data[data < 128] = 0
        hf = h5py.File(args.output_dir + '/volume_' + str(vol_id) + '.h5', 'w')
        hf.create_dataset('main', data=result[vol_id][:, z_pad:-z_pad, y_pad:-y_pad, x_pad:-x_pad])
        hf.close()
        print("Saved vol_id {}.".format(vol_id))


def main():
    args = get_args()

    print('0. initial setup')
    check_output(args)
    model_input_shape = get_model_input_shape(args)
    device = get_preferred_device(args)

    print('1. data setup')
    test_loader, result, weight = load_volumes(args, model_input_shape)

    print('2. model setup')
    model = load_model(args, device)

    print('3. testing')
    test(args, test_loader, result, weight, model, device, model_input_shape)

    print('Finished.')


if __name__ == "__main__":
    main()
