import os,sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import h5py
import argparse
import torch
import torch.utils.data
import torchvision.utils as vutils

# tensorboardX
from tensorboardX import SummaryWriter

from em_net.data.dataset import AffinityDataset, collate_fn
#from em_net.data.dataset import AffinityDataset, collate_fn, build_augmentor, build_sampler
from em_net.model.loss import WeightedBCELoss
from em_net.model.unet import UNet_PNI
from em_net.libs.sync import DataParallelWithCallback
from em_net.util.options import *

def get_args():
    parser = argparse.ArgumentParser(description='A script for training the PNI 3D UNET model for predicting ' +
                                                 'affinities.')
    # I/O options------------------------------------------------------------------------------------------------------#
    optIO(parser, 'train')
    # data options-----------------------------------------------------------------------------------------------------#
    optDataAug(parser)
    # model options----------------------------------------------------------------------------------------------------#
    optModel(parser)
    # optimization options---------------------------------------------------------------------------------------------#
    optTrain(parser)
    # system options---------------------------------------------------------------------------------------------#
    optSystem(parser)

    # initial parser---------------------------------------------------------------------------------------------------#
    args = parser.parse_args()
    # additional parser------------------------------------------------------------------------------------------------#

    optParse(args)
    return args


def get_device(args):
    if args.num_gpu < 0:
        raise ValueError("The number of GPUs must be greater than or equal to zero.")
    return torch.device("cuda" if torch.cuda.is_available() and args.num_gpu > 0 else "cpu")


def print_volume_stats(volume, name):
    print('Statistics for {}:'.format(name))
    print('Shape: {}'.format(volume.shape))
    print('Min: {}'.format(volume.min()))
    print('Max: {}'.format(volume.max()))
    print('Mean: {}'.format(volume.mean()))


def load_data(args):
    # Parse all the input paths to both train and label volumes
    input_volume_paths = args.input_volume.split('@')
    label_volume_paths = args.label_volume.split('@')
    # Parse the ratio of the input data to be used for training
    train_ratio = args.train_data_ratio
    # Make sure they are equal in length
    assert len(input_volume_paths) == len(label_volume_paths)

    # Load the volumes
    train_input = []
    train_label = []
    validation_input = []
    validation_label = []
    print("Loading volumes...")
    for i in range(len(input_volume_paths)):
        input_volume = np.array(h5py.File(input_volume_paths[i], 'r')['main']).astype(np.float32) / 255.0
        print("Loaded {}".format(input_volume_paths[i]))
        print_volume_stats(input_volume, "input_volume")

        label_volume = np.array(h5py.File(label_volume_paths[i], 'r')['main'])
        print("Loaded {}".format(label_volume_paths[i]))
        print_volume_stats(label_volume, "label_volume")

        assert input_volume.shape == label_volume.shape

        # Divide both input volume and label volume to train and validation sets
        div_point = int(train_ratio * len(input_volume))
        train_input.append(input_volume[: div_point])
        validation_input.append(input_volume[div_point:])

        train_label.append(label_volume[: div_point])
        validation_label.append(label_volume[div_point:])

    # Create Pytorch Datasets
    augmentor = 1
    train_dataset = AffinityDataset(volume=train_input, label=train_label, sample_input_size=args.input_shape,
                                    sample_label_size=args.input_shape, augmentor=augmentor, mode='train')
    valid_dataset = AffinityDataset(volume=validation_input, label=validation_label, sample_input_size=args.input_shape,
                                    sample_label_size=args.input_shape, augmentor=augmentor, mode='train')
    # Create Pytorch DataLoaders
    print('Batch size: {}'.format(args.batch_size))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                               shuffle=True, collate_fn=collate_fn,
                                               num_workers=args.num_procs, pin_memory=True)
    # TODO: Check whether this will work with args.num_procs.
    validation_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch_size,
                                                    shuffle=True, collate_fn=collate_fn,
                                                    num_workers=1, pin_memory=True)
    return train_loader, validation_loader


def load_model(args, device):
    model = UNet_PNI(in_planes=1, out_planes=3)
    model = DataParallelWithCallback(model, device_ids=range(args.num_gpu))
    print("Loading model to device: {}.".format(device))
    model = model.to(device)
    print("Finished.")
    print("Finished loading.")
    if len(args.pre_model)>0:
        model.load_state_dict(torch.load(args.pre_model))
        print('fine-tune on previous model:')
        print(args.pre_model)
    return model


def get_loggers(args):
    # Set loggers names.
    logger = open(args.output_dir + '/log.txt', 'w')  # unbuffered, write instantly

    # tensorboardX
    writer = SummaryWriter(args.output_dir)
    print("Saving Tensorboard summary to {}".format(args.output_dir))
    return logger, writer


def train(args, train_loader, validation_loader, model, device, criterion, optimizer, logger, writer):
    # switch to train mode
    model.train()
    volume_id = 0
    # Validation dataset iterator:
    val_data_iter = iter(validation_loader)

    for _, volume, label, class_weight, _ in train_loader:
        volume_id += args.batch_size

        volume, label = volume.to(device), label.to(device)
        class_weight = class_weight.to(device)
        output = model(volume)

        loss = criterion(output, label, class_weight)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print("[Volume %d] train_loss=%0.4f lr=%.5f\n" % (volume_id, loss.item(), optimizer.param_groups[0]['lr']))

        writer.add_scalar('Training Loss', loss.item(), volume_id)
        logger.write("[Volume %d] train_loss=%0.4f lr=%.5f\n" % (volume_id,
                                                                 loss.item(), optimizer.param_groups[0]['lr']))

        # Get the validation result if it's time
        # use the same variable name to reduce GPU memory usage
        if volume_id % args.volume_valid < args.batch_size or volume_id >= args.volume_total:
            _, volume, label, class_weight, _ = next(val_data_iter)
            model.eval()
            volume, label = volume.to(device), label.to(device)
            class_weight = class_weight.to(device)
            output = model(volume)
            loss = criterion(output, label, class_weight)

            writer.add_scalar('Validation Loss', loss.item(), volume_id)
            writer.add_image('Validation Input', vutils.make_grid(volume[:,0,:1]), volume_id)
            writer.add_image('Validation Output', vutils.make_grid(output[:,1,:1]), volume_id)
            writer.add_image('Validation Label', vutils.make_grid(label[:,0,:1]), volume_id)
            logger.write("validation_loss=%0.4f lr=%.5f\n" % (loss.item(), optimizer.param_groups[0]['lr']))

            model.train()

        if volume_id % args.volume_save < args.batch_size or volume_id >= args.volume_total:
            # Save the model if it's time.
            print("Saving the model in {}....".format(args.output + ('/volume_%d_%f.pth' % (volume_id, loss))))
            torch.save(model.state_dict(), args.output + ('/volume_%d_%f.pth' % (volume_id, loss)))
        # Terminate
        if volume_id >= args.volume_total:
            break  #

def main():
    args = get_args()

    print('0. initial setup')
    device = get_device(args)
    logger, writer = get_loggers(args)

    print('1. setup data')
    train_loader, valid_loader = load_data(args)

    print('2.0 setup model')
    model = load_model(args, device)

    print('2.1 setup loss function')
    criterion = WeightedBCELoss()

    print('3. setup optimizer')
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.999),
                                 eps=0.01, weight_decay=1e-6, amsgrad=True)

    print('4. start training')
    train(args, train_loader, valid_loader, model, device, criterion, optimizer, logger, writer)

    print('5. finish training')
    logger.close()
    writer.close()


if __name__ == "__main__":
    main()
