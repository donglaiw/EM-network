import os,sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import h5py
import argparse
import torch
import torch.utils.data
import torchvision.utils as vutils
import gc

# tensorboardX
from tensorboardX import SummaryWriter

from em_net.data.dataset import OnehotDataset, collate_fn
#from em_net.data.dataset import AffinityDataset, collate_fn, build_augmentor, build_sampler
from em_net.model.io import load_checkpoint,save_checkpoint
from em_net.model.loss import WeightedBCELoss
from em_net.model.unet import UNet_PNI

from em_net.data.io import loadVolume

from em_net.libs.sync import DataParallelWithCallback
from em_net.util.options import *
from em_net.optimizer.monitor import monitor_lr

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

def load_data(args):
    # Parse all the input paths to both train and label volumes
    # Parse the ratio of the input data to be used for training
    # Make sure they are equal in length
    assert len(args.input_vol) == len(args.label_vol)

    # Load the volumes
    train_input, train_label, validation_input, validation_label = \
            loadVolume(args.input_vol, args.label_vol, args.data_chunk, \
                       args.train_ratio, args.invalid_mask)

    # Create Pytorch Datasets
    augmentor = 1
    train_dataset = OnehotDataset(volume=train_input, label=train_label, sample_input_size=args.data_shape,
                                    sample_label_size=args.data_shape, augmentor=augmentor, mode='train')
    valid_dataset = OnehotDataset(volume=validation_input, label=validation_label, sample_input_size=args.data_shape,
                                    sample_label_size=args.data_shape, augmentor=augmentor, mode='train')
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
    model = UNet_PNI(in_planes=1, out_planes=3, decode_ratio = args.decode_ratio,\
                     bn_mode=args.bn_mode, relu_mode=args.relu_mode, init_mode=args.init_mode)
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


def train(args, train_loader, validation_loader, model, device, criterion, optimizer, monitor, logger, writer):
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

        """
        if True: #volume_id % args.volume_valid < args.batch_size or volume_id >= args.volume_total:
            writer.add_image('Train Input', vutils.make_grid(volume.data.cpu()[:,0,volume.shape[2]//2:volume.shape[2]//2+1]), volume_id)
            writer.add_image('Train Output', vutils.make_grid(output.data.cpu()[:,1,volume.shape[2]//2:volume.shape[2]//2+1]), volume_id)
            writer.add_image('Train Label', vutils.make_grid(label.data.cpu()[:,1,volume.shape[2]//2:volume.shape[2]//2+1]), volume_id)
            writer.add_image('Train Weight', vutils.make_grid(class_weight.data.cpu()[:,1,volume.shape[2]//2:volume.shape[2]//2+1]==0), volume_id)
            # from scipy.misc import imsave
            # imsave('hh.png', np.clip(class_weight.data.cpu().numpy()[1,1,0]*255,0,255).astype(np.uint8))
            import pdb; pdb.set_trace()
        """

        print("[Volume %d] train_loss=%0.4f lr=%.5f" % (volume_id, loss.item(), optimizer.param_groups[0]['lr']))

        writer.add_scalar('Training Loss', loss.item(), volume_id)
        logger.write("[Volume %d] train_loss=%0.4f lr=%.5f\n" % (volume_id,
                                                                 loss.item(), optimizer.param_groups[0]['lr']))
        del volume, label, output, class_weight

        # Get the validation result if it's time
        # use the same variable name to reduce GPU memory usage
        if volume_id % args.volume_valid < args.batch_size or volume_id >= args.volume_total:
            # global running mean is so bad
            #model.eval()
            with torch.no_grad():
                _, volume, label, class_weight, _ = next(val_data_iter)
                volume, label = volume.to(device), label.to(device)
                class_weight = class_weight.to(device)
                output = model(volume)
                loss = criterion(output, label, class_weight)

                writer.add_scalar('Validation Loss', loss.item(), volume_id)
                writer.add_image('Validation Input', vutils.make_grid(volume.detach()[:,0,volume.shape[2]//2:volume.shape[2]//2+1]), volume_id)
                writer.add_image('Validation Label', vutils.make_grid(label.detach()[:,0,volume.shape[2]//2:volume.shape[2]//2+1]), volume_id)
                writer.add_image('Validation Output', vutils.make_grid(output.detach()[:,1,volume.shape[2]//2:volume.shape[2]//2+1]), volume_id)
                logger.write("validation_loss=%0.4f lr=%.5f\n" % (loss.item(), optimizer.param_groups[0]['lr']))
                monitor.add(loss.item())
                print('[Valid %d] Validation Loss = %0.4f'%(monitor.val_id,loss.item()))

                if monitor.toChange(): # easier to reconstruct the optimizer
                    logger.write("time to halve learning rate \n")
                    
                    if monitor.num_change > args.lr_halve_time:
                        break # time to exit: early stopping
                    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate*(0.5**monitor.num_change), betas=(0.9, 0.999),
                                                 eps=0.01, weight_decay=1e-6, amsgrad=True)

                del volume, label, output, class_weight
            #model.train()

        if volume_id % args.volume_save < args.batch_size or volume_id >= args.volume_total:
            # Save the model if it's time.
            print("Saving the model in {}....".format(args.output_dir + ('/volume_%d.pth' % (volume_id))))
            save_checkpoint(model, args.output_dir+('/volume_%d.pth' % (volume_id)), optimizer, volume_id)
        
        gc.collect()

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

    monitor = monitor_lr(step_bin=args.lr_halve_bin, step_wait=args.lr_halve_wait,\
                            thres=args.lr_halve_thres, step_max=args.lr_halve_max)

    print('4. start training')
    train(args, train_loader, valid_loader, model, device, criterion, optimizer, monitor, logger, writer)

    print('5. finish training')
    logger.close()
    writer.close()


if __name__ == "__main__":
    main()
