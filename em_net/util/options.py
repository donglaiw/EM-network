import os,argparse
import datetime

import numpy as np
from em_dataLib.options import optIO, optDataAug

def optSystem(parser):
    parser.add_argument('-g', '--num-gpu', type=int, default=1,
                        help='Number of CUDA-enabled graphics cards to be used to train the model.')
    parser.add_argument('-j', '--num-procs', type=int, default=1,
                        help='Number of processes to be used for the training data loader. The validation data loader' +
                        'will use only one process to load.')
    parser.add_argument('-bs', '--batch-size', type=int, default=1,
                        help='Batch size.')

def optTrain(parser):
    # training loss
    parser.add_argument('-l','--loss-opt', type=int, default=0,
                        help='loss type')
    parser.add_argument('-lw','--loss-weight-opt', type=float, default=2.0,
                        help='weighted loss type')
    # optimization
    parser.add_argument('-lr', '--learning-rate', type=float, default=0.0001,
                        help='learning rate')
    parser.add_argument('-lr_decay', default='inv,0.0001,0.75',
                        help='learning rate decay')
    parser.add_argument('-betas', default='0.99,0.999',
                        help='beta for adam')
    parser.add_argument('-wd', type=float, default=5e-6,
                        help='weight decay')
    # pre-train
    parser.add_argument('-pe', '--pre-epoch', type=int, default=0,
                        help='pre-train number of epoch')
    parser.add_argument('-pm', '--pre-model', type=str, default='',
                        help='Pre-trained model path')
    # logging
    parser.add_argument('--volume-total', type=int, default=70000,
                        help='Total number of iteration')
    parser.add_argument('--volume-save', type=int, default=10000,
                        help='Number of iterations for the script to save the model.')
    parser.add_argument('--volume-valid', type=int, default=1000,
                        help='Number of iterations for the script to validate the model.')


def optModel(parser):
    parser.add_argument('-m','--model-id',  type=float, default=0,
                        help='model id')
    parser.add_argument('-ma','--opt-arch', type=str,  default='0,0@0@0,0,0@0',
                        help='model type')
    parser.add_argument('-mp','--opt-param', type=str,  default='0@0@0@0',
                        help='model param')
    parser.add_argument('-mi','--model-input', type=str,  default='31,204,204',
                        help='model input size')
    parser.add_argument('-mo','--model-output', type=str,  default='3,116,116',
                        help='model input size')
    parser.add_argument('-f', '--num-filter', default='24,72,216,648',
                        help='number of filters per layer')
    parser.add_argument('-ps', '--pad-size', type=int, default=0,
                        help='pad size')
    parser.add_argument('-do', '--has-dropout', type=float, default=0,
                        help='use dropout')

    parser.add_argument('-rl', '--relu-mode', type=float, default=0.005,
                        help='relu mode')
    parser.add_argument('-bn', '--batch-mode', type=int, default=0,
                        help='BatchNorm mode')
    parser.add_argument('-pt', '--pad-mode', default='constant,0',
                        help='pad mode')
    parser.add_argument('-it','--init-mode', type=int,  default=-1,
                        help='model initialization type')


def optParse(args):
    # additional parsing

    # model input shape
    if args.init_mode >= 0:
        args.init_mode = ['kaiming_normal','kaiming_uniform','xavier_normal','xavier_uniform'][args.init_mode]
    else:
        args.init_mode = ''
    # model input shape
    args.input_shape = np.array([int(x) for x in args.input_shape.split(',')])

    # output folder
    tt = str(datetime.datetime.now()).split(' ')
    date = tt[0]
    time = tt[1].split('.')[0].replace(':','-')
    args.output_dir += '/log_' + date + '_' + time
    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)
        print('Output directory was created.')

