from em_segLib.seg_util import markInvalid
from em_net.util.io import readh5
import numpy as np

def loadVolume(input_vol_paths, label_vol_paths, \
            chunk=None,train_ratio=[1], mark_invalid=[False]):
    # Load the vols
    train_input = []
    train_label = []
    valid_input = []
    valid_label = []
    if chunk is None:
        chunk = [None] * len(label_vol_paths)
    print("Loading vols...")
    for i in range(len(input_vol_paths)):
        # input: uint8
        input_fid = readh5(input_vol_paths[i], do_np=False)
        label_fid = readh5(label_vol_paths[i], do_np=False)
        # chunk: first and last slice index
        if chunk[i] is None: # read full chunk
            chunk[i] = [0,input_fid.shape[0]-1]
        for j in range(len(chunk[i])//2):
            input_vol = np.array(input_fid[chunk[i][j*2]:chunk[i][j*2+1]+1]).astype(np.float32) / 255.0
            print("Loaded {}".format(input_vol_paths[i], chunk[i][j*2:j*2+2]))
            print_vol_stats(input_vol, "input_vol")

            label_vol = np.array(label_fid[chunk[i][j*2]:chunk[i][j*2+1]+1])
            print("Loaded {}".format(label_vol_paths[i], chunk[i][j*2:j*2+2]))
            print_vol_stats(label_vol, "label_vol")

            assert input_vol.shape == label_vol.shape

            if mark_invalid[i]:
                label_vol = markInvalid(label_vol)
        
            # Divide both input vol and label vol to train and valid sets
            # train_ratio=0: all for valid
            # train_ratio=1: all for training
            div_point = int(train_ratio[i] * input_vol.shape[0])
            train_input.append(input_vol[: div_point])
            valid_input.append(input_vol[div_point:])

            train_label.append(label_vol[: div_point])
            valid_label.append(label_vol[div_point:])

    return train_input, train_label, valid_input, valid_label

def print_vol_stats(volume, name):
    print('Statistics for {}:'.format(name))
    print('Shape: {}'.format(volume.shape))
    print('Min: {}'.format(volume.min()))
    print('Max: {}'.format(volume.max()))
    print('Mean: {}'.format(volume.mean()))



