import sys
import numpy as np
import h5py
import torch

# 1. model i/o
def save_checkpoint(model, filename='checkpoint.pth', optimizer=None, epoch=1):
    # model or model_state
    out = model if type(model) is dict else model.state_dict()
         
    if optimizer is None:
        torch.save({
            'epoch': epoch,
            'state_dict': out,
        }, filename)
    else:
        torch.save({
            'epoch': epoch,
            'state_dict': out,
            'optimizer' : optimizer.state_dict()
        }, filename)

def convert_state_dict(state_dict, num_gpu):
    # multi-single gpu conversion
    if num_gpu==1 and list(state_dict.keys())[0][:7]=='module.':
        # modify the saved model for single GPU
        tobe_popped = []
        # Should not modify state_dict during iteration
        # otherwise will have erderedDict mutated during iteration
        for k,v in state_dict.items():
            tobe_popped.append(k)
        for k in tobe_popped:
            state_dict[k[7:]] = state_dict[k]
            state_dict.pop(k,None)            
    elif num_gpu>1 and (len(list(state_dict.keys())[0])<7 or list(state_dict.keys())[0][:7]!='module.'):
        # modify the single gpu model for multi-GPU
        tobe_popped = []
        for k,v in state_dict.items():
            tobe_popped.append(k)
        for k in tobe_popped:
            state_dict['module.'+k] = state_dict[k]
            state_dict.pop(k,None)

def load_checkpoint(snapshot, num_gpu=1):
    if isinstance(snapshot, str):
        cp = torch.load(snapshot)
        if type(cp) is not dict:
            # model -> state_dict
            cp={'epoch':0, 'state_dict': cp.state_dict()}
    else:
        cp={'epoch':0, 'state_dict': snapshot}
    convert_state_dict(cp['state_dict'], num_gpu)
    return cp
