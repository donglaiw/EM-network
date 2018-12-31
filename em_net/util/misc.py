import sys
import numpy as np
import h5py
import random
import os

from subprocess import check_output

# 3. evaluation
def runBash(cmd):
    fn = '/tmp/tmp_'+str(random.random())[2:]+'.sh'
    print('tmp bash file:',fn)
    writetxt(fn, cmd)
    os.system('chmod 755 %s'%fn)
    out = check_output([fn])
    os.remove(fn)
    print(out)
