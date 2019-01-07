import h5py

def readh5(filename, dsetname='', do_np=True):
    fid = h5py.File(filename,'r')
    if len(dsetname) == 0:
        dsetname = [k for k in fid.keys()][0]
    if do_np:
        return np.array(fid[dsetname])
    else:
        return fid[dsetname]
