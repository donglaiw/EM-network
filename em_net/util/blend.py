

def gaussian_blend(sz, sigma, mu=0.0, offset=1e-4):
    import numpy as np

    zz, yy, xx = np.meshgrid(np.linspace(-1, 1, sz[0], dtype=np.float32),
                             np.linspace(-1, 1, sz[1], dtype=np.float32),
                             np.linspace(-1, 1, sz[2], dtype=np.float32), indexing='ij')

    dd = np.sqrt(zz*zz + yy*yy + xx*xx)
    ww = offset + np.exp(-((dd-mu)**2 / (2.0 * sigma**2)))
    return ww



