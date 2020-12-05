"""
The library of codes used to work with .roi.hdf5 file generate by scattering experiments

The roi files
"""

def images_hits_reconstruct(roi_name,frame=-1):
    """Reconstructs images and hits from roi and hits_coord. If frame = -1,
    returns 3D versions; if a non-negative integer returns image and hits for a
    single frame specified by 'frame'."""
    from numpy import zeros
    from time import time
    import h5py
    t0 = time()
    f = h5py.File(roi_name,'r')
    shape = f['shape']
    mask = f['mask']
    hits_coord = f['hits_coord']
    roi = f['roi']
    hits = zeros(shape,bool)
    hits[tuple(hits_coord)] = True
    if frame == -1:
        images = zeros(shape,dtype='int16')
        images[:,mask] = roi
        images = images.reshape(shape)
    else:
        hits = hits[frame]
        shape = f['mask'].shape
        images = zeros(shape,dtype='int16')
        images[mask] = roi[frame]
    print('time to reconstruct images,hits [s]: {:0.3f}'.format(time()-t0))
    return images,hits
