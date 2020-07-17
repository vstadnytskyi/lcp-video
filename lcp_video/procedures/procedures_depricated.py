

def compress_hdf5_file(filenames, destination_root):
    """
    """
    from h5py import File
    from numpy import ndarray
    for filename in filenames:
        with File(filename,'r') as f:
            prefix = filename.split('.hdf5')[0]
            with File(prefix+'.gzip.hdf5','a') as fnew:
                for key in list(f.keys()):
                    data = f[key]
                    if key == 'images':
                        fnew.create_dataset(key, data = data, compression = 'gzip', chunks = (1,200,256))
                    else:
                        fnew.create_dataset(key, data = data)






def get_spots_mask(root,data,stats_ref,stats_dark):
    """
    """
    from h5py import File
    from ubcs_auxiliary.save_load_object import load_from_file
    from skimage import measure
    from numpy import where, zeros_like, bool, zeros,sqrt, ones
    from time import time
    stats_ref = load_from_file(root+'camera_8mm_Reference.stats.pkl')
    stats_dark = load_from_file(root+'/camera_8mm_Dark.stats.pkl')
    data = root + 'camera_8mm_Spray.gzip.hdf5'
    mask_filename = root + 'camera_8mm_Spray.mask.gzip.hdf5'
    M1 = stats_ref['M1']
    M2 = stats_ref['M2']
    M3 = stats_ref['M3']
    row = 3000
    col = 4096
    N_images = 600
    sigma = 6
    slit_mask = ones((row,col))
    slit_mask[:,3950:] = 0
    N_count = zeros(N_images)
    t1 = time()
    with File(data,'r') as f_data:
        with File(mask_filename,'a') as f_mask:
            N_images,row,col = (600,3000,4096)
            f_mask.create_dataset('mask', (N_images,3000,4096), dtype = 'int8', chunks = (1,3000,4096), compression = 'gzip')
            f_mask.create_dataset('blobs', (N_images,3000,4096), dtype = 'int16', chunks = (1,3000,4096), compression = 'gzip')
            images = f_data['images']
            skew_threshold = 0.3
            normal_mask = (M3 > -skew_threshold) & (M3 < skew_threshold)
            threshold = []

            for i in range(N_images):
                t0 = time()
                mask = zeros((3000,4096))
                for j in [2,3,4,5,6]:
                    if j == 2:
                        c = 2
                    else:
                        c = 1
                    mask += c*(((images[i] - M1) > j*sqrt(M2))*normal_mask*slit_mask)
                blobs = measure.label(mask==6).astype('int16')
                f_mask['blobs'][i] = blobs
                f_mask['mask'][i] = mask

                print(i,time()-t0, blobs.max())
    t2 = time()
    print(t2-t1)


def rename_old_files(root):
    pass
