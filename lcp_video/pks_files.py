"""
The library of codes used to work with .roi.hdf5 file generate by scattering experiments

The roi files
"""
import os
os.system('export HDF5_USE_FILE_LOCKING=FALSE')

def list_dataset(filename):
    """
    filename is a raw.hdf5 file, any of them. The function returns  all filenames associated with the dataset.

    the dataset raw.hdf5 file structure is the following:
    camera-name_dataset-name_chunk-number.raw.hdf5
    """
    import os
    import numpy as np


    root,camera_name,dataset_name,chunk,extension = split_roi_filename(filename)
    lst_dir = os.listdir(root)
    lst_dataset = []
    lst_order = []
    for item in lst_dir:
        if item.find(camera_name+'_'+dataset_name + '_') != -1:
            rt,camera_name,dataset_name,chunk,extension = split_roi_filename(item)
            if extension == "pks.hdf5":
                lst_dataset.append(os.path.join(root,item))
                lst_order.append(int(chunk))
    lst_sorted = np.argsort(np.array(lst_order))
    return list(np.array(lst_dataset)[lst_sorted])

def split_roi_filename(filepath):
    import os
    root, name = os.path.split(filepath)
    head, ext1, ext2 = name.split('.')
    extension = ext1+'.'+ext2
    camera_name, dataset_name, chunk = head.split('_')
    return root,camera_name,dataset_name,chunk,extension

def get_frameids(filename):
    """
    returns a vector of frameIDs from a given dataset.
    """
    import numpy as np
    import h5py
    pathnames = list_dataset(filename)
    with h5py.File(pathnames[0],'r') as f:
        length = len(pathnames)*f['frameIDs'].shape[0]

    arr = np.zeros((length,))
    for pathname in pathnames:
        index = pathnames.index(pathname)
        with h5py.File(pathname,'r') as f:
            arr[index*256:(index+1)*256] = f['frameIDs'][()]
    return arr

def get_timestamps_lab(filename):
        """
        returns a vector of frameIDs from a given dataset.
        """
        import numpy as np
        import h5py
        pathnames = list_dataset(filename)
        with h5py.File(pathnames[0],'r') as f:
            length = len(pathnames)*f['timestamps_lab'].shape[0]
        arr = np.zeros((length,))
        for pathname in pathnames:
            index = pathnames.index(pathname)
            with h5py.File(pathname,'r') as f:
                arr[index*256:(index+1)*256] = f['timestamps_lab'][()]
        return arr

def get_chunk_temperature(filename):
    """
    returns a vector of frameIDs from a given dataset.
    """
    import numpy as np
    import h5py
    pathnames = list_dataset(filename)
    with h5py.File(pathnames[0],'r') as f:
        length = len(pathnames)

    arr = np.zeros((length,))
    for pathname in pathnames:
        index = pathnames.index(pathname)
        with h5py.File(pathname,'r') as f:
            arr[index] = f['temperature'][()]
    return arr

def get_chunk_frame_pair_from_time(pks_filename,t = 0):
    """
    returns the closest frame number and a chunk number to a give time.
    """
    from lcp_video.pks_files import get_timestamps_lab
    from numpy import gradient
    timestamps = get_timestamps_lab(pks_filename)
    mask = gradient(timestamps > t)>0
