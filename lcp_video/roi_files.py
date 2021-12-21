"""
The library of codes used to work with .roi.hdf5 file generate by scattering experiments

The roi files
"""
import os
os.system('export HDF5_USE_FILE_LOCKING=FALSE')

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


def image_from_roi(filename, frame):
    """
    returns image array from give roi file for given frame number.

    Reconstructs images and hits from roi and hits_coord. If frame = -1,
    returns 3D versions; if a non-negative integer returns image and hits for a
    single frame specified by 'frame'.

    returns a list of unique datasets in a directory

    Parameters
    ----------
    f :: (h5py file handler)
    frame :: (integer)

    Returns
    -------
    images :: (array)
    hits :: (array)

    Examples
    --------
    >>> moments = get_moments(image = image)

    """
    from numpy import zeros
    from time import time
    import h5py
    t0 = time()
    with h5py.File('filename','r') as f:
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
    #print('time to reconstruct images,hits [s]: {:0.3f}'.format(time()-t0))
    return images,hits

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
            if extension == "roi.hdf5":
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

def generate_mono_video(filename, verbose = False, scale = 'log', rotate = 0, fliplr = False, N = 3e9):
    """
    generates video from a set of raw.hdf5 files specified by filename.
    """
    import h5py
    import cv2
    import numpy as np
    from time import ctime, time
    import os

    root,camera_name,dataset_name,chunk,extension = split_roi_filename(filename)
    if scale == 'linear':
        video_pathname = os.path.join(root,camera_name +'_'+dataset_name+'.linear_mono.mp4')
    elif scale == 'log':
        video_pathname = os.path.join(root,camera_name +'_'+dataset_name+'.log_mono.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')

    f = h5py.File(filename,'r')
    fps = 10**6/f['exposure time'][()]
    width = f['image width'][()]
    height = f['image height'][()]
    video = cv2.VideoWriter(video_pathname, fourcc, fps, (width, height))
    img = np.rot90(np.zeros((height,width,3), dtype = 'uint8'),rotate)
    from lcp_video.analysis import mono12p_to_image
    pathnames = list_dataset(filename)
    length = f['shape'][()][0]
    j = 0
    for pathname in pathnames:
        with h5py.File(pathname,'r') as f:
            i = 0
            while i < length:
                if verbose:
                    print(f"{ctime(time())}: converting frame with ID = {f['frameIDs'][i]}")
                if scale == 'linear':
                    data = image_from_roi(f,i)[0]
                    green = ((np.rot90(data,rotate)/4095)*255).astype('uint8')
                elif scale == 'log':
                    data = image_from_roi(f,i)[0]
                    #print(data.mean())
                    green = (np.log2(np.rot90(data,rotate))*255/12).astype('uint8')
                img = img*0 + 0
                img[:,:,1] = green
                # font
                font = cv2.FONT_HERSHEY_SIMPLEX
                org = (50, 50)
                fontScale = 1
                color = (0, 0, 255,0) #CMYK?
                thickness = 2
                img = cv2.putText(img,f'{dataset_name} {ctime(f["timestamps_lab"][()][i])} frame# = {j}', org, font,
                                   fontScale, color, thickness, cv2.LINE_AA)
                video.write(img)
                j+=1
                i+=1
                if j>N:
                    break
        if j>N:
            break

    cv2.destroyAllWindows()
    video.release()
    if verbose:
        print('done')

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

def get_timestamps_camera(filename):
    """
    returns a vector of frameIDs from a given dataset.
    """
    import numpy as np
    import h5py
    import sys
    pathnames = list_dataset(filename)
    with h5py.File(pathnames[0],'r') as f:
        length = len(pathnames)*f['timestamps_camera'].shape[0]
    arr = np.zeros((length,))
    for pathname in pathnames:
        index = pathnames.index(pathname)
        with h5py.File(pathname,'r') as f:
            arr[index*256:(index+1)*256] = f['timestamps_camera'][()]
        number = round((index/len(pathnames))*100,1)
        text = f"\r processing status: {number} % done"
        sys.stdout.write(text)
        sys.stdout.flush()
    return arr

def get_chunk_frame_pair_from_time(pks_filename,t = None):
    """
    returns the closest frame number and a chunk number to a give time.
    """

    #get_timestamps_lab
    pass


def get_hits1(filename):
    """
    returns a vector of frameIDs from a given dataset.
    """
    import numpy as np
    import h5py
    import sys
    pathnames = list_dataset(filename)
    with h5py.File(pathnames[0],'r') as f:
        length = len(pathnames)*f['hits1'].shape[0]
    arr1 = np.zeros((length,))
    arr2 = np.zeros((length,))
    for pathname in pathnames:
        index = pathnames.index(pathname)
        with h5py.File(pathname,'r') as f:
            arr1[index*256:(index+1)*256] = f['timestamps_camera'][()]
            arr2[index*256:(index+1)*256] = f['hits1'][()]
        number = round((index/len(pathnames))*100,1)
        text = f"\r processing status: {number} % done"
        sys.stdout.write(text)
        sys.stdout.flush()
    return arr1,arr2

def get_imax_in_range(pathname,rng = [0,1]):
    """
    """
    from h5py import File
    from lcp_video.roi_files import list_dataset
    import numpy as np
    with File(pathname,'r') as f:
        imax = f['Imax'][()]
    imax *= 0
    filenames = list_dataset(pathname)
    for filename in filenames[rng[0]:rng[1]]:
        with File(filename,'r') as f:
            imax = np.maximum(imax, f['Imax'][()])
    string = str(filenames[rng[0]])
    string += '\n'
    string += str(filenames[rng[1]])

    return imax, string

def plot_imax_in_range(pathname, rng = [0,1]):
    imax, string = get_imax_in_range(pathname,rng)
    from matplotlib import pyplot as plt
    import numpy as np
    plt.figure()
    plt.imshow(np.log10(imax))
    plt.colorbar()
    plt.title(string)
