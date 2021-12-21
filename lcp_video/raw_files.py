"""
The library of codes used to work with .raw.hdf5 file generate by scattering experiments

"""

def convert_timestamp(timestamp):
    from dateimt import datetime
    string = datetime.fromtimestamp(timestamp).strftime("%A, %B %d, %Y %I:%M:%S.%f")
    return sting

def split_raw_filename(filepath):
    import os
    root, name = os.path.split(filepath)
    head, ext1, ext2 = name.split('.')
    extension = ext1+'.'+ext2
    camera_name, dataset_name, chunk = head.split('_')
    return root,camera_name,dataset_name,chunk,extension

def join_raw_filename(root,camera_name,dataset_name,chunk,extension):
    """
    """
    import os
    filename = camera_name +'_'+dataset_name + '_' + chunk + '.' + extension
    path = os.path.join(root,filename)
    return path

def list_dataset(filename):
    """
    filename is a raw.hdf5 file, any of them. The function returns  all filenames associated with the dataset.

    the dataset raw.hdf5 file structure is the following:
    camera-name_dataset-name_chunk-number.raw.hdf5
    """
    import os
    import numpy as np

    root,camera_name,dataset_name,chunk,extension = split_raw_filename(filename)
    lst_dir = os.listdir(root)
    lst_dataset = []
    lst_order = []
    for item in lst_dir:
        if item.find(camera_name+'_'+dataset_name + '_') != -1:
            rt,camera_name,dataset_name,chunk,extension = split_raw_filename(item)
            if extension == "raw.hdf5":
                lst_dataset.append(os.path.join(root,item))
                lst_order.append(int(chunk))
    lst_sorted = np.argsort(np.array(lst_order))
    return list(np.array(lst_dataset)[lst_sorted])

def generate_mono_video(filename, verbose = False, scale = 'log', rotate = 0, fliplr = False, N = 3e9):
    """
    generates video from a set of raw.hdf5 files specified by filename.
    """
    import h5py
    import cv2
    import numpy as np
    from time import ctime, time
    import os

    root,camera_name,dataset_name,chunk,extension = split_raw_filename(filename)
    if scale == 'linear':
        video_pathname = os.path.join(root,camera_name +'_'+dataset_name+'.linear_mono.mp4')
    elif scale == 'log':
        video_pathname = os.path.join(root,camera_name +'_'+dataset_name+'.log_mono.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')

    f = h5py.File(filename,'r')
    fps = 10**6/f['exposure time'][()]
    width = f['image width'][()]
    height = f['image height'][()]
    if rotate%2 == 0:
        video = cv2.VideoWriter(video_pathname, fourcc, fps, (width, height))
        img = np.rot90(np.zeros((height,width,3), dtype = 'uint8'),rotate)
    else:
        video = cv2.VideoWriter(video_pathname, fourcc, fps, (height,width))
        img = np.zeros((width,height,3), dtype = 'uint8')
    from lcp_video.analysis import mono12p_to_image
    pathnames = list_dataset(filename)
    length = len(pathnames)*256
    print(f"number of frames to analyze {length}")
    j = 0
    t1 = time()
    for pathname in pathnames:
        with h5py.File(pathname,'r') as f:
            i = 0
            while i < f['images'].shape[0]:
                if verbose:
                    print(f"{ctime(time())}: converting frame with ID = {f['frameIDs'][i]}")
                if scale == 'linear':
                    green = ((np.rot90(mono12p_to_image(f['images'][i],height,width,reshape = True),rotate)/4095)*255).astype('uint8')
                elif scale == 'log':
                    green = (np.log2(np.rot90(mono12p_to_image(f['images'][i],height,width,reshape = True),rotate))*255/12).astype('uint8')
                frameID = f['frameIDs'][i]
                img = img*0
                img[:,:,1] = green
                # font
                font = cv2.FONT_HERSHEY_SIMPLEX
                org = (50, 50)
                fontScale = 1
                color = (255, 0, 0)
                thickness = 2
                image = cv2.putText(img,f'{dataset_name} {ctime(f["timestamps_lab"][()][i])} frame# = {j}', org, font,
                                   fontScale, color, thickness, cv2.LINE_AA)
                video.write(img)
                j+=1
                i+=1
                if j>N:
                    break
                if j%10000 == 0:
                    t2 = time()
                    dt = t2-t1
                    print(ctime(time()))
                    print(f'last 10000 frames took {dt} seconds, with {dt/10000} seconds per frame')
                    print(f'-----------------------------------')
                    t1 = time()
        if j>N:
            break

    cv2.destroyAllWindows()
    video.release()
    if verbose:
        print('done')

def generate_rgb_video(filename, verbose = False, scale = 'log', N = 3e9):
    """
    generates video from raw files with "RGB" encoding of "previous/current/next frames"
    """
    import h5py
    import cv2
    import numpy as np
    from time import ctime, time

    root,camera_name,dataset_name,chunk,extension = split_raw_filename(filename)
    video_pathname = os.path.join(root,camera_name +'_'+dataset_name+'.rgb.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    fps = 15.67
    f = h5py.File(filename,'r')
    width = f['image width'][()]
    height = f['image height'][()]
    video = cv2.VideoWriter(video_pathname, fourcc, fps, (height, width))
    img = np.zeros((width,height,3), dtype = 'uint8')
    from lcp_video.analysis import mono12p_to_image
    pathnames = list_dataset(filename)
    for pathname in pathnames:
        with h5py.File(pathname,'r') as f:
            for i in range(f['images'].shape[0]):
                if verbose:
                    print(f"{ctime(time())}: converting frame with ID = {f['frameIDs'][i]}")
                green = ((np.fliplr(np.rot90(mono12p_to_image(f['images'][i],height,width,reshape = True),1))/4095)*255).astype('uint8')
                if i >0:
                    red = ((np.fliplr(np.rot90(mono12p_to_image(f['images'][i-1],height,width,reshape = True),1))/4095)*255).astype('uint8')
                else:
                    red = green*0
                if i < 255:
                    blue = ((np.fliplr(np.rot90(mono12p_to_image(f['images'][i-1],height,width,reshape = True),1))/4095)*255).astype('uint8')
                else:
                    blue = green*0
                img[:,:,0] = red
                img[:,:,1] = green
                img[:,:,2] = blue
                video.write(img)
            if i>N:
                break


    cv2.destroyAllWindows()
    video.release()
    if verbose:
        print('done')

def list_unique_datasets(root):
    """
    returns a list of unique datasets in a directory
    """
    raise NotImplementedError

def get_frameids(filename):
    """
    returns a vector of frameIDs from a given dataset.
    """
    import numpy as np
    import h5py
    pathnames = list_dataset(filename)
    with h5py.File(pathnames[0],'r') as f:
        length = len(pathnames)*f['images'].shape[0]

    arr = np.zeros((length,))
    for pathname in pathnames:
        index = pathnames.index(pathname)
        with h5py.File(pathname,'r') as f:
            arr[index*256:(index+1)*256] = f['frameIDs'][()]
    return arr

def get_frameids(filename):
    """
    returns a vector of frameIDs from a given dataset.
    """
    import numpy as np
    import h5py
    pathnames = list_dataset(filename)
    with h5py.File(pathnames[0],'r') as f:
        length = len(pathnames)*f['images'].shape[0]

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
            length = len(pathnames)*f['images'].shape[0]
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
    pathnames = list_dataset(filename)
    with h5py.File(pathnames[0],'r') as f:
        length = len(pathnames)*f['images'].shape[0]
    arr = np.zeros((length,))
    for pathname in pathnames:
        index = pathnames.index(pathname)
        with h5py.File(pathname,'r') as f:
            arr[index*256:(index+1)*256] = f['timestamps_camera'][()]
    return arr

def convert_raw_to_img(src_filename,dst_filename, verbose = False, numba = False):
    """
    converts .raw. data files to .img. data files preserving all original fields. The function will create a new file with the same name but different suffix
    """
    if verbose:
        print(f'source filename: {src_filename}')
        print(f'destination filename: {dst_filename}')
    from h5py import File
    from lcp_video.analysis import mono12p_to_image, mono12p_to_image_numba
    if numba:
        mono12p_to_image = mono12p_to_image_numba
    src = File(src_filename,'r')
    width =  src['image width'][()]
    height =  src['image height'][()]
    length = src['images'].shape[0]

    with File(dst_filename,'w') as dst:
        for key in src.keys():
            if "images" not in key :
                dst.create_dataset(key, data = src[key][()])
            else:
                dst.create_dataset('images', (length,height,width), dtype='int16', chunks = (1,height,width))
        for i in range(length):
            raw = src['images'][i]
            dst['images'][i] = mono12p_to_image(raw,height,width).reshape((height,width))

def convert_all_raw_to_img_once(src_root, dst_root = None, retain = True,verbose = False, overwrite = False, numba = False):
    """
    looks for .raw. files in a directory and converts them to .img. files.

    Parameters
    ----------
    src_root (string)
    dst_root (string)
    retain (boolean)
    verbose (boolean)
    overwrite (boolean)

    Returns
    -------
    """
    from time import time, sleep
    import ubcs_auxiliary as ua
    from lcp_video.raw_files import convert_raw_to_img
    lstdir = ua.os.listdir(src_root)
    import os

    lstdir = []
    for root, dirs, files in os.walk(src_root):
        for file in files:
            if file.endswith('.raw.hdf5'):
                lstdir.append(os.path.join(root, file))
    for src_filename in lstdir:
        head,tail = os.path.split(src_filename)
        timestamp = os.path.getmtime(src_filename)
        if dst_root is None:
            dst_root_l = head

        dst_filename = os.path.join(dst_root_l,tail.replace('.raw.hdf5','.img.hdf5'))
        flag = ua.os.does_filename_have_counterpart(src_filename,dst_root_l,'.img.hdf5')
        if overwrite:
            flag = False
        if not flag:
            t1 = time()
            convert_raw_to_img(src_filename,dst_filename.replace('.img.hdf5','.tmpimg.hdf5'), verbose = verbose)

            t2 = time()
            dt = t2-t1
            os.rename(dst_filename.replace('.img.hdf5','.tmpimg.hdf5'),dst_filename)
            os.utime(dst_filename,(timestamp, timestamp))
            if verbose:
                print(f'renamed to {dst_filename}')
            if not retain:
                os.remove(src_filename)
                if verbose:
                    print(f'removed {src_filename}')

            if verbose:
                print(f'conversion done in {dt} seconds')
                print(f'-----------------------------')
        else:
            pass
    return None

def convert_all_raw_to_img_always(src_root, dst_root = None, verbose = False, overwrite = False, retain = True):
    """
    """
    from time import time, sleep
    sleep_time = 60
    while True:
        t1 = time()
        convert_all_raw_to_img_once(src_root, dst_root, verbose = verbose, overwrite = overwrite, retain = retain)
        t2 = time()
        dt = t2-t1
        if dt < sleep_time:
            if verbose:
                print('waiting for files ...')
            sleep(60)



def create_treshold(filename, chunk):
    pass
