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
    filename is a img.hdf5 file, any of them. The function returns  all filenames associated with the dataset.

    the dataset img.hdf5 file structure is the following:
    camera-name_dataset-name_chunk-number.img.hdf5
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
            if extension == "img.hdf5":
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


def create_treshold(filename, chunk):
    pass

def func(filename):
     import numpy as np
     from h5py import File
     f = File(filename)
     data = f['images'][()]
     sorted = np.sort(data, axis = 0)
     from ubcs_auxiliary import save_load_object
     from os import path
     head, tail = path.split(filename)
     dst_name = tail.split('.')[0]
     dic = {}
     dic['imax'] = sorted[254:]
     dic['mean'] = np.mean(sorted[2:-2],axis =0)
     dic['std'] = np.std(sorted[2:-2], axis = 0)
     dic['imin'] = sorted[:2]
     save_load_object.save_to_file(path.join(head, dst_name + '.dict.npy'),dic)
     #save_load_object.save_to_file(path.join(head, dst_name + '.data.npy'),data)

class DataSet(object):
    def __init__(self,pathname):
        from os import path
        from lcp_video import img_files
        head, tail = path.split(pathname)
        self.root = head
        dataset_name = tail.split('.')[0]
        dataset_parts = dataset_name.split('_')
        name = "_".join(dataset_parts[:-1])
        self.pathname = pathname
        self.name = name
        self.list = img_files.list_dataset(pathname)
        self.frame_ids = img_files.get_frameids(pathname).astype('int64')


    def get_frame_by_id(self, id):
        """
        """
        import numpy as np
        from h5py import File
        import os
        if id <= self.frame_ids[-1]:
            idx = np.where(self.frame_ids == id)[0][0]
            chunk = int(idx/256)
            frame = idx - chunk*256
            filename = os.path.join(self.root, self.name + f'_{chunk}.img.hdf5')
            with File(filename,'r') as f:
                img = f['images'][frame]
        else:
            img = None
        return img

    def get_frameid_by_id(self, id):
        """
        """
        import numpy as np
        from h5py import File
        import os
        if id <= self.frame_ids[-1]:
            idx = np.where(self.frame_ids == id)[0][0]
            chunk = int(idx/256)
            frame = idx - chunk*256
            filename = os.path.join(self.root, self.name + f'_{chunk}.img.hdf5')
            with File(filename,'r') as f:
                frame_id = f['frameIDs'][frame]
        else:
            frame_id = None
        return frame_id

def extract_channel_archiver_data():
    """
    extract channel archiver data for given set of PVs. Creates a separate log file with a separate entry for each frameID.

    get the list of all .img. files that conform to the selected
    """
    pass
