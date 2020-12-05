def get_list_of_files(source, suffix = '.hdf5', has = ''):
    """
    """
    import os
    from numpy import zeros
    tlst = os.listdir(source)
    lst = []
    for item in tlst:
        if (suffix in item) and ('.gzip.' not in item) and ('.zfs.' not in item) and (has in item):
            lst.append(item)
    arr = zeros((len(lst),))
    for i in range(len(lst)):
        arr[i] = os.path.getmtime(source + lst[i])
        idx = arr.argsort()
    lst_res = []
    for i in range(len(lst)):
        lst_res.append(lst[idx[i]])
    return lst_res

def move_files_with_compression(source, destination, suffix = '.raw.hdf5', has = '', compression = '', compression_opts = 4):
    """
    move all 'extension' files that have 'has' in their names from the 'source' directory to the 'destination' direcory with decoding and compression.

    Parameters
    ----------
    source :: (string)
    destination :: (string)
    extension :: (string)
    has :: (string)

    Returns
    -------

    Examples
    --------
    >>> from ubcs_auxiliary.multiprocessing import new_process
    >>> new_process(move_files_with_compression,'/mnt/data/','/mnt/data/2020.06.30/','.raw.hdf5','breathing')
    """
    import os
    from time import time, ctime
    from h5py import File
    lst = get_list_of_files(source = source, suffix = suffix, has = has)
    for filename in lst:
        prefix = filename.split('.raw.hdf5')[0]
        print(ctime(time()),f'moving file....')
        print(f'from{source+filename}')
        print(f'to {destination+prefix}.tmpdata.hdf5')
        t1 = time()
        with File(source+filename,'r') as f:
            with File(destination+prefix+'.tmpdata.hdf5','a') as fnew:
                for key in list(f.keys()):
                    data = f[key]
                    if key == 'images':
                        if compression == 'gzip':
                            fnew.create_dataset(key, data=data, dtype='int16',compression = compression, compression_opts = compression_opts)
                        elif compression == 'lzf':
                            fnew.create_dataset(key, data=data, dtype='int16',compression = compression)
                        else:
                            fnew.create_dataset(key, data=data, dtype='int16')
                    else:
                        fnew.create_dataset(key, data = data)
            timestamp = f['timestamps_lab'][0]
        t2 = time()
        print(ctime(time()),f'time: {t2-t1} with size {os.path.getsize(source+filename)/(1024*1024)}, speed {os.path.getsize(source+filename)/((t2-t1)*1024*1024)} MB/s')
        print(f'removing file {source+filename}')
        print(f'changing {destination+prefix}.tmpdata.hdf5')
        os.utime(destination+prefix+'.tmpdata.hdf5',(timestamp, timestamp))
        os.rename(destination+prefix+'.tmpdata.hdf5',destination+prefix+'.data.hdf5')
        os.remove(source+filename)

def move_flat_files_with_compression(source, destination, suffix = '.raw.hdf5', has = '', N = 0, compression = True, reverse = True, pixel_format = ''):
    """
    move all 'extension' files that have 'has' in their names from the 'source' directory to the 'destination' direcory with decoding and compression. This procedure converts flat .raw. files to image format.

    The raw files have data saved as 8-bit vector with no information about the size of the image. The image is reconstructed and saved accordingly in a new .data.hdf5 file. Such division of labor allows fast writing on a drive with later reshaping during data trasnfer.

    Parameters
    ----------
    source :: (string)
    destination :: (string)
    extension :: (string)
    has :: (string)

    Returns
    -------

    Examples
    --------
    >>> from ubcs_auxiliary.multiprocessing import new_process
    >>> new_process(move_flat_files_with_compression,'/mnt/data/','/mnt/data/2020.06.30/','.raw.hdf5','breathing')
    """

    import os
    from time import time, ctime
    from h5py import File
    from numpy import copy
    from ubcs_auxiliary.multiprocessing import new_child_process
    from lcp_video.analysis import mono12p_to_image, mono12packed_to_image, get_mono12packed_conversion_mask,get_mono12p_conversion_mask_8bit, get_mono12p_conversion_mask

    lst = get_list_of_files(source = source, suffix = suffix, has = has)
    if reverse:
        lst = lst[::-1]
    def once(lst):
        import os
        from time import time, ctime
        from h5py import File
        from numpy import copy
        with File(source+lst[0],'r') as f:
            width = f['image width'][()]
            height = f['image height'][()]
            length = 1
            pixel_format = f['pixel format'][()]

        for filename in lst:
            prefix = filename.split('.raw.hdf5')[0]
            print(ctime(time()),f'moving file....')
            print(f'from{source+filename}')
            print(f'to {destination+prefix}.tmpdata.hdf5')
            t1 = time()
            with File(source+filename,'r') as f:
                width = f['image width'][()]
                height = f['image height'][()]
                length = 1
                with File(destination+prefix+'.tmpdata.hdf5','a') as fnew:
                    for key in list(f.keys()):
                        data = f[key]
                        if key == 'images':
                            if compression:
                                fnew.create_dataset(key,(data.shape[0],height,width), compression='gzip', chunks=(1,height/8,width/8), dtype='int16')
                            else:
                                fnew.create_dataset(key,(data.shape[0],height,width), chunks=(1,height/8,width/8), dtype='int16')
                            for i in range(data.shape[0]):
                                if pixel_format == 'mono12p':
                                    fnew['images'][i] = mono12p_to_image(data[i],height,width).reshape((height,width))
                                elif pixel_format == 'mono12p_16':
                                    fnew['images'][i] = data[i].reshape((height,width))
                        else:
                            fnew.create_dataset(key, data = data)

                timestamp = f['timestamps_lab'][0]
            t2 = time()
            print(ctime(time()),f'time: {t2-t1} with size {os.path.getsize(source+filename)/(1024*1024)}, speed {os.path.getsize(source+filename)/((t2-t1)*1024*1024)} MB/s')
            print(f'removing file {source+filename}')
            print(f'changing {destination+prefix}.tmpdata.hdf5')
            os.utime(destination+prefix+'.tmpdata.hdf5',(timestamp, timestamp))
            os.rename(destination+prefix+'.tmpdata.hdf5',destination+prefix+'.data.hdf5')
            os.remove(source+filename)
    if N == 0:
        once(lst)
    else:
        step = int(len(lst)/(N))
        for i in range(N-1):
            sublst = (lst[i*step:(i+1)*step])
            p = new_process(once,sublst)
            print(f'{sublst} in process {i} --> {p}')
        sublst = lst[(i+1)*step:]
        p = new_process(once,sublst)
        print(f'{sublst} in process {i+1} --> {p}')

def zfs_light_data_hdf5(dark_zfs_filename, light_filename, clip = 2):
    """
    """
    from lcp_video.analysis import zinger_free_statistics
    from ubcs_auxiliary.save_load_object import load_from_file, save_to_file
    from numpy import ndarray
    from h5py import File
    print(f'Getting DarkData from {dark_zfs_filename}')
    print(f'Analysing LightData from {light_filename}')
    filename = light_filename.split(',')[0] + '.light_zfs'
    print(f'ZFS LightData will be saved to {filename}')
    dmean = load_from_file(dark_zfs_filename)['M1']
    with File(light_filename, 'r') as f:
        dic = zinger_free_statistics(f['images'],Dmean = dmean, clip = clip)
        save_to_file(filename,dic)
    filename_zfs = filename.split('.hdf5')[0] + '.light_zfs.hdf5'
    with File(filename_zfs,'a') as f_new:
        for key in list(dic.keys()):
            data = dic[key]
            if type(data) is ndarray:
                f_new.create_dataset(key, data = data, compression = 'gzip')
            else:
                f_new.create_dataset(key, data = data)
    print(f'analysis of {light_filename} is done')

def zfs_dark_data_hdf5(filename, clip):
    from h5py import File
    from numpy import ndarray
    from lcp_video.analysis import zinger_free_statistics
    with File(filename,'r') as f:
        dic = zinger_free_statistics(f['images'])
    from ubcs_auxiliary.save_load_object import save_to_file
    save_to_file(filename.split('.hdf5')[0] +'.zfs',dic)

    filename_zfs = filename.split('.hdf5')[0] + '.zfs.hdf5'
    with File(filename_zfs,'a') as f_new:
        for key in list(dic.keys()):
            data = dic[key]
            if type(data) is ndarray:
                f_new.create_dataset(key, data = data, compression = 'gzip')
            else:
                f_new.create_dataset(key, data = data)

def frame_by_frame_1(root, data_filename, dark_zfs_filename, sigma = 6):
    from h5py import File
    from numpy import ndarray, copy, bool
    with File(dark_zfs_filename, 'r') as f_dark_zfs:
        dark_M1 = copy(f_dark_zfs['M1'])
        dark_M2 = copy(f_dark_zfs['M2'])
    with File(data_filename,'r') as f_data:
        with File(data_filename.split('.hdf5')[0]+'.spots.hdf5','a') as f_analysis:
            f_analysis.create_dataset('spots', (600,3000,4096), dtype = bool, chunks = (1,3000,4096), compression = 'gzip')
            f_analysis.create_dataset('N_pixels', (600,), compression = 'gzip')
            for i in range(600):
                mask = (((f_data['images'][i]-dark_M1)/dark_M2) > sigma)
                f_analysis['spots'][i] = mask
                f_analysis['N_pixels'][i] = mask.sum()


def find_spots_sigma(filename_source, filename_destination = None, ref_dic = None, roi_mask = None):
    """
    """
    from numpy import zeros_like, rint, sqrt, maximum, where, ones
    from h5py import File
    from time import ctime, time
    from skimage import measure
    from ubcs_auxiliary.save_load_object import load_from_file
    ref_dic = load_from_file(ref_dic)
    if roi_mask is None:
        roi_mask = ones((3000,4096))
    with File(filename_source,'r') as fs:
        if filename_destination == None:
            filename_destination = filename_source.replace('.gzip.hdf5','.mask1.hdf5')
        with File(filename_destination,'a') as fd:
            length,width,height = fs['images'].shape
            fd.create_dataset('pixel_mask', (length,3000,4096), dtype = 'int8', chunks = (1,3000,4096), compression = 'gzip')
            fd.create_dataset('pixel_hits', (length,) , dtype = 'int32', compression = 'gzip')
            fd.create_dataset('particle_mask', (length,3000,4096), dtype = 'int16', chunks = (1,3000,4096), compression = 'gzip')
            fd.create_dataset('particle_hits', (length,) , dtype = 'int16', compression = 'gzip')
            M1 =  (ref_dic['S1'] - ref_dic['Imax1']- ref_dic['Imin']) /(ref_dic['N']-2)
            sigma = sqrt(ref_dic['S2']/ref_dic['N'] - M1**2)
            for i in range(length):
                image = fs['images'][i]
                mask = zeros_like(image)
                for j in range(2,7):
                    if j == 2:
                        coeff = 2
                    else:
                        coeff = 1
                    mask += (((image - M1)*roi_mask) > j*sigma)*coeff
                idx = where(mask == 6)
                fd['pixel_mask'][i] = mask
                enummask = measure.label(mask==6)
                fd['particle_mask'][i] = enummask
                fd['particle_hits'][i] = enummask.max()
                fd['pixel_hits'][i] = mask[idx].sum()

def list_find_spots_sigma(filelist,ref_dic, roi):
    from time import time, ctime
    for filesource in filelist:
        print(f'{ctime(time())} analyzing file {filesource}')
        find_spots_sigma(filesource, None, ref_dic, roi)

def calculate_spots_from_mask(data_filename,mask_filename,stats_filename,ffcorr_filename):
    """
    creates a catalog(table) of unique particles according to the supplied mask. The
    """
    from time import time,ctime
    import h5py
    from ubcs_auxiliary.save_load_object import load_from_file
    import numpy as np
    import cv2
    import os
    from lcp_video.analysis import grow_mask, get_array_piece

    temp_arr = np.zeros((21,))
    d_file = h5py.File(data_filename,'r')
    m_file = h5py.File(mask_filename,'r')
    stats_data  = load_from_file(stats_filename)
    M1 = (stats_data['S1'] - stats_data['Imax1']- stats_data['Imin']) /(stats_data['N']-2)
    FFcorr = load_from_file(ffcorr_filename)['FF_corr']
    N_spots = np.sum(m_file['particle_hits'])

    head, tail = os.path.split(data_filename)
    destination_filename = os.path.join(head, tail.split('.')[0] + '.spots.gzip.hdf5')
    with h5py.File(destination_filename,'a') as f_destination:
        f_destination.create_dataset('spots', (N_spots,21), compression = 'gzip')
        f_destination.create_dataset('spots_images', (N_spots,31,31), compression = 'gzip')
        f_destination.create_dataset('stats file', data = stats_filename)
        f_destination.create_dataset('FFcorr file', data = ffcorr_filename)
        f_destination.create_dataset('time', data = ctime(time()))
        spot_i = 0
        for i in range(600):
            print(f'{ctime(time())}: processing image {i} and saving into file {f_destination}')
            spots_N = m_file['particle_hits'][i]
            enummask = m_file['particle_mask'][i]
            image = d_file['images'][i] - M1
            t = d_file['timestamp'][i]
            for j in range(spots_N):
                mask = grow_mask((enummask == (j+1)),2)
                img = image*mask/FFcorr
                temp_arr[0] = i
                temp_arr[1] = spot_i
                temp_arr[2] = t
                temp_arr[3] = np.max(img)
                temp_arr[4] = np.where(img== temp_arr[3])[1][0]
                temp_arr[5] = np.where(img == temp_arr[3])[0][0]
                temp_arr[6] = np.sum(mask)
                m = cv2.moments(img)
                temp_arr[7] = m['m00']
                temp_arr[8] = m['m10']
                temp_arr[9] = m['m01']
                temp_arr[10] = m['m11']
                temp_arr[11] = m['m20']
                temp_arr[12] = m['m02']
                temp_arr[13] = m['m21']
                temp_arr[14] = m['m12']
                temp_arr[15] = m['m30']
                temp_arr[16] = m['m03']
                x_mean = m['m10']/m['m00']
                y_mean = m['m01']/m['m00']
                x_var = (m['m20']/m['m00']) - (m['m10']/m['m00'])**2
                y_var = (m['m02']/m['m00']) - (m['m01']/m['m00'])**2
                temp_arr[17] = x_mean
                temp_arr[18] = y_mean
                temp_arr[19] = x_var
                temp_arr[20] = y_var
                spots_image = get_array_piece(img, center = (int(x_mean),int(y_mean)), radius = 15)
                f_destination['spots'][spot_i] = temp_arr
                f_destination['spots_images'][spot_i] = spots_image
                spot_i+=1

def generate_video_from_hdf5file(filename):
    """
    Generates a movie clip from a collection of images.

    image modification:
        - sqrt (just take a square root of images)
        - log10 (take log base 10 of images)

    Parameters
    ----------
    filename :: (string)

    Returns
    -------

    Examples
    --------
    >>> from ubcs_auxiliary.multiprocessing import new_process
    >>> new_process(generate_video_from_hdf5file,'/mnt/data/dmout_breathing_1.data.hdf5')
    """
    from numpy import sqrt, log10, zeros
    import cv2
    import h5py
    import os
    from time import time, ctime
    from ubcs_auxiliary.save_load_object import load_from_file

    root, tail = os.path.split(filename)
    basename, extension = tail.split('.')[0], '.'.join(tail.split('.')[1:])
    fdata = h5py.File(filename,'r')

    width = fdata['image width'][()]
    height = fdata['image height'][()]
    size = (width,height)
    fps = 1/0.032
    stats = load_from_file(os.path.join(root,basename+'.stats.pkl'))

    img_mean = stats['S1']/stats['N']
    video_filename = os.path.join(root,basename+'.mp4')
    print(f'the video filename is: {video_filename}')
    video_out = cv2.VideoWriter(video_filename,cv2.VideoWriter_fourcc(*'mp4v'), fps, size)
    for i in range(fdata['images'].shape[0]):
        # writing to a image array
        img = zeros((height,width,3))
        img[:,:,1] = fdata['images'][i] - img_mean
        if i-1>=0:
            img[:,:,0] = fdata['images'][i-1]- img_mean
        if i+1<fdata['images'].shape[0]:
            img[:,:,2] = fdata['images'][i+1]- img_mean
        img = img.astype('uint8')
        video_out.write(img)
    video_out.release()

def generate_video_from_list_of_hdf5files(root,lst,stats_finame):
        """
        Generates a movie clip from a collection of hdf5 files. The input is a list of hdf5 filenames in correct order

        image modification:
            - sqrt (just take a square root of images)
            - log10 (take log base 10 of images)

        Parameters
        ----------
        filename :: (string)

        Returns
        -------

        Examples
        --------
        >>> from ubcs_auxiliary.multiprocessing import new_process
        >>> new_process(generate_video_from_hdf5file,['/mnt/data/dmout_breathing_1.data.hdf5'])
        """
        from numpy import sqrt, log10, zeros, where
        import cv2
        import h5py
        import os
        from time import time, ctime
        from ubcs_auxiliary.save_load_object import load_from_file
        roottemp, tail = os.path.split(lst[0])
        basename, extension = tail.split('.')[0], '.'.join(tail.split('.')[1:])
        vbasename = '_'.join(basename.split('_')[:-1])
        fdata = h5py.File(os.path.join(root,tail),'r')
        width = fdata['image width'][()]
        height = fdata['image height'][()]
        size = (width,height)
        fps = 1/0.032
        stats = load_from_file(stats_finame)
        img_mean = stats['S1']/stats['N']
        video_filename = os.path.join(root,vbasename+'.mp4')
        print(f'the video filename is: {video_filename}')
        video_out = cv2.VideoWriter(video_filename,cv2.VideoWriter_fourcc(*'mp4v'), fps, size)
        for item in lst:
            print(f'{ctime(time())} processing file {item}')
            fdata = h5py.File(os.path.join(root,item),'r')
            for i in range(fdata['images'].shape[0]):
                # writing to a image array
                img = zeros((height,width,3))
                img[:,:,1] = fdata['images'][i]-img_mean

                if i-1>=0:
                    img[:,:,0] = fdata['images'][i-1]-img_mean
                if i+1<fdata['images'].shape[0]:
                    img[:,:,2] = fdata['images'][i+1]-img_mean
                idx = where(img < 0)
                img[idx] = 0
                img = img.astype('uint8')
                video_out.write(img)
        video_out.release()

def generate_video_from_list_of_stats(root,lst,key = 'dmax_frame'):
        """
        Generates a movie clip from a collection of stats files.

        image modification:
            - sqrt (just take a square root of images)
            - log10 (take log base 10 of images)

        Parameters
        ----------
        root :: (string)
        lst :: (list of filenames)
        key :: (string)

        Returns
        -------

        Examples
        --------
        >>> from ubcs_auxiliary.multiprocessing import new_process
        >>> new_process(generate_video_from_list_of_stats,'/mnt/data/dmout_breathing_1.stats.pkl')
        """
        from numpy import sqrt, log10, zeros, where
        import cv2
        import h5py
        import os
        from time import time, ctime
        from ubcs_auxiliary.save_load_object import load_from_file
        roottemp, tail = os.path.split(lst[0])
        basename, extension = tail.split('.')[0], '.'.join(tail.split('.')[1:])
        vbasename = '_'.join(basename.split('_')[:-1])
        fstats = load_from_file(os.path.join(root,tail))
        width = fstats[key].shape[1]
        height = fstats[key].shape[0]
        size = (width,height)
        fps = 24
        video_filename = os.path.join(root,vbasename+f'.{key}.mp4')
        print(f'the video filename is: {video_filename}')
        video_out = cv2.VideoWriter(video_filename,cv2.VideoWriter_fourcc(*'mp4v'), fps, size)
        for item in lst:
            print(f'{ctime(time())} processing file {item}')
            fstats = load_from_file(os.path.join(root,item))
            for i in range(int(fstats[key].max()+1)):
                # writing to a image array
                img = zeros((height,width,3), dtype = 'uint8')
                img[:,:,1] = (fstats[key] == i)
                if i-1>=0:
                    img[:,:,0] = (fstats[key] == i-1)
                if i+1 < 600:
                    img[:,:,2] = (fstats[key] == i+1)
                img = img*254
                video_out.write(img)
        video_out.release()

def check_dataset(root, has = '', extension = ''):
    """
    procedure to check if the data set is intact.

    test1. plot difference between all timestamps_cam.
    """
    import os
    from h5py import File
    from numpy import asarray, concatenate
    filenames = get_list_of_files(root,suffix = extension, has = has)
    timestamps_camera = []
    timestamps_lab = []
    for filename in filenames:
        f = File(root+filename,'r')
        timestamps_camera.append(f['timestamps_camera'][()])
        timestamps_lab.append(f['timestamps_lab'][()])
    timestamps_camera_result = concatenate(asarray(timestamps_camera))
    timestamps_lab_result = concatenate(asarray(timestamps_lab))
    return timestamps_camera_result, timestamps_lab_result

def change_hdf5(root,namelist):
    """
    a template used to change the hdf5 file.
    the change is done in 3 steps.
    - rename original file by adding "temp" at the beginning -> temp{}
    - creating a new file with changed dtype or whatever else needs to be changed
    - copy timestamps from origin file to new file
    - delete old file "temp{}""
    """
    from h5py import File
    import os
    from time import ctime, time
    for name in namelist:
        print(f'{ctime(time())}: chaning {name}')
        os.rename(root+name,root+'temp'+name)
        with File(root+name, 'a') as fnew:
            with File(root+'temp'+name,'r') as f:
                for key in list(f.keys()):
                    data = f[key]
                    if key == 'images':
                        height = f['image height'][()]
                        width = f['image width'][()]
                        fnew.create_dataset(key,data = data, chunks=(1,height/8,width/8), dtype='uint16', compression='gzip')
                    else:
                        fnew.create_dataset(key, data = data)
                timestamp = f['timestamps_lab'][0]
        os.utime(root+name,(timestamp, timestamp))
        os.remove(root+'temp'+name)

def generate_mask_from_stats(root,filename):
    """
    Generates a binary mask from a .stats. file

    image modification:
        - sqrt (just take a square root of images)
        - log10 (take log base 10 of images)

    Parameters
    ----------
    root :: (string)
    filename :: (string)

    Returns
    -------

    Examples
    --------
    >>> from ubcs_auxiliary.multiprocessing import new_child_process
    >>> new_child_process(generate_mask_from_stats,'/mnt/data/', dmout_breathing_1.stats.pkl')
    """
    from ubcs_auxiliary.save_load_object import load_from_file
    from h5py import File
    from numpy import zeros, where
    from time import time, ctime
    print(f"{ctime(time())} Start creating {root+filename.split('.')[0]+'.mask.hdf5'} file")
    stats = load_from_file(root+filename)
    fdata = File(root+filename.split('.')[0]+'.data.hdf5', 'r')
    mask = zeros(fdata['images'].shape, dtype = 'bool')
    zhit_dic = Zhit(stats)
    hit_mask = zhit_dic[4]*stats['dmax_frame']
    for i in range(600):
        mask[i] = (hit_mask == i)
    print(f'{ctime(time())} writing .mask.hdf5')
    with File(root+filename.split('.')[0]+'.mask.hdf5', 'a') as fmask:
        fmask.create_dataset('masks', data = mask, dtype = 'bool', compression = 'gzip', compression_opts = 9)
    print(f'{ctime(time())} Done')


def generate_emask_from_mask(source,destination,filename, mask_exclude = None):
    """
    Generates an enumerated mask(.emask.) from a binary mask (.mask.) file

    Parameters
    ----------
    root :: (string)
    filename :: (string)

    Returns
    -------

    Examples
    --------
    >>> from ubcs_auxiliary.multiprocessing import new_child_process
    >>> new_child_process(generate_emask_from_mask,'/mnt/data/', dmout_breathing_1.mask.hdf5')
    """
    from ubcs_auxiliary.save_load_object import load_from_file
    from h5py import File
    from numpy import zeros, where, ones,unique, ones_like
    from time import time, ctime
    from scipy.ndimage import label
    from ubcs_auxiliary.numerical import grow_mask
    fdata = File(source['data']+filename.split('.')[0]+'.data.hdf5', 'r')
    fmask = File(source['hits']+filename.split('.')[0]+'.hits.hdf5', 'r')
    if mask_exclude is None:
        mask_exclude = ones_like(f['hits2'][()])

    width = fdata['image width'][()]
    height = fdata['image height'][()]
    length = fdata['images'].shape[0]
    with File(destination+filename.split('.')[0]+'.emask.hdf5', 'a') as femask:
        femask.create_dataset('emasks', (length,height,width), chunks = (1,height,width), dtype = 'int32', compression = 'gzip', compression_opts = 9)
        femask.create_dataset('spots', (length,),dtype = 'int32')

    with File(destination+filename.split('.')[0]+'.emask.hdf5', 'a') as femask:
        N_spot_max = 0
        i = 0
        mask = fmask['hits3'][i]*mask_exclude
        emask = label(mask,ones((3,3),bool))[0]
        idx = where(emask != 0)
        emask[idx] += N_spot_max
        N_spot_max = emask.max()
        femask['emasks'][i] = emask
        femask['spots'][i] = emask.max()
        for i in range(1,length,1):
            prev_mask = fmask['hits3'][i-1]
            prev_emask = femask['emasks'][i-1]
            mask = fmask['hits3'][i]*mask_exclude

            #create new enumarated mask. start enumeration from the last value in the previous emask.
            emask = label(mask,ones((3,3),bool))[0]
            idx = where(emask != 0)
            emask[idx] += N_spot_max
            N_spot_max = emask.max()

            #take care of overlapping contiguious spots by assigning them to a number from previous frame.

            # temp_mask = grow_mask(prev_mask,1)*mask
            # temp_label = label(temp_mask,ones((3,3),bool))[0]
            #
            # # Compute overlaps with the previous emask.
            #
            # #Grow emasks by M(default 1) and get indices of overlapping seciton. Multiplication of masks returns only pixels that are non-zero in both masks.
            # prev_emask_g =grow_mask(prev_emask,1)
            # emask_g = grow_mask(emask,1)
            # indices = where(prev_emask_g*emask_g > 0)
            #
            # #obtain unique values and indices
            # values_u, idx_u = unique(emask_g[indices], return_index = True)
            #
            # #construct a set of unique indices
            # indices_u = (indices[0][idx_u],indices[1][idx_u])
            #
            # #replace values in emask with values in prev_emask
            # for i in range(values_u.shape[0]):
            #     r = indices_u[0][i]
            #     c = indices_u[1][i]
            #     idxs = where(emask_g == prev_emask_g[r,c])
            #     emask[idxs] = prev_emask_g[r,c]

            #save emask to hdf5file
            femask['emasks'][i] = emask
            femask['spots'][i] = emask.max()
        femask.create_dataset('Nmax', data = N_spot_max)


def copy_emaks_to_particle(root, filename):
    """
    """
    from h5py import File
    femask = File(root+filename.split('.')[0]+'.emask.hdf5', 'r')
    fdata = File(root+filename.split('.')[0]+'.data.hdf5', 'r')
    width = fdata['image width'][()]
    height = fdata['image height'][()]
    length = fdata['images'].shape[0]
    with File(root+filename.split('.')[0]+'.particle.hdf5', 'a') as fparticle:
        fparticle.create_dataset('particles', (length,height,width), data  = femask['emasks'], chunks = (1,height,width), dtype = 'int32', compression = 'gzip', compression_opts = 9)

def generate_table_from_emask(root, filename):
    """
    generates the catalog of all unique spots in the frames(from .emask. file) and saves into .table.

    Simple header: [frame, particle, r, c, size]

    ----------
    root :: (string)
    filename :: (string)

    Returns
    -------

    Examples
    --------
    >>> from ubcs_auxiliary.multiprocessing import new_process
    >>> new_process(gererate_table_from_emask,'/mnt/data/', dmout_breathing_1.emask.hdf5')
    """
    from h5py import File
    import numpy as np
    from cv2 import moments
    from time import time, ctime, sleep
    table_filename = root+filename.split('.')[0]+'.table.hdf5'
    emask_filename = root+filename.split('.')[0]+'.emask.hdf5'
    femask = File(emask_filename, 'r')
    Nmax = femask['Nmax'][()]
    spots = femask['spots'][()]
    header =  ['frame', 'spot', 'track', 'row_center', 'col_center','size','lambdap','lambdam','theta','positive']

    entry = np.zeros((6,))
    with File(table_filename, 'a') as ftable:
        table = ftable.create_group("table")
        for item in header:
            table.create_dataset(item, (Nmax,))

        i = 0
        spot= 0
        for frame in range(spots.shape[0]):
            emask = femask['emasks'][frame]
            print(f'{ctime(time())}, frame {frame}')
            if emask.any() > 0:
                minvalue = np.min(emask[emask != 0])
                maxvalue = np.max(emask[emask != 0])
                for spot in range(minvalue,maxvalue+1,1):
                    mask = (emask == spot)
                    mom = get_binary_moments(mask)

                    table['frame'][i] = frame
                    table['spot'][i] = spot
                    table['track'][i]  = 0
                    table['row_center'][i] = m['row_center']
                    table['col_center'][i] = m['col_center']
                    table['size'][i] =  m['size']
                    table['lambdap'][i] = m['lambdap']
                    table['lambdam'][i] = m['lambdam']
                    table['size'][i] = m['theta']
                    table['positive'][i] = 1
                    i+=1

def find_overlapping_spots_in_table(filename):
    """
    """
    from h5py import File


def Zhit(stats):
    """From stats, find pixels whose intensity exceeds Z_threshold, which is defined to target 1 false positive per image. Returns Z, the ratio of dmax1/sigma, the coordinates of each 'hit', the frame number in which it was found, and its count value."""
    from numpy import argsort,where, array

    # Extract relevant stats
    N = stats['N']
    dmin = stats['dmin']
    dmax0 = stats['dmax0']
    dmax1 = stats['dmax1']
    dmax_frame = stats['dmax_frame']
    row,col = dmin.shape

    # Calculate Z and Z_threshold
    sigma = 0.5*(dmax0-dmin+2)/z_statistic(N)
    Z = dmax1/sigma
    Z_threshold = z_statistic(dmax1.size)
    hit = (Z > Z_threshold)

    # Find coordinates, frame number, and counts for each hit
    coord = where(hit)
    frame = dmax_frame[coord]
    counts = dmax1[coord]

    # Sort results according to frame number

    return Z,coord,frame,counts,hit

def z_statistic(N):
    """Returns z, which corresponds to the threshold in units of sigma where
    the probability of measuring a value greater than z*sigma above the mean
    is 1 out of N."""
    from numpy import sqrt
    from scipy.special import erfinv
    return sqrt(2)*erfinv(1-2/N)

def show_usage():
    """
    a simple function that shows the usage of the functions. It is more like
    """
    print('Move the data from .raw.hdf5 to .data.hdf5')
    print("root = '/net/*****-data2/C/covid19Data/Data/2020.06.19/'")
    print("----------------------------------------")

    print('Generate a list of files to analyze')
    print(" lst = procedures.get_list_of_files(root,'.data.hdf5','vocalization')  ")
    print("----------------------------------------")

    print('Generate a video from a list of hdf5 files')
    print("procedures.generate_video_from_list_of_hdf5files(root,lst)")
    print("----------------------------------------")

    print('Generatea a video from a list of stats files')
    print("procedures.generate_video_from_list_of_hdf5files(root,lst)")
    print("----------------------------------------")

    print("To run .stats. procedures")
    print('open new shell. cd /net/*****/C/SAXS-WAXS\ Analysis/software/')
    print('run ipython3')
    print('import covid_procedures_VS')
    print("root = '/net/*****-data2/C/covid19Data/Data/2020.06.19/'")
    print("covid_procedures_VS.process_data(root,select=['.data.hdf5'])")
    print("----------------------------------------")

    print("To run a process as a separate child process")
    print("from ubcs_auxiliary.multiprocessing import new_child_process")
    print("new_child_process(function,arg1,arg2, kwarg1 = 'keyword value', kwarg2 = 1234)")
    print("----------------------------------------")

def create_one_big_hdf5(root,prefix):
    """
    a procedure design to fix a problem with skipped images.
    """
    from h5py import File
    #step1 create massive file
    #
def get_complete_sorted_list(root, term, extension):
    """
    returns complete sorted list from a directory. the procedure designed to fix problems with skipped images.
    """

    import os
    from numpy import argsort, array
    #files = [os.path.join(root,file) for file in os.listdir(root)]
    files = os.listdir(root)
    selected = []
    for file in files:
        if (term in file) and (extension in file):
            selected.append(file)
    repeat = []
    for file in selected:
        repeat.append(int(file.split('_')[-1].split('.')[0]))
    repeat_arr = array(repeat)
    selected_arr = array(selected)
    idx_sorted = argsort(repeat_arr)
    selected_sorted = selected_arr[idx_sorted]

    return selected_sorted

def copy_file_skipped_images(source, destination,lst):
    """
    """
    from time import time, ctime
    from h5py import File
    import os
    from numpy import arange, array
    import numpy as np
    files_src = [os.path.join(source,file) for file in lst]
    files_dst = [os.path.join(destination,file) for file in lst]

    keys_float = ['black level all', 'black level analog', 'black level digital', 'exposure time', 'gain', 'image height', 'image width', 'temperature', 'time']
    keys_arr = ['timestamps_camera', 'timestamps_lab']

    fsrc_0 = File(files_src[0],'r')
    start_ID = fsrc_0['frameIDs'][0]

    frameIDs = array([])
    chunks = array([])
    arr_zeros = np.zeros((256,))
    for i in range(len(files_src)):
        fsrc = File(files_src[i],'r')
        frameIDs = np.concatenate((frameIDs,fsrc['frameIDs']))
        chunks = np.concatenate((chunks,arr_zeros+i))
    fsrcs = []
    for i in range(len(files_dst)):
        fsrcs.append(File(files_src[i],'r'))
    for i in range(len(files_dst)):
        fsrc = File(files_src[i],'r')
        print(ctime(time()),i)
        new_frameIDs = arange(0+i*(256),256*(i+1),1)+start_ID
        curr_frameIDs = fsrc['frameIDs']
        with File(files_dst[i],'w') as fdst:
            for key in keys_float:
                fdst.create_dataset(key,data = fsrc[key][()])
            for key in keys_arr:
                fdst.create_dataset(key,fsrc[key].shape)
            for key in ['images']:
                fdst.create_dataset(key,fsrc[key].shape, dtype = 'int16')
            for key in ['frameIDs']:
                fdst.create_dataset(key,data = new_frameIDs)
            for frameID in list(fdst['frameIDs']):
                if frameID != 0.0:
                    idx = np.where(frameIDs == frameID)
                    chunk = int(chunks[idx][0])
                    idx_src = np.where(fsrcs[chunk]['frameIDs'][()] == frameID)
                    image = fsrcs[chunk]['images'][idx_src]
                    timestamp_camera = fsrcs[chunk]['timestamps_camera'][idx_src]
                    timestamp_lab = fsrcs[chunk]['timestamps_lab'][idx_src]


                    idx_dst = np.where(fdst['frameIDs'][()] == frameID)
                    fdst['images'][idx_dst] = image
                    fdst['timestamps_camera'][idx_dst] = timestamp_camera
                    fdst['timestamps_lab'][idx_dst] = timestamp_lab

def update_hdf5_neighbours_dataset(filename, filename_destination = None, N = 5):
    """
    updates the input hdf5 file specified by 'filename' with new keys related to nearest neighbours. if filename_destination is left None, the original file will be updated.

    example: update_hdf5_neighbours_dataset('/Data/2020.08.10/dm4_singing-1.stats2.hdf5')
    """
    def get_neighbours_dataset(filename, N = 5):
        """
        N - number of neighbours to return
        """
        from lcp_video import analysis
        from numpy import empty
        import numpy as np
        from h5py import File
        from time import ctime,sleep, time
        f_stats = File(filename,'r')
        row = f_stats['row0'][()]
        col = f_stats['col0'][()]
        rfn = f_stats['rfn'][()]
        peak = f_stats['peak'][()].astype('bool')
        f_stats.close()

        lst = []
        for i in range(1,N+1):
            lst.append(f'row{i}')
            lst.append(f'col{i}')
            lst.append(f'rfn{i}')

        result_dic = {}
        for item in lst:
            result_dic[item] = np.empty(peak.shape,dtype='int32')

        # print(f"------------")
        # print(f"{ctime(time())}, processing frame {i}")
        for frame in range(rfn[1],rfn[-1]):
            slctr_i = (rfn == frame)&peak
            if frame%1000 == 0:
                print(ctime(time()),frame,f'out of total = {rfn[-1]}')
            slctr_pmi = ((rfn == frame-1)|(rfn == frame)|(rfn == frame+1))&peak
            if slctr_i.sum()>0:
                row2 = np.copy(row[slctr_pmi]).astype('float64')
                col2 = np.copy(col[slctr_pmi]).astype('float64')
                rfn2 = np.copy(rfn[slctr_pmi]).astype('float64')

                row_i, row_j = np.meshgrid(row2, row2, sparse=True)
                row_m = ((row_i-row_j)**2)
                col_i, col_j = np.meshgrid(col2, col2, sparse=True)
                col_m = ((col_i-col_j)**2)
                matrix = np.sqrt(row_m + col_m)
                matrix_argsort = np.argsort(matrix,axis=0)
                #matrix_sort = np.sort(matrix,axis=1)
                sorted_idx = matrix_argsort[:N+2,:]
                dic = {}
                idx = (rfn2 == frame)

                for k in range(1,N+1):
                    try:
                        dic[f'row{k}'] = row2[sorted_idx[k]].astype('int16')
                        dic[f'col{k}'] = col2[sorted_idx[k]].astype('int16')
                        dic[f'rfn{k}'] = rfn2[sorted_idx[k]].astype('int16')
                    except:
                        pass
                for key in list(dic.keys()):
                    result_dic[key][slctr_i] = dic[key][idx]
        return result_dic

    with File(filename,'r') as fstats:
        row = fstats['row0'][()]
        col = fstats['col0'][()]
        rfn = fstats['rfn'][()]

    result_dic = {}
    for i in range(1,N+1):
        result_dic['col{i}'] = np.empty(col.shape,dtype=col.dtype)
        result_dic['row1{i}'] = np.empty(row.shape,dtype=row.dtype)
        result_dic['rfn{i}'] = np.empty(rfn.shape,dtype='int32')


    result_dic = get_neighbours_dataset(filename)

    if filename_destination is None:
        filename_destination = filename
    with File(filename_destination,'a') as fdest:
        for key in result_dic.keys():
            if key in fdest:
                print(f"{key} dataset exists. rewriting")
                fdest[key] = result_dic[key]
            else:
                print(f"{key} dataset doesn't exists. creating and writing")
                fdest.create_dataset(key, data = result_dic[key])


def stats_from_chunk(reference,sigma=6):
    """Returns mean, var, and threshold (in counts) for reference. The mean
    and var are calculated after omitting the largest and smallest values
    found for each pixel, which assumes few particles in the laser
    sheet. The threshold statistic corresponds to the specified sigma level.
    Adding 0.5 to var helps compensate for digitization error so that false
    positives in the light sheet approximately match that outside the light
    sheet. The threshold for pixels defined by 'mask' are reset to 4095, which
    ensures they won't contribute to hits."""
    from lcp_video.procedures.analysis_functions import poisson_array,dm16_mask,images_from_file,save_to_file
    from numpy import sqrt,ceil,cast,array,zeros_like
    from time import time
    from os import path
    t0=time()
    if '.data.' in reference: stats_name = reference.replace('.data.hdf5','.stats.pkl')
    if '.raw.'  in reference: stats_name = reference.replace('.raw.hdf5','.stats.pkl')
    if  path.exists(stats_name):
        stats = load_from_file(stats_name)
        median = stats['median']
        mean = stats['mean']
        var = stats['var']
        threshold = stats['threshold']
    else:
        print('Processing {} ... please wait'.format(reference))
        # Load images and sort in place to minmimze memory footprint
        images = images_from_file(reference)
        images.sort(axis=0)
        mask = zeros_like(images[0])
        if 'dm16' in reference: mask = dm16_mask()
        # Compute median, then mean and var after omitting smallest and largest values.
        N = len(images)
        M = int(N/2)
        median = images[M]
        mean = images[1:-1].mean(axis=0,dtype='float32')
        var = images[1:-1].var(axis=0,dtype='float32')
        # Compute std_ratio; used to rescale stdev.
        std_ratio = []
        for i in range(10000):
            dist = poisson_array(3,N,sort=True)
            std_ratio.append(dist.std()/dist[1:-1].std())
        std_ratio_mean = array(std_ratio).mean()
        # Compute threshold to nearest larger integer; recast as int16.
        threshold = ceil(mean + sigma*std_ratio_mean*sqrt(var+0.5))
        threshold = cast['int16'](threshold) - median
        threshold[mask] = 4095
        save_to_file(stats_name,{'median':median,'mean':mean,'var':var,'threshold':threshold})
    print('time to execute stats_from_chunk [s]: {:0.1f}'.format(time()-t0))
    return median,mean,var,threshold

def find_pathnames(root,terms):
    """Searches 'root' and finds files that contains 'terms', which is a list
    of strings, i.e., ['dm16','.data.hdf5']. Returns list os sorted according
    to file timestamps."""
    from numpy import argsort,array
    from os import path,listdir
    from os.path import getmtime
    files = [path.join(root,file) for file in listdir(root)]
    select = terms
    selected = []
    [selected.append(file) for file in files if all([term in file for term in select])]
    path_names = selected.copy()
    if len(path_names) > 0:
        creation_times = [getmtime(file) for file in path_names]
        sort_order = argsort(creation_times)
        path_names = array(path_names)[sort_order]
    return path_names


def images_hits_reconstruct(roi_name,frame=-1):
    """Reconstructs images and hits from roi and hits_coord. If frame = -1,
    returns 3D versions; if a non-negative integer, returns image and hits for a
    single frame specified by 'frame'."""
    from numpy import zeros
    #from time import time
    import h5py
    #t0 = time()
    with h5py.File(roi_name,'r') as f:
        shape = f['shape'][()]
        mask = f['mask'][()]
        hits_coord = f['hits_coord'][()]
        roi = f['roi'][()]
    hits = zeros(shape,bool)
    hits[tuple(hits_coord)] = True
    if frame == -1:
        images = zeros(shape,dtype='int16')
        images[:,mask] = roi
        images = images.reshape(shape)
    else:
        hits = hits[frame]
        images = zeros(mask.shape,dtype='int16')
        images[mask] = roi[frame]
    #print('time to reconstruct images,hits [s]: {:0.3f}'.format(time()-t0))
    return images,hits



def inspect_stats_file(filename):
    """
    """
    from h5py import File
    from matplotlib import pyplot as plt
    from ubcs_auxiliary import numerical
    filename = '/net/femto-data2/C/covid19Data/Data/2020.08.13/dm4_ethanol-peg.stats.hdf5'
    filename_ref = '/net/femto-data2/C/covid19Data/Data/2020.08.14/dm4_reference-set.stats.hdf5'

    with File(filename,'r') as f:
        counts= f['counts'][()]
        rfn0 = f['rfn0'][()]
        sat = f['sat'][()]
        chunk = f['chunk'][(0)]
        peak = f['peak'][()]




    with File(filename_ref,'r') as f:
        counts_ref= f['counts'][()]
        rfn0_ref = f['rfn0'][()]
        sat_ref = f['sat'][()]
        chunk_ref = f['chunk'][(0)]
        peak_ref = f['peak'][()]

    plt.figure()
    plt.plot(rfn0[peak],counts[peak],'.', color = 'r')
    plt.plot(rfn0_ref[peak_ref],counts_ref[peak_ref],'.', color = 'b')
