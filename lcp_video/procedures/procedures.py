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

def move_files_with_compression(source, destination, extension = '.raw.hdf5', has = '', compression_opt = 4):
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
                        fnew.create_dataset(key, data=data, compression='gzip', chunks=(1,200,256), dtype='int16',compression_opt = compression)
                    else:
                        fnew.create_dataset(key, data = data)
            timestamp = f['timestamps'][0]
        t2 = time()
        print(ctime(time()),f'time: {t2-t1} with size {os.path.getsize(source+filename)/(1024*1024)}, speed {os.path.getsize(source+filename)/((t2-t1)*1024*1024)} MB/s')
        print(f'removing file {source+filename}')
        print(f'changing {destination+prefix}.tmpdata.hdf5')
        os.utime(destination+prefix+'.tmpdata.hdf5',(timestamp, timestamp))
        os.rename(destination+prefix+'.tmpdata.hdf5',destination+prefix+'.data.hdf5')
        os.remove(source+filename)

def move_flat_files_with_compression(source, destination, suffix = '.raw.hdf5', has = '', N = 0, compression = True, reverse = True):
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
    from lcp_video.analysis import mono12p_to_image, mono12packed_to_image, get_mono12packed_conversion_mask

    lst = get_list_of_files(source = source, suffix = suffix, has = has, pixel_format = 'Mono12Packed')
    if reverse:
        lst = lst[::-1]
    def once(lst):
        import os
        from time import time, ctime
        from h5py import File
        from numpy import copy
        if pixel_format == 'Mono12Packed':
            mask = get_mono12packed_conversion_mask(int(width*height*1.5))
        for filename in lst:
            prefix = filename.split('.raw.hdf5')[0]
            print(ctime(time()),f'moving file....')
            print(f'from{source+filename}')
            print(f'to {destination+prefix}.tmpdata.hdf5')
            t1 = time()
            with File(source+filename,'r') as f:
                width = f['image width'][()]
                height = f['image height'][()]
                with File(destination+prefix+'.tmpdata.hdf5','a') as fnew:
                    for key in list(f.keys()):
                        data = f[key][()]
                        if key == 'images':
                            if compression:
                                fnew.create_dataset(key,(data.shape[0],height,width), compression='gzip', chunks=(1,height/8,width/8), dtype='int16')
                            else:
                                fnew.create_dataset(key,(data.shape[0],height,width), chunks=(1,height/8,width/8), dtype='int16')
                            for i in range(data.shape[0]):
                                if pixel_format == 'Mono12Packed':
                                    fnew['images'][i] = mono12packed_to_image(data[i],height,width,mask)
                                else:
                                    fnew['images'][i] = mono12p_to_image(datai,height,width)
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


def generate_emask_from_mask(root,filename):
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
    from numpy import zeros, where, ones,unique
    from time import time, ctime
    from scipy.ndimage import label
    from ubcs_auxiliary.numerical import grow_mask
    fdata = File(root+filename.split('.')[0]+'.data.hdf5', 'r')
    fmask = File(root+filename.split('.')[0]+'.mask.hdf5', 'r')
    width = fdata['image width'][()]
    height = fdata['image height'][()]
    length = fdata['images'].shape[0]
    with File(root+filename.split('.')[0]+'.emask.hdf5', 'a') as femask:
        femask.create_dataset('emasks', (length,height,width), chunks = (1,height,width), dtype = 'int32', compression = 'gzip', compression_opts = 9)
        femask.create_dataset('spots', (length,),dtype = 'int32')

    with File(root+filename.split('.')[0]+'.emask.hdf5', 'a') as femask:
        N_spot_max = 0
        i = 0
        mask = fmask['masks'][i]
        emask = label(mask,ones((3,3),bool))[0]
        idx = where(emask != 0)
        emask[idx] += N_spot_max
        N_spot_max = emask.max()
        femask['emasks'][i] = emask
        femask['spots'][i] = emask.max()
        for i in range(1,length,1):
            prev_mask = fmask['masks'][i-1]
            prev_emask = femask['emasks'][i-1]
            mask = fmask['masks'][i]

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
    from time import time, ctime,sleep
    table_filename = root+filename.split('.')[0]+'.table.hdf5'
    emask_filename = root+filename.split('.')[0]+'.emask.hdf5'
    femask = File(emask_filename, 'r')
    Nmax = femask['Nmax'][()]
    spots = femask['spots'][()]
    header =  [b'frame', b'spot', b'track', b'row_center', b'col_center',b'size']

    entry = np.zeros((6,))
    with File(table_filename, 'a') as ftable:
        table = ftable.create_group("table")
        table.create_dataset('frame', (Nmax,), compression = 'gzip')
        table.create_dataset('spot', (Nmax,), compression = 'gzip')
        table.create_dataset('track', (Nmax,), compression = 'gzip')
        table.create_dataset('row_center', (Nmax,), compression = 'gzip')
        table.create_dataset('col_center', (Nmax,), compression = 'gzip')
        table.create_dataset('size', (Nmax,), compression = 'gzip')

        i = 0
        spot= 0
        for frame in range(spots.shape[0]):
            emask = femask['emasks'][frame]
            if emask.any() > 0:
                minvalue = np.min(emask[emask != 0])
                maxvalue = np.max(emask[emask != 0])
                for spot in range(minvalue,maxvalue+1,1):
                    mask = (emask == spot)
                    size = mask.sum()
                    if size > 1:
                        m = moments(mask.astype('int16'))
                        center_col = m['m10']/m['m00']
                        center_row = m['m01']/m['m00']
                    else:
                        idx = np.where(mask == 1)
                        center_row = idx[0][0]
                        center_col = idx[1][0]

                    table['frame'][i] = frame
                    table['spot'][i] = spot
                    table['track'][i]  = 0
                    table['row_center'][i] = center_row
                    table['col_center'][i] = center_col
                    table['size'][i] = size
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
