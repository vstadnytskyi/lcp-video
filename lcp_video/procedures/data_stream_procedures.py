#step 1:
# Process first chunk and generate mean, std and treshgold.

root = '/mnt/data/'
term = 'background'
camera = 'dm16'
N = 0

def stats_from_data(images):
    from numpy import mean, std
    M1 = mean(images,axis=0)
    M2 = std(images,axis=0)
    threshold = (M2*6+0.5).astype('uint8')

    return {'M1':M1, 'M2':M2, 'threshold':threshold}

def hits_from_data(images):
    from numpy import mean, std, sum
    hits = (images > 30)
    res = {}
    res['hits0'] = sum(hits)
    res['hits1'] = sum(hits, axis = (1,2))
    res['hits2'] = (sum(hits, axis = 0) > 0)
    res['hits3'] = hits
    return res

def roi_from_hits_and_data(hits2, images):
    from ubcs_auxiliary.numerical import grow_mask
    hits2 = grow_mask(hits2,1)
    return images*hits2

def convert_raw_to_images(raw, pixel_format, length, height, width):
    """
    """
    from lcp_video.analysis import mono12p_to_image
    from numpy import zeros, empty

    if pixel_format == 'mono12p_16':
        images = raw[()].reshape((length,height,width))
    elif pixel_format == 'mono12p':
        images = empty((length,height,width))
        for i in range(length):
            images[i] = mono12p_to_image(raw[i],width = width,height = height).reshape((height,width))
    return images

def convert_raw_to_image(raw, pixel_format, height, width):
    """
    """
    from lcp_video.analysis import mono12p_to_image
    from numpy import zeros, empty

    if pixel_format == 'mono12p_16':
        images = raw.reshape((height,width))
    elif pixel_format == 'mono12p':
        images = mono12p_to_image(raw,width = width,height = height).reshape((height,width))
    return image


def get_listdir(root,camera,term,N=None, sort = ''):
    import os
    from h5py import File
    from numpy import argsort, array, zeros
    if N is None:
        include = [camera,term,f'.raw.hdf5']
    else:
        include = [camera,term,f'_{N}.raw.hdf5']
    exclude = []
    files = [os.path.join(root,file) for file in os.listdir(root)]
    selected = []
    [selected.append(file) for file in files if ((all([term in file for term in include])) and (all([term2 not in file for term2 in exclude])))]
    return selected


def find_recent_filename(root, include, exclude = [], newest_first = True):
    """
    find the list of files or folders that have any terms specified in the list 'include' and do not have terms specified in 'exclude'. The extra parameter reverse_order specified whether return the newest or oldest one.

    Parameters
    ----------
    source (string)
    include (list)
    exclude (list)
    sort (string)

    Returns
    -------
    file_list (list)
    """
    from os import listdir,path
    from os.path import getmtime
    from numpy import argsort, array
    files = [path.join(root,file) for file in listdir(root)]
    selected = []
    [selected.append(file) for file in files if ((all([term in file for term in include])) and (all([term2 not in file for term2 in exclude])))]
    path_names = selected.copy()
    if len(path_names) > 0:
        creation_times = [getmtime(file) for file in path_names]
        sort_order = argsort(creation_times)
        if newest_first:
            return array(path_names)[sort_order][-1]
        else:
            return array(path_names)[sort_order][0]
    else:
        return ''

def process_threshold_chunk(filename):
    """

    """

    import os
    from h5py import File
    from numpy import argsort, array, zeros, empty

    f_reference = File(filename,'r')
    raw = f_reference['images'][()]
    pixel_format = f_reference['pixel format'][()]
    width = f_reference['image width'][()]
    height = f_reference['image height'][()]

    images = convert_raw_to_images(raw,pixel_format,length,height,width)
    stats = stats_from_data(images)

    from ubcs_auxiliary.save_load_object import save_to_file
    stats_filename = filename.replace(f'_0.raw.hdf5','.stats.pkl')
    save_to_file(stats_filename,stats)

def process_raw(filename,stats, delete = False):
    """
    """
    from h5py import File
    from numpy import empty
    fraw = File(filename,'r')
    raw = fraw['images'][()]
    length = fraw['images'].shape[0]
    width = fraw['image width'][()]
    height = fraw['image height'][()]
    pixel_format = fraw['pixel format'][()]
    #preparation. Create ROI file and populate it with all fields.

    images = convert_raw_to_images(raw, pixel_format=pixel_format,height=height,width=width,length=length)

    hits_dict = hits_from_chunk(images)
    filename_hits = filename.replace('.raw.hdf5','.hits.hdf5')


    hits_header = ['hits0','hits1','hits2','hits3']
    with File(filename_hits,'w') as fhits:
        fhits.create_dataset('hits0', data = hits_dict['hits0'], dtype = 'int32')
        fhits.create_dataset('hits1', data = hits_dict['hits1'], dtype = 'int32')
        fhits.create_dataset('hits2',  data = hits_dict['hits2'], dtype = 'uint8')
        fhits.create_dataset('hits3', data = hits_dict['hits3'], dtype = 'uint8', compression = 'lzf')

    roi = roi_from_hits_and_data(hits2 = hits_dict['hits2'], images = images)

    filename_roi = filename.replace('.raw.hdf5','.roi.hdf5')
    with h5py.File(filename_roi,'w') as froi:
        for key in fraw.keys():
            if key != 'images':
                froi.create_dataset(key, data = fraw[key])
            elif key == 'images':
                froi.create_dataset(key, data = roi, chunks = (1,height,width), compression = 'lzf', dtype = 'int16')
            froi.create_dataset

def run_once(source, destination, camera, term, repeat = False, newest_first = False):
    import os
    filename_raw = find_recent_filename(source, include = [camera,term,'.raw.'], exclude = [], newest_first = False)
    root, tail = os.path.split(filename_raw)
    head = ''.join(tail.split('_')[:-1])
    filename_hits = os.path.join(destination,tail).replace('.raw.hdf5','.hits.hdf5')
    filename_stats = os.path.join(source,head+'.stats.pkl')
    filename_roi = os.path.join(destination,tail).replace('.raw.hdf5','.roi.hdf5')

    if not os.path.exists(filename_stats):
        filename = os.path.join(source, head+'_0.raw.hdf5')
        process_threshold_chunk(filename)

    if os.path.exists(filename_hits):
        print('hits file exists')
    else:
        process_raw(filename,stats, delete = True)

def run(source, destination, camera, term, repeat = False):
    #find_recent_filename('source,include = ['dm4','.data.'], exclude = [], newest_first = False)
    import os
