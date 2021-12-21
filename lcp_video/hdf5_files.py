"""
The library of codes used to work with .raw.hdf5 file generate by scattering experiments

"""
import os
os.system('export HDF5_USE_FILE_LOCKING=FALSE')

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

def list_dataset(filename, type = 'roi'):
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
            if extension == f"{type}.hdf5":
                lst_dataset.append(os.path.join(root,item))
                lst_order.append(int(chunk))
    lst_sorted = np.argsort(np.array(lst_order))
    return list(np.array(lst_dataset)[lst_sorted])

def list_unique_datasets(root, type ='roi'):
    """
    returns a list of unique datasets in a directory

    Parameters
    ----------
    root :: (string)
    type :: (string)

    Returns
    -------
    list :: (list)

    Examples
    --------
    >>> moments = get_moments(image = image)
    """
    import os
    lst_dir = os.listdir(root)
    lst_datasets = []
    for item in lst_dir:
        if f".{type}.hdf5" in item:
            rt,camera_name,dataset_name,chunk,extension = split_raw_filename(item)
            if dataset_name not in lst_datasets:
                lst_datasets.append(dataset_name)
    return lst_datasets
