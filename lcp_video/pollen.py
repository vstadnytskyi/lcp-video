
from numba import jit

from h5py import File
import numpy as np
from matplotlib import pyplot as plt
#%matplotlib widget
#%matplotlib inline
from importlib import reload
from lcp_video import roi_files
reload(roi_files)
import ubcs_auxiliary as ua
from lcp_video import raw_files, img_files
from time import time,ctime
from skimage import measure
from numpy import where, zeros_like, bool
import os
from threading import RLock as Lock
from pdb import pm
lock = Lock()
def get_spots_masks(mask):
    """
    takes boolean mask and
    """
    from skimage import measure
    from numpy import where, zeros_like, bool
    blobs = measure.label(mask==1)
    spots = []
    for blob_number in range(1,blobs.max()+1):
        temp_mask = zeros_like(mask,dtype = bool)
        these_pixels = where(blobs==blob_number)
        if len(these_pixels[0]) < 5000:
            temp_mask[these_pixels] = True
            spots.append(temp_mask)
    return spots

def analyze_pollen(filename):
    import sys
    from time import time, ctime
    from h5py import File
    mask_camera = np.zeros((1550, 4096))
    mask_camera[:,750:3500] = 1



    head, tail = os.path.split(filename)
    datasetname = tail.split('.')[0]
    list_chunks = img_files.list_dataset(filename)


    f = File(filename,'r')
    bckg = f['images'][0]
    particle_number = []
    frameIDs = []
    t1 = time()
    for chunk in list_chunks:
        with File(chunk,'r') as f:
            for i in range(256):
                img = (f['images'][i] - bckg)*mask_camera
                mask = (img > 30)
                frameID = f['frameIDs'][i]
                spots = get_spots_masks(mask)
                #print('number of spots',len(spots),'at',ctime(time()))
                spots_real_lst = []
                number = 0
                for spot in spots:
                    size = spot.sum()
                    intensity = (img[spot>0]).sum()
                    string = f'{frameID},{number},{size},{intensity}, \n'
                    with lock:
                         with open(os.path.join(head,datasetname[:-2]+".spots"), "a") as myfile:
                             myfile.write(string)
                    number += 1
                bckg = np.copy(f['images'][i])
                index = list_chunks.index(chunk)
                number = ((index*256+i+1)/(len(list_chunks)*256))
                text = f"\r processing status: {round(number*100,4)} % done"
                text += f'{chunk} '
                t2 = time()
                text += f'{ctime(((t2-t1)/number) + t1)} '
                sys.stdout.write(text)
                sys.stdout.flush()
