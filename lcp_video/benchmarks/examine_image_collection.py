"""
examine performance of image collection for 3 cameras. it is implied that data for all 3 cameras is available.

input: dataset name


'black level all', 'black level analog', 'black level digital', 'exposure time', 'frameIDs', 'gain', 'image height', 'image width', 'images', 'temperature'
, 'time', 'timestamps_camera', 'timestamps_lab'

"""
import sys
import os
from h5py import File
import numpy as np
from matplotlib import pyplot as plt

#one representative filename
dataset_filename = sys.argv[1]

head, tail = os.path.split(dataset_filename)

lst = ['dm4','dm16','dm34']
filename = {}
fhdf5 = {}
timestamps = {}
for item in lst:
    filename[item] = os.path.join(head, '_'.join([item]+tail.split('_')[1:]))
    fhdf5[item] = File(filename[item],'r')
    timestamps[item] = fhdf5[item]['timestamps_camera']

plt.ion()
plt.figure()
for item in lst:
    x = (timestamps[item]-timestamps[item][0])[:-2]
    y = np.diff(timestamps[item])[:-1]*10**-6
    plt.plot(x,y,label = item)
