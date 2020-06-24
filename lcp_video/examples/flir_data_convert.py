"""
Convert raw data from FLIR camera.
input data: (N,) 8bit vector

step1:
convert to (N,8) 1bit arrays

Step2:
flatten the (N,8) array to (N*8,)

Step3:
construct (M,12) array from (N*8,)

Step4. Sum axis 1.
"""
import numpy as np
from ubcs_auxiliary.save_load_object import load_from_file
from matplotlib import pyplot as plt
image = load_from_file('./lcp_video/test_data/flir_image.pkl')
rawdata = load_from_file('./lcp_video/test_data/flir_rawdata.pkl')

def get_mask(length):
    from numpy import vstack, tile, hstack, arange
    b0 = 2**hstack((arange(4,12,1),arange(4)))
    b1 = 2**arange(12)
    b = vstack((b0,b1))
    bt = tile(b,(int((length/(2*1.5))),1)).astype('uint16')
    return bt

def raw_to_image(rawdata, height, witdh, mask):
    from numpy import vstack, tile, hstack, arange,reshape
    data_Nx8 = ((rawdata.reshape((-1,1)) & (2**arange(8))) != 0)
    data_N8x1 = data_Nx8.flatten()
    data_Mx12 = data_N8x1.reshape((int(rawdata.shape[0]/1.5),12))
    data = (data_Mx12*mask).sum(axis=1)
    return data.reshape((height,witdh)).astype('uint16')
