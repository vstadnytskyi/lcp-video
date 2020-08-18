"""
benchmark file to test conversion of raw byte stream to image format
"""

def save_frame_to_hdf5_file(fsrc, key = 'images', compression = 0):
    """
    a benchmark function that would test the write speed from an hdf5 file
    """
    preparation = ""
    preparation += "from h5py import File;"
    preparation += "from tempfile import gettempdir;"
    preparation += "import os;"
    preparation += "root = gettempdir()"
    preparation += "filename_dst = os.path.join(root,'test_destination.hdf5')"
    preparation += "filename_dst = os.path.join(root,'test_destination.hdf5')"
    testcode = ''


def hits_1():
    """
    """
    pass

def benchmark_get_mono12packed_conversion_mask():
    from lcp_video.analysis import get_mono12packed_conversion_mask
    from ubcs_auxiliary.save_load_object import load_from_file
    from time import time
    import timeit
    from numpy import array
    preparation= ''
    preparation += "from lcp_video.analysis import get_mono12packed_conversion_mask;"
    preparation += "length=1,height = 2048; width = 2448;"
    testcode = 'mask = get_mono12packed_conversion_mask(int(width*height*length*1.5))'
    t = timeit.Timer(testcode,preparation)
    temp = array(t.repeat(4,20))/20
    print(f"{round(temp.mean(),3)} +- {round(temp.std(),3)}")

def benchmark_mono12packed_to_image():
    #mono12packed_to_image(rawdata, height, width, mask)
    from lcp_video.analysis import get_mono12packed_conversion_mask
    from ubcs_auxiliary.save_load_object import load_from_file
    from time import time
    import timeit
    from numpy import array
    preparation = ""
    testcode = 'data = mono12packed_to_image(rawdata,height,width,mask)'

    preparation += "from ubcs_auxiliary.save_load_object import load_from_file;"
    preparation += "from lcp_video.analysis import get_mono12packed_conversion_mask;"
    preparation += "from lcp_video.analysis import mono12packed_to_image;"
    preparation += "rawdata = load_from_file('lcp_video/test_data/flir_rawdata_mono12packed.pkl');"
    preparation += "length=1,height = 2048; width = 2448;"
    preparation += "mask = get_mono12packed_conversion_mask(int(width*height*1.5));"
    preparation += 'from numpy import vstack, tile, hstack, arange,reshape,vstack, tile, hstack, arange,reshape;'

    print('tested code')
    print("l1: data_Nx8 = ((rawdata.reshape((-1,1)) & (2**arange(8))) != 0)")
    print("l2: data_N8x1 = data_Nx8.flatten()")
    print("l3: data_Mx12 = data_N8x1.reshape((int(rawdata.shape[0]/1.5),12))")
    print("l4: data = (data_Mx12*mask).sum(axis=1)")
    print("l5: data.reshape((length,height,width)).astype('int16')")

    t = timeit.Timer(testcode,preparation)
    temp = array(t.repeat(4,20))/20
    print(f"full code: {round(temp.mean(),3)} +- {round(temp.std(),3)}")



    testcode = "data_Nx8 = ((rawdata.reshape((-1,1)) & (2**arange(8))) != 0);"
    t = timeit.Timer(testcode,preparation)
    temp = array(t.repeat(4,20))/20
    print(f"line1: {round(temp.mean(),3)} +- {round(temp.std(),3)}")
    preparation +=testcode

    testcode = "data_N8x1 = data_Nx8.flatten();"
    t = timeit.Timer(testcode,preparation)
    temp = array(t.repeat(4,20))/20
    print(f"line2: {round(temp.mean(),3)} +- {round(temp.std(),3)}")
    preparation +=testcode

    testcode = "data_Mx12 = data_N8x1.reshape((int(rawdata.shape[0]/1.5),12)).astype('int16');"
    t = timeit.Timer(testcode,preparation)
    temp = array(t.repeat(4,20))/20
    print(f"line3: {round(temp.mean(),3)} +- {round(temp.std(),3)}")
    preparation +=testcode


    testcode = "data = (data_Mx12*mask).sum(axis=1);"
    t = timeit.Timer(testcode,preparation)
    temp = array(t.repeat(4,20))/20
    print(f"line4: {round(temp.mean(),3)} +- {round(temp.std(),3)}")
    preparation +=testcode

    testcode = "data.reshape((height,width)).astype('int16');"
    t = timeit.Timer(testcode,preparation)
    temp = array(t.repeat(4,20))/20
    print(f"line5: {round(temp.mean(),3)} +- {round(temp.std(),3)}")
    preparation +=testcode

def benchmark_mono12p_to_image(N = 2):
    #mono12packed_to_image(rawdata, height, width, mask)
    from ubcs_auxiliary.save_load_object import load_from_file
    from time import time
    import timeit
    from numpy import array
    preparation = ""
    testcode = 'data = mono12p_to_image(rawdata,length,height,width,mask,mask8)'

    preparation += "from ubcs_auxiliary.save_load_object import load_from_file;"
    preparation += 'from numpy import vstack, tile, hstack, arange,reshape, vstack, tile, hstack, arange,reshape, packbits, reshape, int16, concatenate, array;'
    preparation += "from lcp_video.analysis import mono12p_to_image;"
    preparation += "from lcp_video.analysis import get_mono12p_conversion_mask,get_mono12p_conversion_mask_8bit;"
    preparation += f"length = {int(N)};height = 2048; width = 2448;"
    preparation += "rawdata = load_from_file('lcp_video/test_data/flir_rawdata_mono12p.pkl'); rawdata = array([rawdata]*length);"
    preparation += "mask = get_mono12p_conversion_mask(int(width*height*length*1.5));"
    preparation += "mask8 = get_mono12p_conversion_mask_8bit(int(width*height*length*1.5));"



    t_full = timeit.Timer(testcode,preparation)

    print('tested code')
    print("1: data_Nx8 = ((rawdata.reshape((-1,1)) & (mask8)) != 0)")
    print("2: data_N8x1 = data_Nx8.flatten()")
    print("3: data_Mx12 = data_N8x1.reshape((int(rawdata.shape[-1]*length/1.5),12)).astype('int16')")
    print("4: data = (data_Mx12 * mask)).T.sum(axis=0)")
    print("5: data.reshape((length,height,width)).astype('int16')')")
    temp = array(t_full.repeat(4,20))/20
    print(f"Full function: {round(temp.mean(),3)} +- {round(temp.std(),3)}")

    testcode = 'data_Nx8 = ((rawdata.reshape((-1,1)) & (mask8)) != 0);'
    t = timeit.Timer(testcode,preparation)
    temp = array(t.repeat(4,20))/20
    print(f"line1: {round(temp.mean(),3)} +- {round(temp.std(),3)} with min of {round(temp.min(),3)} ")

    preparation +=testcode
    testcode = 'data_N8x1 = data_Nx8.flatten();'
    t = timeit.Timer(testcode,preparation)
    temp = array(t.repeat(4,20))/20
    print(f"line2: {round(temp.mean(),3)} +- {round(temp.std(),3)} with min of {round(temp.min(),3)} ")

    preparation +=testcode
    testcode = "data_Mx12 = data_N8x1.reshape((int(rawdata.shape[-1]*length/1.5),12));"
    t = timeit.Timer(testcode,preparation)
    temp = array(t.repeat(4,20))/20
    print(f"line3: {round(temp.mean(),3)} +- {round(temp.std(),3)} with min of {round(temp.min(),3)} ")


    preparation +=testcode
    testcode = "data = (data_Mx12*mask).T.sum(axis=0);"
    t = timeit.Timer(testcode,preparation)
    temp = array(t.repeat(4,20))/20
    print(f"line4: {round(temp.mean(),3)} +- {round(temp.std(),3)} with min of {round(temp.min(),3)} ")

    preparation +=testcode
    testcode = "data.reshape((length,height,width)).astype('int16');"
    t = timeit.Timer(testcode,preparation)
    temp = array(t.repeat(4,20))/20
    print(f"line5: {round(temp.mean(),3)} +- {round(temp.std(),3)} with min of {round(temp.min(),3)} ")

def profile_mono12p_to_image():
    from ubcs_auxiliary.save_load_object import load_from_file
    from lcp_video.analysis import mono12p_to_image
    mono12p = load_from_file('lcp_video/test_data/flir_rawdata_mono12p.pkl')
    height = 2048
    width = 2448
    data = mono12p_to_image(mono12p,height,width)
