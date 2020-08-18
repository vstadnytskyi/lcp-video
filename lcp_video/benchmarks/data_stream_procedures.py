"""
benchmark file to test conversion of raw byte stream to image format
"""


def benchmark_process_threshold_chunk(root,camera,term,N):
    from lcp_video.procedures import data_stream_procedures
    from ubcs_auxiliary.save_load_object import load_from_file
    from time import time
    import timeit
    from numpy import array
    preparation= ''
    preparation += f"root = '{root}'; term = '{term}'; camera = '{camera}'; N = {N};   "
    preparation += "from lcp_video.procedures import data_stream_procedures"
    testcode = 'data_stream_procedures.process_threshold_chunk(root,camera,term,N)'
    t = timeit.Timer(testcode,preparation)
    temp = array(t.repeat(2,2))/2
    print(f"Full function: {round(temp.mean(),3)} +- {round(temp.std(),3)}, with min {round(temp.min(),3)} and max {round(temp.max(),3)}")

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




def benchmark_u12_to_16():
    #mono12packed_to_image(rawdata, height, width, mask)
    from ubcs_auxiliary.save_load_object import load_from_file
    from time import time
    import timeit
    from numpy import array,ones,uint8
    preparation = ""
    testcode = 'data = u12_to_16(rawdata, height=height, width = width)'

    preparation += "from ubcs_auxiliary.save_load_object import load_from_file;\n"
    preparation += 'from numpy import vstack, tile, hstack, arange,reshape, vstack, tile, hstack, arange,reshape, packbits, reshape, int16, concatenate, array, ones, uint8;\n'
    preparation += "from lcp_video.analysis import u12_to_16;\n"
    preparation += f"height = 3000; width = 4096;\n"
    preparation += "rawdata = load_from_file('lcp_video/test_data/mono12p_dataset_12Mpixels.pkl');\n "

    t_full = timeit.Timer(testcode,preparation)

    print("Preparation")
    print(preparation)
    print('tested code')
    print("    arr = rawdata.reshape(-1,3) \n        byte_even = arr[:,0]+256*(bitwise_and(arr[:,1],15)) \n \
        byte_odd = right_shift(bitwise_and(arr[:,1],240),4) + right_shift(256*arr[:,2],4) \n \
        img = empty(height*width,dtype='int16') \n \
        img[0::2] = byte_even\n \
        img[1::2] = byte_odd\n \
        return img.reshape(height,width) \n")
    temp = array(t_full.repeat(4,100))/100
    print(f"Full function: {round(temp.mean(),3)} +- {round(temp.std(),3)}, with min {round(temp.min(),3)} and max {round(temp.max(),3)}")

    from numpy import vstack, tile, hstack, arange,reshape, vstack, tile, hstack, arange,reshape, packbits, reshape, int16, concatenate, array, ones, uint8
    from ubcs_auxiliary.save_load_object import load_from_file;
    height = 2048; width = 2448;
    from lcp_video.analysis import mono12p_to_image;
    rawdata = load_from_file('lcp_video/test_data/flir_mono12p_rawdata.pkl');
    img = load_from_file('lcp_video/test_data/flir_mono12p_image.pkl');
    data = mono12p_to_image(rawdata, height=height, width = width)

    print((data.reshape(height,width) == img).any())

    from matplotlib import pyplot as plt
    plt.figure();plt.imshow(img)
    plt.figure();plt.imshow(data.reshape(height,width))
