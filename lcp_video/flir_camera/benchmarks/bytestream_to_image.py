"""
benchmark file to test conversion of raw byte stream to image format
"""


def benchmark_get_mono12packed_conversion_mask():
    from lcp_video.analysis import get_mono12packed_conversion_mask
    from ubcs_auxiliary.save_load_object import load_from_file
    from time import time
    import timeit
    from numpy import array
    preparation= ''
    preparation += "from lcp_video.analysis import get_mono12packed_conversion_mask;"
    preparation += "height = 2048; width = 2448;"
    testcode = 'mask = get_mono12packed_conversion_mask(int(width*height*1.5))'
    t = timeit.Timer(testcode,preparation)
    temp = array(t.repeat(4,20))/20
    print(f"{round(temp.mean(),3)} +- {round(temp.std(),3)}")

def benchmark_mono12packed_to_image():
    #mono12packed_to_image(rawdata, height, witdh, mask)
    from lcp_video.analysis import get_mono12packed_conversion_mask
    from ubcs_auxiliary.save_load_object import load_from_file
    from time import time
    import timeit
    from numpy import array
    preparation = ""
    testcode = 'data = mono12packed_to_image(mono12packed,height,width,mask)'

    preparation += "from ubcs_auxiliary.save_load_object import load_from_file;"
    preparation += "from lcp_video.analysis import get_mono12packed_conversion_mask;"
    preparation += "from lcp_video.analysis import mono12packed_to_image;"
    preparation += "mono12packed = load_from_file('lcp_video/test_data/flir_rawdata_mono12packed.pkl');"
    preparation += "height = 2048; width = 2448;"
    preparation += "mask = get_mono12packed_conversion_mask(int(width*height*1.5))"
    preparation += 'from numpy import vstack, tile, hstack, arange,reshape;'
    t = timeit.Timer(testcode,preparation)

    temp = array(t.repeat(4,20))/20
    print(f"{round(temp.mean(),3)} +- {round(temp.std(),3)}")


def benchmark_mono12p_to_image():
    #mono12packed_to_image(rawdata, height, witdh, mask)
    from lcp_video.analysis import get_mono12packed_conversion_mask
    from ubcs_auxiliary.save_load_object import load_from_file
    from time import time
    import timeit
    from numpy import array
    preparation = ""
    testcode = 'data = mono12p_to_image(mono12p,height,width)'

    preparation += "from ubcs_auxiliary.save_load_object import load_from_file;"
    preparation += "from lcp_video.analysis import mono12p_to_image;"
    preparation += "mono12p = load_from_file('lcp_video/test_data/flir_rawdata_mono12p.pkl');"
    preparation += "height = 2048; width = 2448;"
    preparation += 'from numpy import vstack, tile, hstack, arange,reshape;'
    t = timeit.Timer(testcode,preparation)

    temp = array(t.repeat(4,20))/20
    print(f"{round(temp.mean(),3)} +- {round(temp.std(),3)}")
