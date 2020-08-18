"""
benchmark file to test conversion of raw byte stream to image format
"""

def maxima():
    from lcp_video.analysis import maxima, images_hits_reconstruct, offset_image
    from ubcs_auxiliary.save_load_object import load_from_file
    from time import time
    import timeit
    from numpy import array
    preparation= ''
    preparation += "from lcp_video.analysis import maxima_from_image, images_hits_reconstruct, offset_image;"
    preparation += "from ubcs_auxiliary.save_load_object import load_from_file;"
    preparation += "hits = load_from_file('lcp_video/test_data/2020.07.23/dm4_singing-1_4_hits148.pkl');"
    preparation += "image = load_from_file('lcp_video/test_data/2020.07.23/dm4_singing-1_4_frame148.pkl');"
    preparation += "stats = load_from_file('lcp_video/test_data/2020.07.23/dm4_singing-1_4_stats.pkl');"
    preparation += "offset = offset_image((3000,4096))"
    testcode = 'cmax, z = maxima_from_image(image,hits,stats,offset,3)'
    t = timeit.Timer(testcode,preparation)
    temp = array(t.repeat(4,20))/20
    print(f"{round(temp.mean(),3)} +- {round(temp.std(),3)}")
    print(f"max {round(temp.max(),3)} and min {round(temp.min(),3)}")

def nearest_neibhour(length):
    """
    a benchmark function to test analysis.nearest_neibhour(rows, cols, frames, N, frameN)
    """
    from lcp_video.analysis import nearest_neibhour
    from ubcs_auxiliary.save_load_object import load_from_file
    from time import time
    import timeit
    from numpy import array, random
    preparation = ''
    preparation += "from lcp_video.analysis import nearest_neibhour\n"
    preparation += "from numpy import array, random\n"
    preparation += f"rows = random.randint(0,4096,({length},), dtype = 'uint16')\n"
    preparation += f"cols = random.randint(0,4096,({length},), dtype = 'uint16')\n"
    preparation += f"frames = random.randint(0,2,({length},), dtype = 'uint16')\n"
    preparation += "frameN = 1\n"
    preparation += "N = 2\n"
    testcode = ""
    testcode += "dic = nearest_neibhour(rows, cols, frames, N, frameN);"
    print('------------ Preparation ------------');
    print(preparation)
    print('------------ Test Code ------------');
    print(testcode)
    t = timeit.Timer(testcode,preparation)
    temp = array(t.repeat(4,20))/20
    print(f"{round(temp.mean(),3)} +- {round(temp.std(),3)}")
    print(f"max {round(temp.max(),3)} and min {round(temp.min(),3)}")
