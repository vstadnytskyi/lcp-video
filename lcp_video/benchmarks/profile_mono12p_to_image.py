from ubcs_auxiliary.save_load_object import load_from_file
from lcp_video.analysis import mono12p_to_image
mono12p = load_from_file('lcp_video/test_data/flir_rawdata_mono12p.pkl')
height = 2048
width = 2448
data = mono12p_to_image(mono12p,height,width)
