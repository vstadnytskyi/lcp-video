def test_mono12p_to_image():
    """
    """
    from lcp_video import analysis
    from ubcs_auxiliary.save_load_object import load_from_file
    height = 2048
    width = 2448
    raw = load_from_file('lcp_video/test_data/flir_mono12p_rawdata.pkl')
    image_original = load_from_file('lcp_video/test_data/flir_mono12p_image.pkl')
    image_reconstruct = analysis.mono12p_to_image(raw, height, width).reshape((2048,2448))

    assert (image_reconstruct == image_original).all()

# def test_mono12packed_to_image():
#     """
#     """
#     from lcp_video import analysis
#     from ubcs_auxiliary.save_load_object import load_from_file
#     height = 2048
#     width = 2448
#     raw = load_from_file('lcp_video/test_data/flir_mono12packed_rawdata.pkl')
#     image_original = load_from_file('lcp_video/test_data/flir_mono12packed_image.pkl')
#     image_reconstruct = analysis.mono12packed_to_image(raw, height, width).reshape((2048,2448))
#
#     assert (image_reconstruct == image_original).all()
