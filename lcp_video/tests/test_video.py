def test_video_info():
    """
    """
    from lcp_video import video
    filename = video.get_test_video_filename()
    dic = video.get_video_info(filename)
    assert dic['height'] == 1080
    assert dic['width'] == 1920
    assert dic['fps'] == 60.0
    assert dic['frame_count'] == 88

def test_get_frame():
    from lcp_video import video
    filename = video.get_test_video_filename()
    frame = video.get_frame(filename,5)

    assert frame.shape == (1080, 1920, 3)
    assert success == True
    assert frame.mean() == 0.08621238425925926
    assert frame.sum() == 536310

    frame  = video.get_test_video_filename(filename,90)
    assert frame == None
