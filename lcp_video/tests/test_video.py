def test_video_info():
    """
    """
    from lcp_video import video
    dic = video.get_video_info('./lcp_video/test_data/test_video_clip.mov')
    assert dic['height'] == 1080
    assert dic['width'] == 1920
    assert dic['fps'] == 60.0
    assert dic['frame_count'] == 88

def test_get_frame():
    from lcp_video import video
    success, frame = video.get_frame('./lcp_video/test_data/test_video_clip.mov',5)
    assert frame.shape == (1080, 1920, 3)
    assert success == True
    assert frame.mean() == 0.08621238425925926
    assert frame.sum() == 536310

    success, frame  = video.get_frame('./lcp_video/test_data/test_video_clip.mov',90)
    assert success == False
    assert frame == None
