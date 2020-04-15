"""
Video handling library
"""

def get_video_info(filename):
    """
    returns information about video file in a dictionary format


    Parameters
    ----------
    filename :: (string)
        full filename including root

    Returns
    -------
    video_info :: (dictionary)

    Examples
    --------
    >>> video_info = get_video_info(filename)
    """
    import cv2
    dic = {}
    vidcap = cv2.VideoCapture(filename)
    dic['filename'] = filename
    dic['width'] = int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH))
    dic['height'] = int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    dic['gain'] = vidcap.get(cv2.CAP_PROP_GAIN)
    dic['iso'] = vidcap.get(cv2.CAP_PROP_ISO_SPEED)
    dic['frame_count'] = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    dic['format'] = vidcap.get(cv2.CAP_PROP_FORMAT)
    dic['fps'] = round(vidcap.get(cv2.CAP_PROP_FPS),0)
    return dic


def get_frames_index_range(filename, start, end):
    """
    returns a range of frames as numpy array. The indecies comply with scimage

    We can then supplement the above table as follows:

    Addendum to dimension names and orders in scikit-image
    Image type: 2D color video
    coordinates: (time, row, col, channel)
    """
    import cv2
    from numpy import array, zeros, nan
    vidcap = cv2.VideoCapture(filename)
    vidcap.set(cv2.CAP_PROP_POS_FRAMES,start)
    i = 0
    success, image = vidcap.read()
    arr = zeros((end-start,image.shape[0],image.shape[1],image.shape[2]))*nan
    arr[i] = image
    i+=1
    while start+i < end:
        success, image = vidcap.read()
        if success:
            arr[i] = image
        i+=1
    return arr

def frame_by_time(filename, time, mode = 'closest'):
    """
    """
    import cv2
    vidcap = cv2.VideoCapture(filename)
    dic = get_video_info(filename)
    fps=dic['fps']
    frame_seq = dic['frame_count']
    frame_n = int(frame_seq /(time*fps))
    vidcap.set(cv2.CAP_PROP_POS_FRAMES,frame_n)
    success,image = vidcap.read()

    #Read the next frame from the video. If you set frame 749 above then the code will return the last frame.
    ret, frame = cap.read()

def get_frame(filename, frame):
    import cv2
    vidcap = cv2.VideoCapture(filename)
    total_frame_count = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame <= total_frame_count:
        vidcap.set(cv2.CAP_PROP_POS_FRAMES,frame)
        success,image = vidcap.read()
    else:
        success = False
    if success:
        return success, image
    else:
        return False, None

def get_cum_frames_index_range(filename, start, end):
    """
    returns a summed of frames as numpy array. The indecies comply with scimage

    Addendum to dimension names and orders in scikit-image
    Image type: 2D color video
    coordinates: (time, row, col, channel)

    """
    import cv2
    from numpy import array, zeros, cumsum
    import function
    vidcap = cv2.VideoCapture(filename)
    vidcap.set(cv2.CAP_PROP_POS_FRAMES,start)
    length = end-start
    i = 0
    success, image = vidcap.read()
    dic = {}
    shape = image.shape
    dic['sum'] = zeros((shape[0],shape[1],shape[2]))*nan
    dic['mean'] = zeros((shape[0],shape[1],shape[2]))*nan
    dic['std'] = zeros((shape[0],shape[1],shape[2]))*nan
    dic['max'] = zeros((shape[0],shape[1],shape[2]),dtype = image.dtype)
    dic['min'] = zeros((shape[0],shape[1],shape[2]),dtype = image.dtype)

    dic['sum'] += image
    dic['mean'] += image
    dic['std'] += image
    dic['max'] = image
    dic['min'] = image

    cum_image = cumsum()
    while frame_start+i < frame_end:
        success, image = vidcap.read()
        if success:
            arr[i] = image
        i+=1
    return arr
