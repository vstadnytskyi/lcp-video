"""
Numerical Analysis and Data Handling library
"""

# Images Analysis Section
def get_moments(image):
    """
    returns moments calculate from an image provided. Uses cv2 (opencv-python) library.

    returns a dictionary of moments:
    'm00', 'm10', 'm01', 'm20', 'm11', 'm02', 'm30', 'm21', 'm12', 'm03', 'mu20', 'mu11', 'mu02', 'mu30', 'mu21', 'mu12', 'mu03
    ', 'nu20', 'nu11', 'nu02', 'nu30', 'nu21', 'nu12', 'nu03'

    Parameters
    ----------
    data :: (numpy array)
        data to append

    Returns
    -------
    moments :: (dictionary)

    Examples
    --------
    >>> moments = get_moments(image = image)
    """
    from cv2 import moments
    return moments(image)

def grow_mask(mask,count=1):
    """Expands area where pixels have value=1 by 'count' pixels in each
    direction, including along the diagonal. If count is 1 or omitted a single
    pixel grows to nine pixels.
    """
    from numpy import array,zeros

    if count < 1: return mask
    if count > 1: mask = grow_mask(mask,count-1)
    w,h = mask.shape
    mask2 = zeros((w,h),mask.dtype)
    mask2 |= mask
    mask2[0:w,0:h-1] |= mask[0:w,1:h] # move up by 1 pixel
    mask2[0:w,1:h] |= mask[0:w,0:h-1] # move down by 1 pixel
    mask2[0:w-1,0:h] |= mask[1:w,0:h] # move to the left by 1 pixel
    mask2[1:w,0:h] |= mask[0:w-1,0:h] # move to the right by 1 pixel

    mask2[0:w-1,0:h-1] |= mask[1:w,1:h] # move left and up by 1 pixel
    mask2[0:w-1,1:h] |= mask[1:w,0:h-1] # move left and down by 1 pixel
    mask2[1:w,0:h-1] |= mask[0:w-1,1:h] # move up and up by 1 pixel
    mask2[1:w,1:h] |= mask[0:w-1,0:h-1] # move up and down by 1 pixel

    return mask2

def get_spots_masks(mask):
    """
    takes boolean mask and
    """
    from skimage import measure
    from numpy import where, zeros_like, bool
    blobs = measure.label(mask==1)
    spots = []
    temp_mask = zeros_like(mask,dtype = bool)
    for blob_number in range(1,blobs.max()+1):
        these_pixels = where(blobs==blob_number)
        if len(these_pixels[0]) < 1000:
            temp_mask[these_pixels] = True
            spots.append(temp_mask)
    return spots

def get_N_of_spots(mask):
    """
    returns number of spots found
    """
    from skimage import measure
    blobs = measure.label(mask==1)
    N = blobs.max()
    return N

def get_histogram(arr,depth = 16):
    """
    assumes unsigned int 16
    """
    from numpy import arange, histogram
    bins = arange(0,(2**depth),1) #calculating histogram
    y,x = histogram(arr,bins = bins)
    return x[:-1],y

#Vector analysis section
def convert_to_rate(x,y, tau = 1):
    """
    takes one dimensional vector of particle count where index in the vector index defines frame number.
    """
    from numpy import where, nan, cumsum, diff
    idx = where(y >0)[0]
    distance= diff(x[idx])
    count = y[idx][:-1]
    rate = count/(distance*tau)
    cm = cumsum(distance)*tau
    return cm,rate, distance

def bin_data(data  = None, x = None, axis = 1, num_of_bins = 300, dtype = 'float'):
        """returns a vector of integers on logarithmic scale starting from decade start, ending decade end with M per decade
        Parameters
        ----------
        data (numpy array)
        x_in (numpy array)
        axis (integer)
        num_of_bins (integer)
        dtype (string)

        Returns
        -------
        dictionary with keys: 'x',y_min','y_max''y_mean'

        Examples
        --------
        >>> from numpy import random, arange
        >>> data = random.rand(4,1000)+ 1
        >>> x_in = arange(0,data.shape[0]+1,1)
        >>> binned_data = bin_data(data  = None, x_in = None, axis = 1, num_of_bins = 300, dtype = 'float')

        .. plot:: ./examples/numerical_bin_data.py
           :include-source:

        """
        from numpy import zeros, nan,arange, nanmax, nanmin, random,nanmean, mean, nansum
        import math

        length = data.shape[0]
        width = data.shape[1]

        if length <= num_of_bins:
            y_max = data
            y_min = data
            y_mean = data
            y_sum = data
            x_out = x
        else:
            y_min = zeros(shape = (width,num_of_bins), dtype = dtype)
            y_max = zeros(shape = (width,num_of_bins), dtype = dtype)
            y_mean = zeros(shape = (width,num_of_bins), dtype = dtype)
            y_sum = zeros(shape = (width,num_of_bins), dtype = dtype)
            x_out = zeros(shape = (num_of_bins,), dtype = dtype)

            for j in range(width):
                idx = 0
                for i in range(num_of_bins):
                    step = int(math.ceil(1.0*(length - idx)/(num_of_bins-i)))

                    start = idx
                    end = idx + step
                    if 'int' in dtype:
                        y_max[j,i] = int(nanmax(data[start:end,j]))
                        y_mean[j,i] = int(nanmean(data[start:end,j]))
                        y_min[j,i] = int(nanmin(data[start:end,j]))
                        y_sum[j,i] = int(nansum(data[start:end,j]))
                    else:
                        y_max[j,i] = nanmax(data[start:end,j])
                        y_mean[j,i] = nanmean(data[start:end,j])
                        y_min[j,i] = nanmin(data[start:end,j])
                        y_sum[j,i] = nansum(data[start:end,j])
                    x_out[i] = mean(x[start:end])
                    idx += step
        dic = {}
        dic['x'] = x_out
        dic['y_min'] = y_min
        dic['y_max'] = y_max
        dic['y_mean'] = y_mean
        dic['y_sum'] = y_sum
        return dic


def distance_matrix(vector):
    """
    takess input vector of point coordinates
    """
    from numpy import zeros
    length = vector.shape[0]
    matrix = zeros((length,length))
    for i in range(length):
        for j in range(length):
            matrix[i,j] = distance(vector[i],vector[j])
    return matrix


def analuse_spot(data, mask):
    """
    returns moments calculate from an image provided. Uses cv2 (opencv-python) library.

    returns a dictionary of moments:
    'm00', 'm10', 'm01', 'm20', 'm11', 'm02', 'm30', 'm21', 'm12', 'm03', 'mu20', 'mu11', 'mu02', 'mu30', 'mu21', 'mu12', 'mu03
    ', 'nu20', 'nu11', 'nu02', 'nu30', 'nu21', 'nu12', 'nu03'

    Parameters
    ----------
    data :: (numpy array)
        data to append

    Returns
    -------
    moments :: (dictionary)

    Examples
    --------
    >>> moments = get_moments(image = image)
    """

def get_test_spots_list():
    """
    returns a list of possible spot IDs from the spots repository.

    Parameters
    ----------s

    Returns
    -------
    list :: (list)

    Examples
    --------
    >>> lst = get_test_spots_list()
    """
    from os.path import exists
    print(exists('./lcp_video/test_data/spots'))

def get_test_spots_image(spot_id = None):
    """
    returns a image from a repository of test images. If spot_id left None, it will return a random image from the repository.

    Parameters
    ----------
    spot_id :: (dictionary)
        dictionary from moments

    Returns
    -------
    eccentricity :: (float)

    Examples
    --------
    >>> spot_properties = get_spot_properties(m)
    """

def get_spot_properties(moments):
    """
    returns properties of the spot from moments dictionary provided by cv2 opencv-python library

    Input keys in the dictionary:
    'm00', 'm10', 'm01', 'm20', 'm11', 'm02', 'm30', 'm21', 'm12', 'm03', 'mu20', 'mu11', 'mu02', 'mu30', 'mu21', 'mu12', 'mu03
    ', 'nu20', 'nu11', 'nu02', 'nu30', 'nu21', 'nu12', 'nu03'

    mu - central moments
    m - raw moments
    nu - scale inveriant Hu moments
    1) M. K. Hu, "Visual Pattern Recognition by Moment Invariants", IRE Trans. Info. Theory, vol. IT-8, pp.179–187, 1962
    2) http://docs.opencv.org/modules/imgproc/doc/structural_analysis_and_shape_descriptors.html?highlight=cvmatchshapes#humoments Hu Moments' OpenCV method

    Parameters
    ----------
    m :: (dictionary)
        dictionary from moments

    Returns
    -------
    eccentricity :: (float)

    Examples
    --------
    >>> spot_properties = get_spot_properties(m)
    """
    from numpy import arctan
    m = moments
    moments['mu00'] = moments['m00']
    #calculate new modified central moments
    m["mu'20"] = moments['mu20']/moments['mu00']
    m["mu'02"] = moments['mu02']/moments['mu00']
    m["mu'11"]= moments['mu11']/moments['mu00']
    lambda_large = 0.5*(m["mu'20"] + m["mu'02"] + (4*m["mu'11"]**2 + (m["mu'20"]-m["mu'02"])**2)**0.5)
    lambda_small = 0.5*(m["mu'20"] + m["mu'02"] - (4*m["mu'11"]**2 + (m["mu'20"]-m["mu'02"])**2)**0.5)

    #calculate eccentricity
    eccentricity = (1-(lambda_large/lambda_small))**0.5
    #calculate theta angle
    theta = 0.5*arctan(2*m["mu'11"]/(m["mu'20"]-m["mu'02"]))

    result = {}
    result['eccentricity'] = eccentricity
    result['theta'] = theta
    result['axis_large'] = lambda_large
    result['axis_small'] = lambda_small

    return result

# HDF File Object Section

class HDF_File():
    def __init__(self,filename):
        import h5py
        from threading import RLock as Lock
        self.lock = Lock()
        self.filename = filename
        with self.lock:
            with h5py.File(self.filename,'a') as f:
                location = 'description'
                try:
                    f[location] = ''
                except:
                    print(f'{location} already exists')

    def append(self,frame,spot,data):
        import h5py
        with self.lock:
            with h5py.File(self.filename,'a') as f:
                location = f'{frame}/{spot}'
                try:
                    f.create_dataset(location, data = data)
                except:
                    print(f'{location} already exists')

    def get_frame_length(self):
        import h5py
        with self.lock:
            with h5py.File(self.filename,'r') as f:
                length = len(f.keys())-1
        return length

    def get_spots(self, threshold):
        import h5py
        from numpy import array
        frames = []
        spots = []
        with self.lock:
            with h5py.File(self.filename,'r') as f:
                keys = f.keys()
                for key in keys:
                    if key != "description":
                        frames.append(int(key))
                        summ = 0
                        for spot in f[key].keys():
                            if f[key][spot][0] >= threshold:
                                summ+=1
                        spots.append(summ)
        return array(frames),array(spots)
