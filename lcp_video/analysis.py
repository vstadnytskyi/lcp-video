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
        if len(these_pixels[0]) < ((2**16)-1):
            temp_mask[these_pixels] = True
            spots.append(temp_mask)
    return spots


def enumerate_mask(mask, value = 1):
    """
    Takes a boolean mask, enumerates spots and returns enumerated mask where the intensity of a pixel indicates the spot number.
    """
    from skimage import measure
    emask = measure.label(mask==value)
    return emask

def get_N_of_spots(mask):
    """
    returns number of spots found
    """
    from skimage import measure
    blobs = measure.label(mask==1)
    N = blobs.max()
    return N

def get_histogram_int(arr,depth = 16):
    """
    assumes unsigned int 16
    """
    from numpy import arange, histogram
    bins = arange(0,(2**depth),1) #calculating histogram
    y,x = histogram(arr,bins = bins)
    return x[:-1],y

def get_histogram(arr,length = 16,step = 1):
    """
    assumes unsigned int 16
    """
    from numpy import arange, histogram
    bins = arange(0,length,step) #calculating histogram
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

def smoothing_NMR_simple(image, r_size = 20, c_size = 50):
    """
    The simple smoothing algorithm that takes into account the background around a single peak. It doesn't work that well with a large spot.
    """
    rows = image.shape[0]
    cols = image.shape[1]
    I = image
    I_prime = I*0
    I_sum = I
    for row in range(r_size,rows-r_size):
        for col in range(c_size,cols-c_size):
            I_sum[row,col] = I[row-r_size:row+r_size,col-c_size:col+c_size].sum()/(I[row-r_size:row+r_size,col-c_size:col+c_size].size)
    I_prime = I -  I_sum
    return I_prime

def smoothing_NMR_sophisticated(image):
    """
    """
    sum_I = image[20:-20,50:-50].sum()
    I_prime = I - sum_I/4141
    return I_prime

#Image ANALYSIS

def add_gaussian2D(image, amplitude = 3000, position = (100,100), sigma = (5,5), dtype = 'int16'):
    """
    return 2D gaussian function in a given 'position' on the provided image. The input image can be made of all zeros.
    """
    import numpy as sqrt,exp,copy
    r_mu = position[0]
    c_mu = position[1]
    r_sigma = sigma[0]
    c_sigma = sigma[1]
    gaussian = copy(image)*0
    for r in range(image.shape[0]):
        for c in range(image.shape[1]):
            gaussian[r,c] = amplitude*exp(-((r-r_mu)**2/(2.0*r_sigma**2))-( (c-c_mu)**2 /( 2.0 * c_sigma**2) ) )

    return gaussian.astype(dtype)+image

def add_noise(image, mean = 3, stdev = 3, dtype = 'int16'):
    """
    adds noise to a given image and re
    """
    from numpy.random import rand
    noise = rand(image.shape[0],image.shape[1])*stdev + mean
    return (image+noise).astype(dtype)

def generate_test_image(self, noise = None, spot_dic = None):
    """
    generate test image for testing moments analysis etc. The function generates an image with noise and spots, maybe one spot.
    """
    from numpy.random import randint

def zinger_free_statistics_from_buffer(buffer,row, col,clip=5,dtype = 'float64'):
    """
    Adapted from SAXS-WAXS Analysis

    *** Original Description ***
    Requires as input 'files', which is a list of image file names, and
    'Dmean', the mean background to be subtracted from the image before
    processing. For a dark dataset, set Dmean = 0.

    Returns zinger- and sink-free M1 (mean), M2 (variance) and M3 (skew)
    images. In addition, returns Imax and Imin images, which correspond to the
    maximum and minimum values found, respectively. If Dmean = 0, it is assumed
    that the data to be processed correspond to a 'dark' dataset, and the
    images will not be scaled according to their integrated intensity before
    performing a moments calculation. If Dmean is a 2D array of background
    offset values, it is subtracted from the image, and the resulting difference
    is scaled according to the integrated number of counts. Then, accumulates
    sum, sum of squared, and sum of cubed pixel intensities and saves the
    clip=M largest and smallest values for each pixel. Subtracts the saved maximum
    and minimum values from the sum, sum of squares, and sum of cubes before
    calculating M1, M2, and M3. Assuming zingers and sinks appear in at most
    clip=M images in the series, the results should be zinger- and sink-free.
    Since all calculations are performed in float64, the memory required for
    9+2*clip 64-bit images (~112 MB each) is nearly 1.5 GB for clip = 2
    (assuming 14 Mpixel images). These data are reduced to a 280 MB dictionary.
    Note that 512 16-bit images (28 MB/image) occupy over 7 GB space."""
    from numpy import sort,zeros,ones,cast,where,float32,float64,seterr,array,isscalar
    if dtype == 'float64':
        dtype=float64
    elif dtype == 'float32':
        dtype=float32
    Isum1 = zeros((row,col))
    Isum2 = zeros((row,col))
    Isum3 = zeros((row,col))
    Imax = zeros((clip,row,col))
    Imin = 255*ones((clip,row,col))
    scale = []
    N_images = buffer.shape[0]
    for i in range(N_images):
        image = cast[dtype](buffer[i])
        image_sum = image.sum()
        scale.append(image_sum)
        Isum1 += image
        Isum2 += image**2
        Isum3 += image**3
        zinger = image > Imax[0]
        Imax[0] = where(zinger,image,Imax[0])
        Imax = sort(Imax,axis=0)
        sink = image < Imin[-1]
        Imin[-1] = where(sink,image,Imin[-1])
        Imin = sort(Imin,axis=0)

    # Compute M1, M2, and M3 afer omitting corresponding Imax and Imin
    Isum1 -=  Imax.sum(axis=0) + Imin.sum(axis=0)
    Isum2 -=  (Imax**2).sum(axis=0) + (Imin**2).sum(axis=0)
    Isum3 -=  (Imax**3).sum(axis=0) + (Imin**3).sum(axis=0)
    scale = array(scale)
    scale_mean = 1
    N = N_images - 2*clip
    M1 = cast[dtype](scale_mean*Isum1/N)
    M2 = cast[dtype](scale_mean**2*Isum2/N - M1**2)
    seterr(divide='ignore',invalid='ignore')
    M3 = cast[dtype]((scale_mean**3*Isum3/N - M1**3 - 3*M1*M2)/M2**1.5)
    seterr(divide='warn',invalid='warn')
    Imax1 = cast[dtype](scale_mean*Imax[-clip:])
    Imin0 = cast[dtype](scale_mean*Imin[:clip])
    zfs_dict = {'N':N,'mean':M1,'variance':M2,'skew':M3,'Imax':Imax1,'Imin':Imin0,'scale':scale}
    return zfs_dict

class ZingerFreeStatistics():
    """
    Adapted from SAXS-WAXS Analysis

    *** Original Description ***
    Requires as input 'files', which is a list of image file names, and
    'Dmean', the mean background to be subtracted from the image before
    processing. For a dark dataset, set Dmean = 0.

    Returns zinger- and sink-free M1 (mean), M2 (variance) and M3 (skew)
    images. In addition, returns Imax and Imin images, which correspond to the
    maximum and minimum values found, respectively. If Dmean = 0, it is assumed
    that the data to be processed correspond to a 'dark' dataset, and the
    images will not be scaled according to their integrated intensity before
    performing a moments calculation. If Dmean is a 2D array of background
    offset values, it is subtracted from the image, and the resulting difference
    is scaled according to the integrated number of counts. Then, accumulates
    sum, sum of squared, and sum of cubed pixel intensities and saves the
    clip=M largest and smallest values for each pixel. Subtracts the saved maximum
    and minimum values from the sum, sum of squares, and sum of cubes before
    calculating M1, M2, and M3. Assuming zingers and sinks appear in at most
    clip=M images in the series, the results should be zinger- and sink-free.
    Since all calculations are performed in float64, the memory required for
    9+2*clip 64-bit images (~112 MB each) is nearly 1.5 GB for clip = 2
    (assuming 14 Mpixel images). These data are reduced to a 280 MB dictionary.
    Note that 512 16-bit images (28 MB/image) occupy over 7 GB space."""
    def __init__(self,row,col,clip, dtype = 'float64'):
        from numpy import sort,zeros,ones,cast,where,float32,float64,seterr,array,isscalar
        if dtype == 'float32':
            self.dtype = float32
        elif dtype == "float64":
            self.dtype = float64
        else:
            self.dtype = float64
        self.N = 0
        self.M1 = None
        self.M2 = None
        self.M3 = None
        self.Imax1 = None
        self.Imin0 = None
        self.scale = None
        self.row = 0
        self.col = 0
        self.clip = 5
        self.Isum1 = zeros((row,col))
        self.Isum2 = zeros((row,col))
        self.Isum3 = zeros((row,col))
        self.Imax = zeros((clip,row,col))
        self.Imin = 255*ones((clip,row,col))

    def append(self, image):
        from numpy import cast
        dtype = self.dtype
        image = cast[dtype](image)
        image_sum = image.sum()
        self.scale.append(image_sum)
        self.Isum1 += image
        self.Isum2 += image**2
        self.Isum3 += image**3
        zinger = image > Imax[0]
        self.Imax[0] = where(zinger,image,Imax[0])
        self.Imax = sort(Imax,axis=0)
        sink = image < Imin[-1]
        self.Imin[-1] = where(sink,image,Imin[-1])
        self.Imin = sort(Imin,axis=0)
        self.N_images += 1

    def compute(self):
        # Compute M1, M2, and M3 afer omitting corresponding Imax and Imin
        Imax = self.Imax
        Imin = self.Imin
        scale = self.scale
        N_images = self.N_images
        clip = self.clip
        dtype = self.dtype

        Isum1 += - Imax.sum(axis=0) - Imin.sum(axis=0)
        Isum2 += - (Imax**2).sum(axis=0) - (Imin**2).sum(axis=0)
        Isum3 += - (Imax**3).sum(axis=0) - (Imin**3).sum(axis=0)
        scale = array(scale)
        scale_mean = 1
        N = N_images - 2*clip
        M1 = cast[dtype](scale_mean*Isum1/N)
        M2 = cast[dtype](scale_mean**2*Isum2/N - M1**2)
        seterr(divide='ignore',invalid='ignore')
        M3 = cast[float32]((scale_mean**3*Isum3/N - M1**3 - 3*M1*M2)/M2**1.5)
        seterr(divide='warn',invalid='warn')
        Imax1 = cast[dtype](scale_mean*Imax[-clip:])
        Imin0 = cast[dtype](scale_mean*Imin[:clip])
        zfs_dict = {'N':N,'mean':M1,'variance':M2,'skew':M3,'Imax':Imax1,'Imin':Imin0,'scale':scale}

        self.N = N
        self.mean = M1
        self.variance = M2
        self.skew = M3
        self.Imax = Imax1
        self.Imin = Imin0
        self.scale_arr = scale
        self.zfs_dict = zfs_dict

class HDF_File():
    """
    HDF_File class to save data
    """
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

    def append(self,frame,data):
        import h5py
        with self.lock:
            with h5py.File(self.filename,'a') as f:
                location = f'{frame}/{spot}'
                try:
                    f.create_dataset(location, data = data, dtype='i16')
                except:
                    print(f'{location} already exists')

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

def zinger_free_statistics_clip(images,clip=2):

    """
    NEW

    Requires as input 'images', which is a 3D numpy array.
    Returns zinger- and sink-free M1 (mean), M2 (variance) and M3 (skew)
    images. In addition, returns Imax and Imin images, which correspond to the
    maximum and minimum values found, respectively. Accumulates
    sum, sum of squared, and sum of cubed pixel intensities and saves the
    clip=M largest and smallest values for each pixel. Subtracts the saved maximum
    and minimum values from the sum, sum of squares, and sum of cubes before
    calculating M1, M2, and M3. Assuming zingers and sinks appear in at most
    clip=M images in the series, the results should be zinger- and sink-free.
    All calculations are performed in float64, but the results are cast as
    32-bit float to reduce the file size when writing files."""
    from numpy import sort,zeros,ones,cast,where,float32,seterr
    N_images,row,col = images.shape
    Isum1 = zeros((row,col))
    Isum2 = zeros((row,col))
    Isum3 = zeros((row,col))
    Imax = zeros((clip,row,col))
    Imin = 65535*ones((clip,row,col))
    scale = zeros(N_images)
    for i in range(0,N_images):
        image = cast[float](images[i]) #25 ms
        scale[i] = image.sum() #20 ms
        Isum1 += image
        Isum2 += image**2 #10ms
        Isum3 += image**3 #40ms
        zinger = image > Imax[0] #10 ms
        Imax[0] = where(zinger,image,Imax[0]) #20 ms
        Imax = sort(Imax,axis=0) #250ms
        sink = image < Imin[-1] #10ms
        Imin[-1] = where(sink,image,Imin[-1]) #20 ms
        Imin = sort(Imin,axis=0) #250ms

    # Compute M1, M2, and M3 afer omitting corresponding Imax and Imin
    Isum1 = Isum1 - Imax.sum(axis=0) - Imin.sum(axis=0)
    Isum2 = Isum2 - (Imax**2).sum(axis=0) - (Imin**2).sum(axis=0)
    Isum3 = Isum3 - (Imax**3).sum(axis=0) - (Imin**3).sum(axis=0)

    N = N_images - 2*clip
    M1 = cast[float32](Isum1/N)
    M2 = cast[float32](Isum2/N - M1**2)
    seterr(divide='ignore',invalid='ignore')
    M3 = cast[float32]((Isum3/N - M1**3 - 3*M1*M2)/M2**1.5)
    seterr(divide='warn',invalid='warn')
    Imax = cast[float32](Imax)
    Imin = cast[float32](Imin)
    zfs_dict = {'N':N,'M1':M1,'M2':M2,'M3':M3,'Imax':Imax,'Imin':Imin,'scale':scale}
    return zfs_dict

def zinger_free_statistics(images,Dmean=None,clip=2):
    """Returns a dictionary of statistics computed from 'images', a 3D array.
    Dmean is the mean value determined from a set of 'dark' images. When
    provided, Dmean is subtracted from the images, the result of which is
    normalized by its sum. When performing zinger-free statistics for a set of
    dark images, leave Dmean set to its default 'None'. This function accumulates
    the sum, sum of squared, and sum of cubed pixel intensities and retains the
    clip=M largest and smallest values for each pixel. Subtracts the saved
    maximum and minimum values from the sum, sum of squares, and sum of cubes
    before calculating M1, M2, and M3. Assuming zingers and sinks appear in at
    most clip=M images in the series, the results should be zinger- and
    sink-free. All calculations are performed in float64. The returned
    dictionary includes zinger- and sink-free M1 (mean), M2 (variance) and M3
    (skew) images. In addition, includes scale and arrays of Imax and Imin
    images, which correspond to the maximum and minimum values found,
    respectively. """
    from numpy import sort,zeros,ones,cast,where,float64,float32,seterr
    N_images,row,col = images.shape
    Isum1 = zeros((row,col))
    Isum2 = zeros((row,col))
    Isum3 = zeros((row,col))
    Imax = zeros((clip,row,col))
    Imin = 65535*ones((clip,row,col))
    scale = zeros(N_images)
    for i in range(N_images):
        if Dmean is None:
            image = cast[float64](images[i])
            scale[i] = image.sum()
        else:
            image = cast[float64](images[i]) - Dmean
            scale[i] = image.sum()
            image /= scale[i]
        Isum1 += image
        Isum2 += image**2
        Isum3 += image**3
        zinger = image > Imax[0]
        Imax[0] = where(zinger,image,Imax[0])
        Imax = sort(Imax,axis=0)
        sink = image < Imin[-1]
        Imin[-1] = where(sink,image,Imin[-1])
        Imin = sort(Imin,axis=0)

    # Compute M1, M2, and M3 afer omitting corresponding Imax and Imin
    Isum1 = Isum1 - Imax.sum(axis=0) - Imin.sum(axis=0)
    Isum2 = Isum2 - (Imax**2).sum(axis=0) - (Imin**2).sum(axis=0)
    Isum3 = Isum3 - (Imax**3).sum(axis=0) - (Imin**3).sum(axis=0)

    if Dmean is None:
        scale_mean = 1
    else:
        scale_mean = scale.mean()
    N = N_images - 2*clip
    M1 = cast[float32](scale_mean*Isum1/N)
    M2 = cast[float32](scale_mean**2*Isum2/N - M1**2)
    seterr(divide='ignore',invalid='ignore')
    M3 = cast[float32]((scale_mean**3*Isum3/N - M1**3 - 3*M1*M2)/M2**1.5)
    seterr(divide='warn',invalid='warn')
    Imax = cast[float32](scale_mean*Imax)
    Imin = cast[float32](scale_mean*Imin)
    zfs_dict = {'N':N,'M1':M1,'M2':M2,'M3':M3,'Imax':Imax,'Imin':Imin,'scale':scale}
    return zfs_dict

def zinger_free_statistics_clip_threaded(images,clip=2, threads = 4):

    """
    Requires as input 'images', which is a 3D numpy array.
    Returns zinger- and sink-free M1 (mean), M2 (variance) and M3 (skew)
    images. In addition, returns Imax and Imin images, which correspond to the
    maximum and minimum values found, respectively. Accumulates
    sum, sum of squared, and sum of cubed pixel intensities and saves the
    clip=M largest and smallest values for each pixel. Subtracts the saved maximum
    and minimum values from the sum, sum of squares, and sum of cubes before
    calculating M1, M2, and M3. Assuming zingers and sinks appear in at most
    clip=M images in the series, the results should be zinger- and sink-free.
    All calculations are performed in float64, but the results are cast as
    32-bit float to reduce the file size when writing files."""
    from numpy import sort,zeros,ones,cast,where,float32,seterr
    from ubcs_auxiliary.threading import new_thread

    dic = {}
    def func(images,rs,re,cs,ce):
        N_images,row,col = images.shape
        Isum1 = zeros((row,col))
        Isum2 = zeros((row,col))
        Isum3 = zeros((row,col))
        Imax = zeros((clip,row,col))
        Imin = 65535*ones((clip,row,col))
        scale = zeros(N_images)
        for i in range(0,N_images):
            image = cast[float](images[i]) #25 ms
            scale[i] = image.sum() #20 ms
            Isum1 += image
            Isum2 += image**2 #10ms
            Isum3 += image**3 #40ms
            zinger = image > Imax[0] #10 ms
            Imax[0] = where(zinger,image,Imax[0]) #20 ms
            Imax = sort(Imax,axis=0) #250ms
            sink = image < Imin[-1] #10ms
            Imin[-1] = where(sink,image,Imin[-1]) #20 ms
            Imin = sort(Imin,axis=0) #250ms
        return Isum1,Isum2,Isum3,Imax,Imin,scale

    N_images,row,col = images.shape
    rows_step = int(row/threads)
    rows_idxs = []
    for i in range(threads+1):
        rows_idxs.append(rows_step*i)



    # Compute M1, M2, and M3 afer omitting corresponding Imax and Imin
    Isum1 = Isum1 - Imax.sum(axis=0) - Imin.sum(axis=0)
    Isum2 = Isum2 - (Imax**2).sum(axis=0) - (Imin**2).sum(axis=0)
    Isum3 = Isum3 - (Imax**3).sum(axis=0) - (Imin**3).sum(axis=0)

    N = N_images - 2*clip
    M1 = cast[float32](Isum1/N)
    M2 = cast[float32](Isum2/N - M1**2)
    seterr(divide='ignore',invalid='ignore')
    M3 = cast[float32]((Isum3/N - M1**3 - 3*M1*M2)/M2**1.5)
    seterr(divide='warn',invalid='warn')
    Imax = cast[float32](Imax)
    Imin = cast[float32](Imin)
    zfs_dict = {'N':N,'M1':M1,'M2':M2,'M3':M3,'Imax':Imax,'Imin':Imin,'scale':scale}
    return zfs_dict

def stats_mode(array,binsize = 0.01):
    """Returns mode for array after binning the data according to binsize."""
    from numpy import cast,int32,rint
    from scipy.stats import mode
    return mode(cast[int32](rint(array[array>0]/binsize)))[0][0]*binsize



def philips_function1(filename, ref_dic):
    """
    root = '/Volumes/C/covid19Data/Data/2020.05.29/'
    filename = 'camera_telecentric_Spray_6um_chunk1of30.gzip.hdf5'
    """
    from numpy import empty_like, rint, sqrt, maximum
    threshold = ref_dic['Imax'] + rint(3*sqrt(maximum(0.73,ref_dic['M2'])))
    path_name = root+filename
    images = h5py.File(path_name,'r')['images']
    particle_mask = empty_like(images, dtype = 'bool')
    hits = zeros((images.shape[0],))
    particle_mask = images > threshold
    spots = particle_mask.sum(axis=1).sum(axis=1)
    return {'mask':particle_mask,'spots':spots}

def find_spots_sigma(filename_source, filename_destination,ref_dic = None):
    """
    """
    from numpy import zeros_like, rint, sqrt, maximum
    from h5py import File
    with File(filename,'r') as f:
        if ref_dic is not None:
            sigma = sqrt(ref_dic['M2'])
            M1 = ref_dic['M1']
        else:
            sigma = 0.73
            M1 = 15.2
        length,width,height = f['images'].shape
        mask = zeros_like(f['images'], dtype = 'int8')
        hits = zeros((length,))
        particle_mask = mask
        for i in range(length):
            for j in range(2,7):
                if j == 2:
                    coeff = 2
                else:
                    coeff = 1
                mask[i] += ((images[i] - M1) > j*sigma)*coeff
        idx = where(mask == 6)
        hits = mask[idx].sum(axis=1).sum(axis=1)
    return {'mask':mask, 'hits':hits}

def get_variance(m00,m10,m01,m20,m02):
    var_x = (m20/m00) - (m10/m00)**2
    var_y = (m02/m00) - (m01/m00)**2
    var = (var_x + var_y)/2
    return var_x, var_y, var

def get_array_piece_2(arr, center = (0,0), radius = 15, dtype = 'uint16'):
    """
    grabs a square box around center with given radius. Note that first entry in center is x coordinate (or cols) and second is y (rows)

    Example: center = (100,100) and radirus = 15.
    return array will contain data with shape (31,31) centered at pixel (100,100).
    """
    from numpy import nan,zeros,array
    x, y = center
    r = radius
    if ((x-r)<0) or ((y-r)<0):
        result = zeros((2*r+1,2*r+1))
        rx = range(x-r,x+r+1)
        ry = range(y-r,y+r+1)
        for ix in rx:
            for iy in ry:
                if ix <0 or iy < 0:
                    result[iy+r-y, ix+r-x] = 0
                else:
                    result[iy+r-y, ix+r-x] = arr[iy,ix]
    elif ((x+r)>arr.shape[1]) or ((y+r)<arr.shape[0]):
        result = zeros((2*r+1,2*r+1))
        rx = range(x-r,x+r+1)
        ry = range(y-r,y+r+1)
        for ix in rx:
            for iy in ry:
                if ix > arr.shape[1] or iy > arr.shape[0]:
                    result[iy+r-y, ix+r-x] = 0
                else:
                    result[iy+r-y, ix+r-x] = arr[iy,ix]
    else:
        result = arr[x-r:x+r+1,y-r:y+r+1]
    return result

def get_array_piece(arr, center = (0,0), radius = 15, dtype = 'uint16'):
    """
    grabs a square box around center with given radius. Note that first entry in center is x coordinate (or cols) and second is y (rows)

    Example: center = (100,100) and radirus = 15.
    return array will contain data with shape (31,31) centered at pixel (100,100).
    """
    from numpy import nan,zeros,array
    x, y = center
    r = radius
    flag1 = ((x+r) < arr.shape[1])
    flag2 = ((y+r) < arr.shape[0])
    flag3 = ((x-r) > 0)
    flag4 = ((y-r) > 0)
    if flag1*flag2*flag3*flag4:
        result = arr[y-r:y+r+1,x-r:x+r+1]
    else:
        result = zeros((2*r+1,2*r+1))
        for idx in range(2*r+1):
            for idy in range(2*r+1):
                isx = x-r+idx
                isy = y-r+idy
                if (isx < arr.shape[1]) and (isy < arr.shape[0]) and (isx >= 0) and (isy >= 0):
                    result[idy,idx] = arr[isy,isx]
                else:
                    result[idy,idx] = 0

    return result

def get_random_array(size = (3000,4096),range = (0,4094), dtype = 'uint16'):
    """
    returns random array
    """
    from numpy.random import randint
    return randint(range[0],range[1],size = size,dtype = dtype)
