def mask_testdata():
    """
    simple procedure that generates a test data set for emask procedure.

    The data set generates several hdf5 files with spots spaced by certain distance.
    """

    from numpy import zeros

    data = zeros((100,1000,1000))

    length = data.shape[0]
    start = [(10,10),(5,5)]
    direction = [(0,5),(0,2)]
    size= [(2.5,2.5),(0.5,0.5)]

    for particle in range(len(start)):
        for frame in range(length):
            r = start[particle][0] + direction[particle][0]*frame
            c = start[particle][1] + direction[particle][1]*frame
            rw = size[particle][0]
            cw = size[particle][1]
            mask = ellipse_mask(rw,cw,r,c,0,data[0])
            if mask.size > 1:
                data[frame] += mask

    return data






def ellipse_mask(a,b,x0,y0,theta,image):
    """Returns elliptical mask of radii (a,b) with rotation theta (in degrees)
    and offset (xo,yo) from the geometric center of 'I' (X0,Y0), which must
    be passed via **kws and must have an odd number of pixels in each
    dimension. The mask has the same shape as 'I' and is assigned 1 (0) for
    pixels outside (inside) the ellipse, and assigned fractional areas for
    pixels intercepted by the ellipse.

    To install astropy and regions, run the following commands from terminal:
        conda install astropy
        pip install regions
    """
    from regions import EllipsePixelRegion,PixCoord
    from astropy.units import deg
    from numpy import array
    I = image
    height,width = I.shape

    # Calculate ellipse center (h,k)
    h = x0
    k = y0
    ellipse = EllipsePixelRegion(PixCoord(h,k), 2*a, 2*b, -theta*deg)
    mask = ellipse.to_mask(mode='exact')
    mask = mask.to_image(I.shape)
    return array(mask,dtype = 'bool')

def generate_test_mask():
    root = '/Users/femto-13/Mirror/Data/2020.07.01/'
    filename = 'test.mask.hdf5'
    with File(root+filename,'a') as f:
        f.create_dataset('masks',(100,1000,1000), compression = 'gzip', data = mask_testdata().astype('bool'))
