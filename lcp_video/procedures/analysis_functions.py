#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 14:18:58 2019

@author: philipa
"""

from lcp_video.procedures.db_functions import *



def DM(params,TK, chart = 0):
    """This function returns the temperature-dependent concentration of
    Dimer (D) and Monomer (M). Input parameters required
    include temperature (T) in Kelvin, concentration (C) in [M], as well as
    enthalpy (H) in [J/mol] and entropy (S) in in [J/mol-K] for dimer
    dissociation (Hdm, Sdm). If chart = 1, a chart of the temperature-dependent
    dimer and monomer concentration is generated."""
    from matplotlib.pyplot import figure,plot,title,ylabel,xlabel,legend,grid
    from numpy import exp,sqrt
    mb = params['mb']
    MW = params['MW']
    Hdm = params['Hdm']
    Sdm = params['Sdm']
    C = mb/MW
    R =  8.314 # J/(mol. · K)
    Gdm = Hdm - TK*Sdm
    Kdm = exp(-Gdm/(R*TK))
    M = 0.25*Kdm*(-1+sqrt(1+8*C/Kdm))
    D = M**2/Kdm
    TC = TK-273.15
    if chart == 1:
        chart = figure(figsize=(8,8))
        plot(TC,D,label = 'Dimer')
        plot(TC,M,label = 'Monomer')
        plot(TC,2*D+M,label = 'Sum')
        title('C [mM]: %1.2f , Hdm: %i , Sdm: %1.1f' %(C*1000,Hdm,Sdm))
        ylabel('Concentration [M]')
        xlabel('T (Celsius)')
        legend()
        grid()
    return D,M

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

def poisson_distribution(k,rate):
    """Returns a poisson distribution over integer array 'k' with the number of
    events per 'k' interval specified by 'rate'."""
    from numpy import exp
    from scipy.special import factorial
    return exp(-rate)*rate**k/factorial(k)

def gassian_distribution(x,scale=1,mean=10,sigma=2):
    """Returns Gaussian distribution evaluated over 'x' according to 'scale',
    'mean', and 'sigma' parameters."""
    from numpy import exp,pi,sqrt
    return scale*(1/(sigma*sqrt(2*pi)))*exp(-(x-mean)**2/(2*sigma**2))

def mccd_timestamp_statistics(ts):
    """Returns xi, residual, slope, and sigma when fitting the timestamps to
    a straight line. Every value of 'ts' is assigned a corresponding integer
    number of periods, xi. The slope corresponds to the best fit period. The
    sigma is the standard deviation of the fit."""
    from numpy import roll,median,around,nansum,isnan,sqrt

    # Determine ts spacing and assign x accordingly.
    diff = (ts - roll(ts,1))
    period = median(diff)
    yi = ts - ts[0]
    xi = (around(yi/period,0)).astype(int)
    xsum = nansum(xi)
    x2sum = nansum(xi**2)
    ysum = nansum(yi)
    xysum = nansum(xi*yi)
    N = len(xi)-sum(isnan(yi))
    delta = N*x2sum - xsum**2
    yint = (x2sum*ysum - xsum*xysum)/delta
    slope = (N*xysum -xsum*ysum)/delta
    yfit = yint+slope*xi
    residual = yi-yfit
    sigma = sqrt(nansum(residual**2)/(N-2))
    return xi,yfit,slope,sigma

def started_finished_times_from_datetime_mccd_time(db_name,beamtime):
    """Fits 'mccd_time' extracted from datasets to a straight line, determines
    its offset relative to 'datetime', and then writes the offset-corrected
    fitted time in the 'finished_time' column of the datasets as well as the
    corresponding values for 'started_time' (started_time = finished time -
    slope + 0.072), where slope is the period between images, and 0.072
    corresponds to the time for the return stroke of the translation stage."""
    #from matplotlib.pyplot import rc,figure,subplot,plot,errorbar,title,grid,\
    #    xlabel,ylabel,xlim,ylim,xticks,yticks,legend,gca,close,tight_layout

    # The oscillator on the Rayonix detector runs fast by 2.05e-5, so the
    # mccd_time is rescaled accordingly.
    from matplotlib.backends.backend_pdf import PdfPages
    from numpy import array,std,median
    from analysis_functions import skewness_prob_distribution
    from matplotlib.pyplot import figure,plot,yscale,ylabel,xlabel,title,grid,show,close
    from time import time
    t0 = time()
    PDF_file = PdfPages(beamtime+'_timestamp_statistics.pdf')
    datasets = db_table_extract_key(db_name,'DATASETS','dataset','datetime_first')
    for table_name in datasets:
        fig = figure(figsize=(7,7))
        values_ref = db_table_extract_key(db_name,table_name,'mccd_time','mccd_time')
        ts = values_ref/(1+2.05e-5)
        xi,yfit,slope,sigma = mccd_timestamp_statistics(ts)
        datetimes = db_table_extract_key(db_name,table_name,'datetime','mccd_time')
        ts_log = []
        for datetime in datetimes:
            ts_log.append(datetime_to_timestamp(datetime,timezone=None))
        ts_log = array(ts_log)
        offset = median(ts_log-yfit)
        finished_time = yfit+offset
        started_time = finished_time - slope + 0.072

        # Update 'finished_time' and 'started_time' in table_name.
        db_table_update_column(db_name,table_name,'finished_time',finished_time,'mccd_time',values_ref)
        db_table_update_column(db_name,table_name,'started_time',started_time,'mccd_time',values_ref)

        plot(xi, ts_log - finished_time,'bo',ms = 2)
        ylabel('time difference [s]')
        xlabel('N-periods (period: {:0.6f} [s]; sigma: {:0.4f} [s])'.format(slope,std(ts_log-ts)))
        truncated_name = beamtime+'/.../'+'/'.join(array(table_name.split('/'))[[-2,-1]])
        title(truncated_name, fontsize=9)
        grid(True)
        PDF_file.savefig(fig)
    show()
    PDF_file.close()
    print('time to process and write data into database: {} [s]'.format(time()-t0))

def skewness_prob_distribution(N,Z_target=2):
    """Returns normalized skewness corresponding to Z-statistic = 'Z_target',
    given N observations. The normalized skewness is a dimensionless quantity
    in which the third moment of samples from the population is normalized to
    the second moment to the 3/2 power. For normally distributed data, the
    skewness is zero. For (Z-target=2, N=100), a skewness of less than 0.480
    (plus or minus) means the distribution is not skewed with 95% confidence
    (for Z = 2).
    """
    from numpy import nan,arange,sqrt,log,where
    try:
        g1 = arange(0,sqrt(N),0.01)
        N = float(N)
        Y = g1*sqrt((N+1)*(N+3)/(6*(N-2)))
        beta2 = 3*(N**2+27*N-70)*(N+1)*(N+3)/((N-2)*(N+5)*(N+7)*(N+9))
        W = sqrt(-1+sqrt(2*(beta2-1)))
        delta = 1/sqrt(log(W))
        alpha = sqrt(2/(W**2-1))
        Z = delta*log(Y/alpha+sqrt((Y/alpha)**2+1))
        index = where(Z>Z_target)[0][0]
        skewness = g1[index-1]+0.01*(Z_target-Z[index-1])/(Z[index]-Z[index-1])
    except:
        skewness = nan
        return skewness
    return skewness

def image_skew(I):
    """Calculates first moment, second moment, and skewness of non-zero elements
    of I (2D array). Returns cumulative M1, M2, and skew.
    """
    from numpy import sort,cast,where,arange,cumsum,isnan
    I_sub0 = sort(I.flatten())
    I_sub1 = cast[float](I_sub0)#[where(I_sub0 > 0)])
    cum_N = arange(len(I_sub1))+1
    cum_sum1 = cumsum(I_sub1)
    cum_sum2 = cumsum(I_sub1**2)
    cum_sum3 = cumsum(I_sub1**3)
    M1 = cum_sum1/cum_N
    M2 = (cum_sum2/cum_N-M1**2)
    skew = (cum_sum3/cum_N - M1**3 - 3.*M1*M2)/M2**1.5
    skew[where(isnan(skew))] = 0
    return M1,M2,skew

def skew(I):
    """Calculates first moment, second moment, and skewness of non-zero elements
    of I. Returns cumulative M1, M2, and skew.
    """
    from numpy import sort,cast,where,arange,cumsum,isnan,float64
    I = sort(cast[float64](I))
    cum_N = arange(len(I))+1
    cum_sum1 = cumsum(I)
    cum_sum2 = cumsum(I**2)
    cum_sum3 = cumsum(I**3)
    M1 = cum_sum1/cum_N
    M2 = (cum_sum2/cum_N-M1**2)
    skew = (cum_sum3/cum_N - M1**3 - 3.*M1*M2)/M2**1.5
    skew[where(isnan(skew))] = 0
    return M1,M2,skew

def Rayonix_roi_read(image_file,rmin,rmax,cmin,cmax):
    """Returns region of interest (roi) sub image from Rayonix image via
    memory mapping. With NumPy indexing, the origin (image[0,0]) is found at
    the top-left corner and corresponds to the geometric center of that pixel.
    The first dimension of the image (image.shape[0]) corresponds to the number
    of rows and the second dimension (image.shape[1]) corresponds to the number
    of columns. The roi is specified by minimum and maximum indices for rows
    and columns, i.e., (rmin,rmax,cmin,cmax)."""
    from numpy import memmap,uint16
    image_size = 3840
    headersize = 4096
    return memmap(image_file,uint16,'r',headersize,(image_size,image_size),'C')[rmin:rmax,cmin:cmax]

def beamstop_images(image_files,X0=1984,Y0=1965,w=29,h=25):
    """Returns array of sub-images extracted from 'image_files' whose region of
    interest (roi) is centered on (x0,y0) and whose width and height is given
    by (w,h). The dimensions for the sub-image are assumed to be odd to ensure
    its middle pixel is located at its geometric center."""
    from numpy import zeros
    from time import time
    t0 = time()
    cmin = int(X0-(w-1)/2)
    cmax = int(X0+(w+1)/2)
    rmin = int(Y0-(h-1)/2)
    rmax = int(Y0+(h+1)/2)
    N_images = len(image_files)
    images = zeros((N_images,h,w),dtype="uint16")
    for i in range(N_images):
        images[i] = Rayonix_roi_read(image_files[i],rmin,rmax,cmin,cmax)
    print('time to read {} image files: {:0.3f} [s]'.format(len(image_files),time()-t0))
    return images

def images_clip_sum(images,clip=2):
    """Returns sum of images after clipping the smallest and largest values
    in the sorted list (preserves the mean value). If rare zingers are present
    in the images, this simple approach should suppress them."""
    from numpy import sort
    images_sorted = sort(images,axis=0)
    image = images_sorted[clip:-clip].sum(axis=0)
    return image

def N_photons(beamcenter,background,photons):
    """Returns ib, photons transmitted through the beamstop for each image."""
    from numpy import vstack,ones,dot,linalg
    a = vstack([background, ones(len(background))]).T
    slope,counts = dot(linalg.inv(dot(a.T, a)), dot(a.T, beamcenter))
    scale_factor = photons/counts
    return (beamcenter-slope*background)*scale_factor

def beamcenter_footprint():
    from numpy import array
    beamcenter_footprint = [[0,0,1,1,1,0,0],
                            [0,1,1,1,1,1,0],
                            [1,1,1,1,1,1,1],
                            [1,1,1,1,1,1,1],
                            [1,1,1,1,1,1,1],
                            [0,1,1,1,1,1,0],
                            [0,0,1,1,1,0,0]]
    return array(beamcenter_footprint)

def beamcenter_mask():
    """Returns beamcenter mask as an array. Given the PSF and the dimensions of
    the beamstop, the minimum intensity around beamcenter occurs at a radius of
    3 pixels, hence a 7x7 mask."""
    from numpy import array
    return array([[0,0,0,0,0,0,0],
                  [0,0,0,0,0,0,0],
                  [0,0,1,1,1,0,0],
                  [0,0,1,1,1,0,0],
                  [0,0,1,1,1,0,0],
                  [0,0,0,0,0,0,0],
                  [0,0,0,0,0,0,0]])

def background_mask():
    """Returns background mask as an array. Given the PSF and the dimensions of
    the beamstop, the minimum intensity around beamcenter occurs at a radius of
    3 pixels, hence a 7x7 mask."""
    from numpy import array
    return array([[0,0,1,1,1,0,0],
                  [0,1,0,0,0,1,0],
                  [1,0,0,0,0,0,1],
                  [1,0,0,0,0,0,1],
                  [1,0,0,0,0,0,1],
                  [0,1,0,0,0,1,0],
                  [0,0,1,1,1,0,0]])

def beam_moments(image_file,X0,Y0):
    """Returns zeroth moment (M0) and first moments of beam position (M1X,M1Y)
    in units of pixels relative to the center pixel in image_file. """
    from numpy import memmap,uint16,array,arange
    image_size = 3840
    headersize = 4096
    rmin = Y0-3;rmax = Y0+4
    cmin = X0-3;cmax = X0+4
    image = memmap(image_file,uint16,'r',headersize,(image_size,image_size),'C')[rmin:rmax,cmin:cmax]
    N_bkg = background_mask().sum()
    x_mask = array([arange(-3,4)]*7)
    y_mask = x_mask.T
    bkg = (background_mask()*image).sum()/N_bkg
    spot = beamcenter_mask()*(image-bkg)
    M0 = spot.sum()
    M1X = (x_mask*spot).sum()/M0
    M1Y = (y_mask*spot).sum()/M0
    return M0,M1X,M1Y

def beam_position(images):
    """Returns first moment of beam position (X1,Y1) in units of pixels
    relative to the center pixel in images. The mask is assumed to be 7x7."""
    from numpy import array,arange
    height,width = images[0].shape
    N_bkg = background_mask().sum()
    X0 = int((width-1)/2)
    Y0 = int((height-1)/2)
    x_mask = array([arange(-3,4)]*7)
    y_mask = x_mask.T
    X1 = []
    Y1 = []
    for image in images:
        box = image[Y0-3:Y0+4,X0-3:X0+4]
        bkg = (background_mask()*box).sum()/N_bkg
        spot = beamcenter_mask()*(box-bkg)
        M0 = spot.sum()
        X1.append((x_mask*spot).sum()/M0)
        Y1.append((y_mask*spot).sum()/M0)
    X1 = array(X1)
    Y1 = array(Y1)
    return X1,Y1

def beamcenter_background_from_images(images,X0=None,Y0=None):
    """Returns sum of pixels in beamcenter and background masks of images. The
    masks are assumed to be 7x7."""
    from numpy import array
    if X0==None:
        height,width = images[0].shape
        X0 = int((width-1)/2)
        Y0 = int((height-1)/2)
    beamcenter = []
    background = []
    for image in images:
        box = image[Y0-3:Y0+4,X0-3:X0+4]
        beamcenter.append((box*beamcenter_mask()).sum())
        background.append((box*background_mask()).sum())
    return array(beamcenter),array(background)

def ggm(image):
    from numpy import cast,float64
    from scipy.ndimage.filters import gaussian_gradient_magnitude
    return gaussian_gradient_magnitude(cast[float64](image),sigma=1)

def mask_from_footprint(image,footprint):
    """Generates a mask the same size as image with footprint on center."""
    from numpy import zeros_like,where
    mask = zeros_like(image,dtype=bool)
    height,width = mask.shape
    yf,xf = footprint.shape
    xycoordinates = where(footprint)
    mask[xycoordinates[1]+int((height-1)/2-((yf-1)/2)),xycoordinates[0]+int((width-1)/2-((xf-1)/2))]=1
    return mask

def ib_ix_iy(images,background_mask,beamcenter_mask):
    """Determines (ib,ix,iy) and writes values to the sqlite3 database."""
    from numpy import where
    xy_coordinates = where(background_mask)
    xi = xy_coordinates[0]
    yi = xy_coordinates[1]
    I_background = images[:,xi,yi].sum(axis=1)
    xy_coordinates = where(beamcenter_mask)
    xi = xy_coordinates[0]
    yi = xy_coordinates[1]
    I_beamcenter = images[:,xi,yi].sum(axis=1)
    ib = I_beamcenter - I_background*beamcenter_mask.sum()/background_mask.sum()
    return ib

def PSF_for_image(image,PSF_params):
    """Returns a normalized point spread function (PSF) with the same shape as
    PSF using a sum of a Gaussian centered on (X0,Y0) plus a stretched
    exponential on the same center. The width of the Gaussians is defined in
    terms of fwhm in pixels, and the 1/e distance for the stretched exponential
    is r0 in pixels, with beta defining the exponent for the stretched
    exponential."""
    from numpy import indices,exp,log,sqrt
    xo =   PSF_params['x0']
    yo =   PSF_params['y0']
    A0 =   PSF_params['A0']
    fwhm = PSF_params['fwhm']
    A1 =   PSF_params['A1']
    r1 =   PSF_params['r1']
    beta = PSF_params['beta']
    w,h = image.shape
    x_indices,y_indices = indices((w,h))
    fwhm_to_sigma = 1/sqrt(8*log(2))
    X0 = (w-1)/2
    Y0 = (h-1)/2
    r = sqrt((x_indices - (X0 + xo))**2 + (y_indices - (Y0 + yo))**2)
    sigma = fwhm*fwhm_to_sigma
    PSFfit = A0*exp(-(r**2/(2*sigma**2))) + A1*exp(-(r/r1)**beta)
    PSFfit /= PSFfit.sum()
    return PSFfit

def point_spread_function(image):
    """Returns a point spread function (PSF) with the same shape as image using
    a sum of Gaussians centered on the center pixel as a model for the
    intensity distribution. The widths of the Gaussians are defined in terms of
    fwhm in pixels, and the normalized amplitude is defined in terms of
    relative photon counts, with their sum equaling the total number of photons
    in the PSF. The parameters are hard-coded according to a fit of real data,
    which was carried out by 'Rayonix_PSF.py'."""
    from numpy import pi,indices,exp,log,sqrt,zeros_like
    w,h = image.shape
    x0 = (w-1)/2
    y0 = (h-1)/2
    x_indices,y_indices = indices((w,h))
    r = sqrt((x_indices - x0)**2+(y_indices - y0)**2)
    fwhm_to_sigma = 1/sqrt(8*log(2))
    PSF = zeros_like(image)
    fwhm = [1.93994,4.79877,10.00415,18.79587]
    norm = [0.58543,0.14339,0.17428,0.09743]
    for i in range(len(fwhm)):
        sigma = fwhm[i]*fwhm_to_sigma
        amp = norm[i]/(2*pi*sigma**2)
        PSF = PSF + amp*exp(-r**2/(2*sigma**2))
    return PSF

def parse_mccdfiles(logfile):
    """Returns mccd_dict, which contains lists of filenames, fragments of
    filenames, delays, temps, and repeat numbers. If there are no dependent
    variables, then the dictionary contains only filenames."""
    from analysis_functions import find_filenames_path,unique
    from pathlib import Path
    from numpy import array,where,arange,argsort
    from os.path import getmtime

    # Specify path to mccd_files; find files and sort them.
    parent = Path(logfile).parent.as_posix()
    filenames = find_filenames_path(path_name=parent+'/xray_images',pattern='*.mccd')
    creation_times = [getmtime(file) for file in filenames]
    sort_indices = argsort(creation_times)
    filenames = array(filenames)[sort_indices]

    #Create dictionary containing all mccd_files found
    mccd_dict = {'mccd_files':filenames}
    mccd_dict['N'] = len(filenames)

    # Parse filenames; define indices for numeric (repeat) and string fragments
    fragments = []
    result = [fragments.append(Path(filename).stem.split('_')) for filename in filenames]
    fragments = array(fragments)

    numeric = []
    for fragment in fragments[-1]:
        try:
            result = float(fragment)
            numeric.append(True)
        except:
            numeric.append(False)

    r_indices = where(numeric)[0]
    s_indices = list(arange(r_indices[0],len(fragments[-1])))
    result = [s_indices.remove(N) for N in r_indices]

    # Define number of search 'terms' in series
    mccd_dict['terms'] = r_indices[-1]-r_indices[0]

    # Generate unique (plural) and full length (singular) arrays of available search terms
    if len(s_indices) > 0:
        d_index = 0
        t_index = 0
        for s_index in s_indices:
            if fragments[0][s_index][-1] == 's':
                d_index = s_index
            elif fragments[0][s_index][-1] == 'C':
                t_index = s_index
        if d_index > 0:
            mccd_dict['delays'] = unique(fragments[:,d_index])
            mccd_dict['delay'] = fragments[:,d_index]
        if t_index > 0:
            mccd_dict['temps'] = unique(fragments[:,t_index])
            mccd_dict['temp'] = fragments[:,t_index]
        if len(r_indices) > 2:
            mccd_dict['repeats1'] = unique(fragments[:,r_indices[1]])
            mccd_dict['repeat1'] = fragments[:,r_indices[1]]
            mccd_dict['repeats2'] = unique(fragments[:,r_indices[2]])
            mccd_dict['repeat2'] = fragments[:,r_indices[2]]
        else:
            mccd_dict['repeats'] = unique(fragments[:,r_indices[-1]])
            mccd_dict['repeat'] = fragments[:,r_indices[-1]]
    return mccd_dict

def zinger_free_statistics_clip_dark(files,row=3840,col=3840,clip=2):
    """Returns zinger- and sink-free M1 (mean), M2 (variance) and M3 (skew)
    images. In addition, returns Imax and Imin images, which correspond to the
    maximum and minimum values found, respectively. First, accumulates sum,
    sum of squared, and sum of cubed pixel intensities and saves the clip=M
    largest and smallest values for each pixel. Subtracts the saved maximum
    and minimum values from the sum, sum of squares, and sum of cubes before
    calculating M1, M2, and M3. Assuming zingers and sinks appear in at most
    clip=M images in the series, the results should be zinger- and sink-free.
    Since all calculations are performed in float64, the memory required for
    9+2*clip 64-bit images (~112 MB each) is nearly 1.5 GB for and clip = 2
    (assuming 14 Mpixel images). These data are reduced to a 280 MB dictionary.
    Note that 512 16-bit images (28 MB/image) occupy over 7 GB space."""
    from matplotlib.pyplot import imread
    from numpy import sort,zeros,ones,cast,uint16,where,float32,seterr
    Isum1 = zeros((row,col))
    Isum2 = zeros((row,col))
    Isum3 = zeros((row,col))
    Imax = zeros((clip,row,col))
    Imin = 65535*ones((clip,row,col))
    N_images = len(files)
    for file in files:
        image = cast[float](imread(file))
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
    N = N_images - 2*clip
    M1 = cast[float32](Isum1/N)
    M2 = cast[float32](Isum2/N - M1**2)
    M3 = cast[float32]((Isum3/N - M1**3 - 3*M1*M2)/M2**1.5)
    Imax1 = cast[float32](Imax[-1])
    Imin0 = cast[float32](Imin[0])
    seterr(divide='ignore',invalid='ignore')
    seterr(divide='warn',invalid='warn')
    zfs_dict = {'N':N,'M1':M1,'M2':M2,'M3':M3,'Imax':Imax1,'Imin':Imin0}
    return zfs_dict


def zinger_free_statistics_clip(files,Dmean,row=3840,col=3840,clip=2):
    """Requires as input 'files', which is a list of image file names, and
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
    from matplotlib.pyplot import imread
    from numpy import sort,zeros,ones,cast,where,float32,seterr,array,isscalar
    Isum1 = zeros((row,col))
    Isum2 = zeros((row,col))
    Isum3 = zeros((row,col))
    Imax = zeros((clip,row,col))
    Imin = 65535*ones((clip,row,col))
    N_images = len(files)
    scale = []
    for file in files:
        image = cast[float](imread(file)-Dmean)
        image_sum = image.sum()
        scale.append(image_sum)
        # If Dmean is not a scalar, normalize the image by sum of counts
        if not isscalar(Dmean):
            image /= image_sum
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
    scale = array(scale)
    scale_mean = 1
    if not isscalar(Dmean):
        scale_mean = scale.mean()
    N = N_images - 2*clip
    M1 = cast[float32](scale_mean*Isum1/N)
    M2 = cast[float32](scale_mean**2*Isum2/N - M1**2)
    seterr(divide='ignore',invalid='ignore')
    M3 = cast[float32]((scale_mean**3*Isum3/N - M1**3 - 3*M1*M2)/M2**1.5)
    seterr(divide='warn',invalid='warn')
    Imax1 = cast[float32](scale_mean*Imax[-1])
    Imin0 = cast[float32](scale_mean*Imin[0])
    zfs_dict = {'N':N,'M1':M1,'M2':M2,'M3':M3,'Imax':Imax1,'Imin':Imin0,'scale':scale}
    return zfs_dict

def zinger_free_statistics_scaled_clip_old(files,Dmean,clip=2):
    """Returns zinger- and sink-free Imean and Ivar images (corresponding to
    the mean and variance). In addition, returns Imax and Imin images, which
    correspond to the maximum and minimum values found, as well as N_images and
    scale, which correspond to the number of images from which Imean and Ivar
    were calculated, and the integrated intensity for each image, respectively.
    First, subtracts off Dmean and normalizes images according to scale. Then,
    accumulates sum and sum of squares of pixel intensities and saves the
    clip=2 largest and smallest values for each pixel. Subtracts the saved
    maximum and minimum values from the sum and sum of squares before
    calculating Imean and Ivar images. Assuming zingers and sinks appear in at
    most clip=2 images in the series, the results should be zinger- and
    sink-free. Returns results as float32 to reduce storage space on disk."""
    from matplotlib.pyplot import imread
    from numpy import sort,zeros,ones,cast,where,float32,array
    row,col = Dmean.shape
    Isum = zeros((row,col))
    Isum2 = zeros((row,col))
    Imax = zeros((clip,row,col))
    Imin = 65535*ones((clip,row,col))
    N_images = len(files)
    scale = []
    for file in files:
        image = cast[float](imread(file) - Dmean)
        image_sum = image.sum()
        scale.append(image_sum)
        image /= image_sum
        Isum += image
        Isum2 += image**2
        zinger = image > Imax[0]
        Imax[0] = where(zinger,image,Imax[0])
        Imax = sort(Imax,axis=0)
        sink = image < Imin[clip-1]
        Imin[clip-1] = where(sink,image,Imin[clip-1])
        Imin = sort(Imin,axis=0)
    Imax_sum = Imax.sum(axis=0)
    Imax_sum2 = (Imax**2).sum(axis=0)
    Imin_sum = Imin.sum(axis=0)
    Imin_sum2 = (Imin**2).sum(axis=0)

    # Compute Imean and Ivar afer omitting Imax and Imin
    Isum = Isum - Imax_sum - Imin_sum
    Isum2 = Isum2 - Imax_sum2 - Imin_sum2
    scale = array(scale)
    scale_mean = scale.mean()
    N = N_images - 2*clip
    Imean = Isum/N
    Ivar = (Isum2 - N*Imean**2)/(N-1)
    Imean = cast[float32](scale_mean*Imean)
    Ivar = cast[float32](scale_mean**2*Ivar)
    Imax1 = cast[float32](scale_mean*Imax[clip-1])
    Imin0 = cast[float32](scale_mean*Imin[0])
    return Imean,Ivar,Imax1,Imin0,N,scale

def image_saxs_sum(image,qdict,qmin=0.02,qmax=0.2):
    qpix = qdict['qpix']
    qsaxs = (qpix < qmax) & (qpix > qmin)
    return (image*qsaxs).sum()

def xenon_f(q):
    """
    Returns the q-dependent atomic form factor for xenon using information
    gathered by F. Schotte, as described below:

        Table of atomic scattering factors
        source: CCP4 file lib/data/atomsf.lib
        Ref:
        International Tables for X-ray Crystallography vol. 4 (2nd ed, 1965)
        pp. 99-101, Table 2.2B
        International Tables for X-ray Crystallography vol. C (2nd ed, 1999)
        pp. 572-574, Table 6.1.1.4

        This file contains the following information:
        ID     atom identifier
        Z      atomic number
        ne     number of electrons
        c      coefficient for form factor calculation
        a[4]   coefficient for form factor calculation
        b[4]   coefficient for form factor calculation

        Formfactor:
        f = a1*exp(-b1*s*s) + a2*exp(-b2*s*s) + a3*exp(-b3*s*s) + a4*exp(-b4*s*s) + c
        with s = sin(theta)/lambda

        Friedrich Schotte, 20 Mar 2012


        atom Z ne          c         a1         a2         a3         a4         b1         b2         b3         b4
        XE   54 54   3.711800  20.293301  19.029800   8.976700   1.990000   3.928200   0.344000  26.465900  64.265800

    According to Wikipedia,
    http://lampx.tugraz.at/~hadley/ss1/crystaldiffraction/atomicformfactors/formfactors.php
    the atomic form factor for xenon is well approximated up to q = 25/Angstrom
    with these coefficients. Note that the Formfactor 'f' for xenon at q = 0
    equals its number of electrons. Note: s = q/(4*pi)
    """
    from numpy import pi,exp
    s = q/(4*pi)
    ne=54
    c= 3.711800
    a1=20.293301
    a2=19.029800
    a3=8.976700
    a4=1.990000
    b1=3.928200
    b2=0.344000
    b3=26.465900
    b4=64.265800
    return a1*exp(-b1*s*s) + a2*exp(-b2*s*s) + a3*exp(-b3*s*s) + a4*exp(-b4*s*s) + c

def xenon_f2_integral(s=1/1.0332):
    """Returns integral of:

        (s*(a1*exp(-b1*s*s) + a2*exp(-b2*s*s) + a3*exp(-b3*s*s) + a4*exp(-b4*s*s) + c)**2)*ds

    evaluated from 0 to 's' and multiplied by 2*pi to obtain a volume integral.
    Since s = q/(4*pi) = sin(theta)/wavelength, the maximum value of s occurs
    when theta=pi/2, which corresponds to s = 1/wavelength."""
    c= 3.711800
    a1=20.293301
    a2=19.029800
    a3=8.976700
    a4=1.990000
    b1=3.928200
    b2=0.344000
    b3=26.465900
    b4=64.265800
    def integral(s):
        from numpy import pi,exp
        return 2*pi*(1/4)*(
                2*c*c*s*s-
                4*a4*exp(-b4*s*s)*(a1*exp(-b1*s*s)/(b1+b4) + a2*exp(-b2*s*s)/(b2+b4) + a3*exp(-b3*s*s)/(b3+b4) + c/b4) -
                4*a3*exp(-b3*s*s)*(a1*exp(-b1*s*s)/(b1+b3) + a2*exp(-b2*s*s)/(b2+b3) + c/b3) -
                4*a2*exp(-b2*s*s)*(a1*exp(-b1*s*s)/(b1+b2) + c/b2)-
                4*a1*c*exp(-b1*s*s)/b1-
                a1*a1*exp(-2*b1*s*s)/b1 -
                a2*a2*exp(-2*b2*s*s)/b2 -
                a3*a3*exp(-2*b3*s*s)/b3-
                a4*a4*exp(-2*b4*s*s)/b4
                )
    return integral(s)-integral(0)

def Rayonix_damaged_spot_mask():
    from numpy import indices,sqrt
    height,width = 3840,3840
    y_indices,x_indices = indices((height,width))
    x0,y0,r0 = 1865,1848,90
    #x1,y1,r1 = 2075,1811,30
    mask0 = sqrt((y_indices-y0)**2 + (x_indices-x0)**2) < r0
    #mask1 = sqrt((y_indices-y1)**2 + (x_indices-x1)**2) < r1
    #mask = mask0 | mask1
    #mask[1910:1923,1940:2030] = True
    return mask0

def slab_transmittance(theta,t,mu=2.30835223):
    """Returns slab transmittance as a function of theta, thickness (t), and
    inverse 1/e penetration depth (mu). This calculation integrates analytically
    all x-ray photon trajectories defined by x, the penetration depth into the
    slab before being diffracted into angle 2*theta. Dimensions are in mm."""
    from numpy import exp,cos,where
    #return exp(-mu*t/cos(2*theta))*(exp(-mu*t*(1-1/cos(2*theta)))-1)/(mu*t*(1/cos(2*theta)-1))
    ST = (exp(-mu*t)-exp(-mu*t/cos(2*theta)))/(mu*t*(1/cos(2*theta)-1))
    ST[where(theta==0)] = exp(-mu*t)
    return ST

def slab_transmittance_numeric(theta,t,mu=2.30835223):
    """Returns slab transmittance as a function of theta, thickness (t), and
    inverse 1/e penetration depth (mu). This function generates the same result
    as slab_transmittance, but integrates photon trajectories defined by x
    numerically rather than analytically, and provides a test for the step size
    required to accurately recover the slab transmittance. Dimensions are in mm."""
    #mu = 2.30835223 is the reciprocal attenuation depth for fused silica at 12 keV
    from numpy import exp,cos,linspace
    ST = 0
    N = 50
    xvals = linspace(0,t,N)
    for x in xvals:
        ST += exp(-mu*x)*exp(-mu*(t-x)/cos(2*theta))/N
    return ST

def capillary_transmittance(theta,psi,R,t,mu_s=0.2913,mu_c=2.30835223):
    """Returns capillary transmittance assuming its wall thickness, t, is thin
    compared to its radius, R. The parameters mu_s and mu_c correspond to the
    inverse 1/e penetration depth of the capillary contents (zero if empty) and
    fused silica, respectively. Dimensions are in mm."""
    from numpy import exp,sin,cos,sqrt
    a = cos(2*theta)**2+sin(2*theta)**2*sin(psi)**2
    b = -2*R*cos(2*theta)
    ct = -2*R*t-t**2
    root_t = sqrt(b**2-4*a*ct)
    l = -b/a
    dl = (root_t+b)/(2*a)
    ST = slab_transmittance(theta,t,mu_c)
    CT = 0.5*(ST*exp(-mu_c*dl-mu_s*l)+ST*exp(-mu_c*t-mu_s*2*R))
    return CT

def sample_transmittance(theta,psi,R,t,mu_s=0.2913,mu_c=2.30835223):
    """Returns sample transmittance as a function of theta and psi given the
    capillary radius (R [mm]) and wall thickness (t [mm]). The x-ray beam is scattered
    and attenuated by sample contained in the capillary, and is further
    attenuated by passing through both entrance and exit walls of the
    capillary. The default inverse 1/e penetration depths, mu_s and mu_c,
    correspond to values for water and fused silica at 12 keV. This
    function averages photon trajectories defined by x, the penetration depth
    into the capillary prior to diffraction at angle 2*theta. Dimensions are in mm."""
    from numpy import exp,sin,cos,sqrt,linspace
    def STx(x):
        a = cos(2*theta)**2+sin(2*theta)**2*sin(psi)**2
        b = -2*x*cos(2*theta)
        c = x**2-R**2
        ct = x**2-(R+t)**2
        root = sqrt(b**2-4*a*c)
        root_t = sqrt(b**2-4*a*ct)
        l = (-b+root)/(2*a)
        dl = (root_t-root)/(2*a)
        return exp(-mu_c*t)*exp(-mu_s*(R-x+l))*exp(-mu_c*dl)
    ST = 0
    N = 50
    xvals = linspace(-R,R,N)
    for x in xvals:
        ST += STx(x)/N
    return ST

def phosphor(theta,keV=12,t=0.04,fill_factor=0.58):
    """Returns phosphor absorption probability as a function of theta given the
    x-ray energy, phosphor thickness, t, and fill factor. The fill factor
    corresponds to the volume fraction of Gadox in the phosphor film. The
    phosphor responsivity is assumed to be proportional to the x-ray absorption
    probability (1-T). Dimensions are in mm.
    """
    from numpy import exp,cos
    mu = mu_Gadox(keV) # inverse 1/e penetration depth [mm-1]
    alpha = mu*t*fill_factor
    return 1. - exp(-alpha/cos(2*theta))

def mu_Si(keV=12):
    """Returns inverse 1/e penetration depth [mm-1] of Si given x-ray energy
    in keV. The transmission through a 200-um thick slab of Si was
    calculated every 100 eV over an energy range spanning 8-16 keV using:
        http://henke.lbl.gov/optical_constants/filter2.html
    This result was then converted to mu and saved as a tab-delimited text
    file. The returned result is calculated using a univariate spline and
    should be valid over the range 8-16 keV."""
    from numpy import loadtxt
    from scipy.interpolate import UnivariateSpline
    E_mu = loadtxt('mu_Si.txt',dtype=float,delimiter='\t')
    us_mu = UnivariateSpline(E_mu[:,0],E_mu[:,1],s=0)
    return us_mu(1000*keV)

def mu_Gadox(keV=12):
    """Returns inverse 1/e penetration depth [mm-1] of Gadox given x-ray energy
    in keV. The transmission through a 10-um thick slab of Gadox (Gd2O2S) was
    calculated every 100 eV over an energy range spanning 8-16 keV using:
        http://henke.lbl.gov/optical_constants/filter2.html
    This result was then converted to mu and saved as a tab-delimited text
    file. The returned result is calculated using a univariate spline and
    should be valid over the range 8-16 keV."""
    from numpy import loadtxt
    from scipy.interpolate import UnivariateSpline
    E_mu = loadtxt('mu_Gadox.txt',dtype=float,delimiter='\t')
    us_mu = UnivariateSpline(E_mu[:,0],E_mu[:,1],s=0)
    return us_mu(1000*keV)

def mu_FusedSilica(keV=12):
    """Returns inverse 1/e penetration depth [mm-1] of fused silica given the
    x-ray energy in keV. The transmission through a 0.2-mm thick slab of SiO2
    was calculated every 100 eV over an energy range spanning 8-16 keV using:
        http://henke.lbl.gov/optical_constants/filter2.html
    This result was then converted to mu and saved as a tab-delimited text
    file. The returned result is calculated using a univariate spline and
    should be valid over the range 8-16 keV."""
    from numpy import loadtxt
    from scipy.interpolate import UnivariateSpline
    E_mu = loadtxt('mu_FusedSilica.txt',dtype=float,delimiter='\t')
    us_mu = UnivariateSpline(E_mu[:,0],E_mu[:,1],s=0)
    return us_mu(1000*keV)

def mu_Water(keV=12):
    """Returns inverse 1/e penetration depth [mm-1] of water given the
    x-ray energy in keV. The transmission through a 1-mm thick slab of water
    was calculated every 100 eV over an energy range spanning 8-16 keV using:
        http://henke.lbl.gov/optical_constants/filter2.html
    This result was then converted to mu and saved as a tab-delimited text
    file. The returned result is calculated using a univariate spline and
    should be valid over the range 8-16 keV."""
    from numpy import loadtxt
    from scipy.interpolate import UnivariateSpline
    E_mu = loadtxt('mu_Water.txt',dtype=float,delimiter='\t')
    us_mu = UnivariateSpline(E_mu[:,0],E_mu[:,1],s=0)
    return us_mu(1000*keV)

def mu_Xe(keV=12):
    """Returns inverse 1/e penetration depth [mm-1 atm-1] of Xe given the
    x-ray energy in keV. The transmission through a 3-mm thick slab of Xe at
    6.17 atm (76 psi) was calculated every 100 eV over an energy range
    spanning 5-17 keV using:
        http://henke.lbl.gov/optical_constants/filter2.html
    This result was then converted to mu and saved as a tab-delimited text
    file. The returned result is calculated using a univariate spline and
    should be valid over the range 5-17 keV."""
    from numpy import loadtxt
    from scipy.interpolate import UnivariateSpline
    E_mu = loadtxt('mu_Xe.txt',dtype=float,delimiter='\t')
    us_mu = UnivariateSpline(E_mu[:,0],E_mu[:,1],s=0)
    return us_mu(1000*keV)

def mu_Air(keV=12):
    """Returns inverse 1/e penetration depth [mm-1] of air given the
    x-ray energy in keV. The transmission through a 300-mm thick slab of air
    was calculated every 100 eV over an energy range spanning 5-17 keV using:
        http://henke.lbl.gov/optical_constants/filter2.html
    This result was then converted to mu and saved as a tab-delimited text
    file. The returned result is calculated using a univariate spline and
    should be valid over the range 5-17 keV."""
    from numpy import loadtxt
    from scipy.interpolate import UnivariateSpline
    E_mu = loadtxt('mu_Air.txt',dtype=float,delimiter='\t')
    us_mu = UnivariateSpline(E_mu[:,0],E_mu[:,1],s=0)
    return us_mu(1000*keV)

def mu_air(BP,RH,TC):
    """Returns the inverse 1/e penetration depth [mm-1] as a function of
    barometric pressure (BP) in torr, relative humidity (RH) in %, and
    temperature in Celsius. The expressions below were derived from equations
    and tabulated data found in:
        https://en.wikipedia.org/wiki/Antoine_equation
    The gas transmission was calculated at 12 keV and 295 K using:
        http://henke.lbl.gov/optical_constants/gastrn2.html
    The temperature dependence of the returned absorption coefficient assumes
    ideal gas behavior. The absorbance coefficient returned corresponds to 12 keV.
    """
    A,B,C = 8.07131,1730.63,233.426
    VP = (RH/100)*10**(A - B/(C+TC))
    mu_a = 4.3803e-07 #(mm-1 torr-1 @ 12 keV)
    mu_wv = 2.8543e-07 #(mm-1 torr-1 @ 12 keV)
    return (BP-VP)*mu_a*295/(273+TC)+VP*mu_wv

def filter_transmittance(theta,BP=760,RH=35,TC=22,keV=12):
    """Returns theta-dependent transmittance of x-rays through all
    materials encountered between the sample capillary and the phosphor in
    front of the x-ray detector. Names, thicknesses, and absorption coefficients
    [mm-1] calculated at 12 keV are hard-coded, except for the humidified air
    contribution, which is a function of BP, RH, and TC. The absorption
    coefficient for helium has been scaled according to BP and TC assuming
    ideal gas behavior. The absorbance coefficients are assumed to scale as the
    inverse cubed power of the x-ray energy."""
    from numpy import array,exp,cos
    material = ['Helium','Polyimide','Humidified Air','ParyleneN','Be','Mylar']
    t = array([85.8,0.1,96.3,0.02,0.03,0.05])
    mu = array([2.90799e-5*295/(273+TC)*BP/760,0.24539,mu_air(BP,RH,TC),0.13368,0.069009,0.26434])
    return exp(-(mu*t).sum()*(12/keV)**3/cos(2*theta))

def geo_dictionary(image,X0=1991,Y0=1973,distance=185.8,pixelsize=0.089,polarization=1):
    """Creates a dictionary containing psi, theta, polfactor, and geofactor
    in the same shape as image. Also includes X0, Y0, sort_indices, and
    reverse_indices. sort_indices are used to sort any flattened image in
    increasing r; reverse_indices are used to reconstruct an image from a
    flattened, r-sorted array."""
    from numpy import indices,sqrt,arctan2,arctan,sin,cos,square,argsort
    # polarization = 1 for Horiz polarization; -1 for Vert polarization.

    # Compute quantities in same shape as image
    row,col = image.shape
    y_indices,x_indices = indices((row,col))
    r = pixelsize*sqrt((y_indices-Y0)**2+(x_indices-X0)**2)
    psi = -arctan2((y_indices-Y0),(x_indices-X0))
    theta = arctan(r/distance)/2
    polfactor = (1+square(cos(2*theta))-polarization*cos(2*psi)*square(sin(2*theta)))/2
    geofactor = cos(2*theta)**3

    # Generate sort_indices and reverse_indices
    sort_indices = argsort(r.flatten())
    reverse_indices = argsort(sort_indices)

    # Assemble dictionary
    geo_dict = {'X0':X0,'Y0':Y0,'psi':psi,'theta':theta,\
                'polfactor':polfactor,'geofactor':geofactor,\
                'sort_indices':sort_indices,'reverse_indices':reverse_indices,\
                'rpix':r/pixelsize}
    return geo_dict

def q_dictionary(geo_dict,keV=12,dq=0.0025):
    """Creates a dictionary containing q_flat, q, qbin1, and qbin2, which are
    needed for integration. The dictionary also contains theta_q, wavelength,
    and keV, which are made available for convenience. """
    from numpy import pi,sin,arange,rint,where,roll,cast,int16,arcsin
    theta = geo_dict['theta']
    sort_indices = geo_dict['sort_indices']
    theta_flat = theta.flatten()[sort_indices]
    h = 4.135667696e-15 # Planck's constant [eV-s]
    c = 299792458e10 # speed of light [Angstroms per second]
    wavelength = h*c/(keV*1000) # x-ray wavelength [Angstroms]

    # Compute q_flat from theta_flat
    q_flat = 4*pi*sin(theta_flat)/wavelength
    Nq = int(q_flat[-1]/dq)
    q = arange(0,dq*Nq,dq)
    qbin = cast[int16](rint(q_flat/dq))
    qbin2 = where(qbin != roll(qbin,-1))[0]
    qbin1 = roll(qbin2,1)+1
    qbin1[0] = 0

    # Omit last bin to ensure sufficient number of unmasked pixels in last bin
    q = q[:-1]
    qbin1 = qbin1[:-1]
    qbin2 = qbin2[:-1]

    # Ensure last entry corresponds to last pixel
    qbin2[-1] = len(q_flat)-1

    # Calculate theta for q
    theta_q = arcsin(wavelength*q/(4*pi))

    # Assemble dictionary
    qdict = {'q_flat':q_flat,'q':q,'qbin1':qbin1,'qbin2':qbin2,\
             'theta_q':theta_q,'wavelength':wavelength,'keV':keV}
    return qdict


def us_fit(q,S,sigS,model=None,w_scale=1.0,k=3):
        """Generates a univariate spline fit of S vs. q. The weight is scaled
        by w_scale to optimize the number of knots used in the fit. If
        model = None, a straight line fit of the last 10 points in S is used to
        define the last three points of S, and their weights are increased to
        ensure the univariate spline fit terminates with the correct slope. If
        model is not None, the slope of the last 3 data points in S are
        modified to ensure the final slope of the spline fit is defined by the
        last three points in the model."""
        from scipy.interpolate import UnivariateSpline
        from numpy import polyfit
        Sc = S.copy()
        sigSc = sigS.copy()
        if model is None:
            p = polyfit(q[-10:], S[-10:], 1, w=1/sigS[-10:])
            Sc[-3:] = q[-3:]*p[0] + p[1]
        else:
            Sc[-3:] = model[-3:]*(S[-20:]/model[-20:]).mean()
        sigSc[-3:] /= 100
        return UnivariateSpline(q,Sc,w=w_scale/sigSc,k = k)

def integrate(Zmask_flat,I_flat,V_flat,q_dict,sigma=5):
    """Returns the weighted mean (S) and standard deviation of the mean (sigS)
    for each bin defined by q, qbin1, and qbin2. Requires as input Zmask_flat
    (boolean), I_flat (counts), V_flat (variance), and q_flat, which are
    flattened and sorted (according to qpix) versions of Zmask, I, V and qpix.
    Also required are q, qbin1, and qbin2, which assigns pixels to bins. The
    data in each bin are fitted to a straight line (I = a + b*q), with the
    weighted mean representing the value of the line at the corresponding value
    of q, and the standard deviation of the mean corresponding to its
    experimental uncertainty. Statistical criteria are employed to identify and
    reject outliers before returning the weighted mean and its standard
    deviation. Also returns N_pixels, N_outliers, and outliers_flat. Prior to
    calling this function, I_flat must be set to zero and V_flat must be set
    to one where Zmask (to exclude masked pixels and avoid dividing by zero).
    """
    from numpy import cumsum,sqrt,zeros,uint16,absolute,median
    q = q_dict['q']
    q_flat = q_dict['q_flat']
    qbin1 = q_dict['qbin1']
    qbin2 = q_dict['qbin2']
    S = zeros(len(q))
    sigS = zeros(len(q))
    N_pixels = zeros(len(q),dtype=uint16)
    N_outliers = zeros(len(q),dtype=uint16)
    outliers_flat = zeros(len(q_flat),bool)
    N_pixels[0] = 1
    S[0]= I_flat[0]
    sigS[0] = sqrt(V_flat[0])
    # Calculate first few bins the old-fashioned way assuming no outliers
    for j in range(1,11):
        first = qbin1[j]
        last  = qbin2[j]
        N_pixels[j] = last-first+1
        S[j]= I_flat[first:last+1].mean()
        sigS[j] = I_flat[first:last+1].std()
    # Find outliers relative to the median within each bin
    for j in range(11,len(q)):
        first = qbin1[j]
        last  = qbin2[j]
        S[j] = median(I_flat[first:last+1][~Zmask_flat[first:last+1]])
        S_fit = S[j]+(q_flat[first:last+1] - q[j])*(S[j] - S[j-1])/q[1]
        outliers_flat[first:last+1] = ((absolute(I_flat[first:last+1] - S_fit)/sqrt(V_flat[first:last+1])) > sigma) & ~Zmask_flat[first:last+1]
        N_outliers[j] = outliers_flat[first:last+1].sum()
    # Omit outliers or masked pixels when calculating cumulative sums
    omit = outliers_flat | Zmask_flat
    cs_I  = cumsum(~omit*I_flat/V_flat)
    cs_I2 = cumsum(~omit*I_flat**2/V_flat)
    cs_q  = cumsum(~omit*q_flat/V_flat)
    cs_q2 = cumsum(~omit*q_flat**2/V_flat)
    cs_qI = cumsum(~omit*q_flat*I_flat/V_flat)
    cs_W  = cumsum(~omit/V_flat)
    cs_N =  cumsum(~omit)
    for j in range(4,len(q)):
        first = qbin1[j]
        last  = qbin2[j]
        sI  =  cs_I[last] -  cs_I[first-1]
        sI2 = cs_I2[last] - cs_I2[first-1]
        sq  =  cs_q[last] -  cs_q[first-1]
        sq2 = cs_q2[last] - cs_q2[first-1]
        sqI = cs_qI[last] - cs_qI[first-1]
        sW  =  cs_W[last] -  cs_W[first-1]
        delta = sW*sq2 - sq**2
        a =  (1/delta)*(sq2*sI - sq*sqI)
        b =  (1/delta)*(sW*sqI - sq*sI)
        S[j] = a + b*q[j]
        N_pixels[j] = cs_N[last] -  cs_N[first-1]
        variance = (sI2 - 2*a*sI - 2*b*sqI + 2*a*b*sq + b*b*sq2 + a*a*sW)/sW/N_pixels[j]
        sigS[j] = sqrt(abs(variance))

    return S,sigS,N_pixels,N_outliers,outliers_flat

def integrate_old(mask_flat,I_flat,V_flat,q_flat,q,qbin1,qbin2,sigma=3):
    """Returns the weighted mean (S) and standard deviation of the mean (sigS)
    for each bin defined by q, qbin1, and qbin2. Requires as input mask_flat
    (boolean), I_flat (counts), V_flat (variance), and q_flat, which are
    flattened and sorted (according to qpix) versions of mask, I, V and qpix.
    Also required are q, qbin1, and qbin2, which assigns pixels to bins. The
    data in each bin are fitted to a straight line (I = a + b*q), with the
    weighted mean representing the value of the line at the corresponding value
    of q, and the standard deviation of the mean corresponds to its
    experimental uncertainty. Statistical criteria are employed to identify and
    reject outliers before returning the weighted mean and its standard
    deviation. Also returns N_pixels, N_outliers, and outliers_flat.
    """
    from numpy import cumsum,sqrt,zeros,uint16,absolute,median
    S = zeros(len(q))
    sigS = zeros(len(q))
    N_pixels = zeros(len(q),dtype=uint16)
    N_outliers = zeros(len(q),dtype=uint16)
    outliers_flat = zeros(len(q_flat),bool)
    N_pixels[0] = 1
    S[0]= I_flat[0]
    sigS[0] = sqrt(V_flat[0])
    # Calculate first few bins the old-fashioned way assuming no outliers
    for j in range(1,11):
        first = qbin1[j]
        last  = qbin2[j]
        N_pixels[j] = last-first+1
        S[j]= I_flat[first:last+1].mean()
        sigS[j] = I_flat[first:last+1].std()
    # Find outliers relative to the median within each bin
    for j in range(11,len(q)):
        first = qbin1[j]
        last  = qbin2[j]
        S[j] = median(I_flat[first:last+1][~mask_flat[first:last+1]])
        S_fit = S[j]+(q_flat[first:last+1] - q[j])*(S[j] - S[j-1])/q[1]
        outliers_flat[first:last+1] = ((absolute(I_flat[first:last+1] - S_fit)/sqrt(V_flat[first:last+1])) > sigma) & ~mask_flat[first:last+1]
        N_outliers[j] = outliers_flat[first:last+1].sum()
    # Omit outliers or masked pixels when calculating cumulative sums
    omit = outliers_flat | mask_flat
    V_flat[V_flat == 0] = 1 # Prevents nan when V_flat = 0
    cs_I  = cumsum(~omit*I_flat/V_flat)
    cs_I2 = cumsum(~omit*I_flat**2/V_flat)
    cs_q  = cumsum(~omit*q_flat/V_flat)
    cs_q2 = cumsum(~omit*q_flat**2/V_flat)
    cs_qI = cumsum(~omit*q_flat*I_flat/V_flat)
    cs_W  = cumsum(~omit/V_flat)
    cs_N =  cumsum(~omit)
    for j in range(4,len(q)):
        first = qbin1[j]
        last  = qbin2[j]
        sI  =  cs_I[last] -  cs_I[first-1]
        sI2 = cs_I2[last] - cs_I2[first-1]
        sq  =  cs_q[last] -  cs_q[first-1]
        sq2 = cs_q2[last] - cs_q2[first-1]
        sqI = cs_qI[last] - cs_qI[first-1]
        sW  =  cs_W[last] -  cs_W[first-1]
        delta = sW*sq2 - sq**2
        a = (1/delta)*(sq2*sI - sq*sqI)
        b = (1/delta)*(sW*sqI - sq*sI)
        S[j] = a + b*q[j]
        sigS[j] = sqrt((sI2 - 2*a*sI - 2*b*sqI + 2*a*b*sq + b*b*sq2 + a*a*sW)/sW)
        N_pixels[j] = cs_N[last] -  cs_N[first-1]
    return S,sigS,N_pixels,N_outliers,outliers_flat

def UC_psi(Z_mask,I,V,q_dict,geo_dict,Z=10):
    """Generates uniformity correction image given input parameters. Returns
    dictionary that contains vectors: {q, S, sigS, N_pixels, N_outliers} and
    images: {outliers, Sfit, Ic, and UCpsi}. """
    q = q_dict['q']
    q_flat = q_dict['q_flat']
    polfactor = geo_dict['polfactor']
    geofactor = geo_dict['geofactor']
    sort_indices = geo_dict['sort_indices']
    reverse_indices = geo_dict['reverse_indices']
    Correction = polfactor*geofactor
    Ic = I/Correction
    Vc = V/Correction**2
    Z_flat = Z_mask.flatten()[sort_indices]
    I_flat = Ic.flatten()[sort_indices]
    V_flat = Vc.flatten()[sort_indices]
    I_flat[Z_flat]=0
    V_flat[Z_flat]=1
    S,sigS,N_pixels,N_outliers,outliers_flat = integrate(Z_flat,I_flat,V_flat,q_dict,sigma=Z)
    w_scale = 0.9
    us = us_fit(q,S,sigS,model=None,w_scale=w_scale)
    Sfit = us(q)
    Sfit_flat = us(q_flat)
    S_image = Sfit_flat[reverse_indices].reshape(Ic.shape)
    outliers = outliers_flat[reverse_indices].reshape(Ic.shape)
    UCpsi = Ic/S_image
    UCdict = {'q':q,'S':S,'sigS':sigS,'N_pixels':N_pixels,'N_outliers':N_outliers,\
              'outliers':outliers,'Sfit':Sfit,'Ic':Ic,'UCpsi':UCpsi}
    return UCdict

def CCD_select(image,row=1,col=2):
    """Returns subset of image defined by row and column, with 0,0 representing
    the upper left CCD. Assumes 3840,3840 with 4x4 array of CCD chips. """
    return image[row*960:(row+1)*960,col*960:(col+1)*960]

def CCD_scale(UC):
    """Returns 2D scale according to mean UC_psi for each CCD chip. Assumes
    3840,3840 with 4x4 array of CCD chips. The result is normalized to
    scale[2,2], which corresponds to the CCD with the beamstop."""
    from numpy import zeros
    scale = zeros((4,4))
    for i in range(4):
        for j in range(4):
            CCDij = UC[i*960:(i+1)*960,j*960:(j+1)*960]
            scale[i,j] = CCDij[CCDij>0].mean()
    scale /= scale[2,2]
    return scale

def CCD_r():
    from numpy import zeros,sqrt
    """Returns 2D image that specifies distance in pixels from the center of
    each CCD chip. Assumes 3840,3840 image with 4x4 array of CCD chips."""
    from numpy import indices
    r = zeros((3840,3840))
    y_indices,x_indices = indices((960,960))
    r_CCD = sqrt((y_indices-(960-1)/2)**2+(x_indices-(960-1)/2)**2)
    for i in range(4):
        for j in range(4):
            r[i*960:(i+1)*960,j*960:(j+1)*960] = r_CCD
    return r

def CCD_norm(scale):
    """Returns 2D normalization image according to scale, which is specified
    for each CCD chip. Assumes 3840,3840 with 4x4 array of CCD chips."""
    from numpy import ones
    norm = ones((3840,3840))
    for i in range(4):
        for j in range(4):
            norm[i*960:(i+1)*960,j*960:(j+1)*960] = norm[i*960:(i+1)*960,j*960:(j+1)*960]*scale[i,j]
    return norm

def S_Compton(q,Z):
    """Reads S_Compton.txt file and uses UnivariateSpline to generate S(q)."""
    from numpy import loadtxt,pi
    from scipy.interpolate import UnivariateSpline
    S = loadtxt('S_Compton.txt',dtype=float,delimiter='\t')
    q_data = S[0]*4*pi
    us = UnivariateSpline(q_data,S[Z],s=0)
    return us(q)

def FF_C(q,Z):
    """Reads form factor and Compton scattering files and uses
    UnivariateSpline to return FF(q) and S(q)."""
    from numpy import loadtxt,pi,where
    from scipy.interpolate import UnivariateSpline
    FF = loadtxt('FF(1-60 and 71-85).txt',dtype=float,delimiter='\t')
    q_data = 4*pi*FF[0,1:]#*qdict['wavelength']
    us_FF = UnivariateSpline(q_data,FF[where(FF[:,0]==Z)[0][0],1:],s=0)
    C = loadtxt('S(1-60 and 71-85).txt',dtype=float,delimiter='\t')
    us_C = UnivariateSpline(q_data,C[where(C[:,0]==Z)[0][0],1:],s=0)
    return us_FF(q),us_C(q)

def circular_indices(image,X0,Y0,r):
    from numpy import indices,sqrt,where
    height,width = image.shape
    y_indices,x_indices = indices((height,width))
    where(sqrt((y_indices-Y0)**2 + (x_indices-X0)**2) < r)
    return where(sqrt((y_indices-Y0)**2 + (x_indices-X0)**2) < r)

def circular_mask(image,X0,Y0,r):
    from numpy import indices,sqrt
    height,width = image.shape
    y_indices,x_indices = indices((height,width))
    return sqrt((y_indices-Y0)**2 + (x_indices-X0)**2) < r

def Guinier(q,I0,Rg):
    """Returns Guinier approximation to the SAXS intensity distribution, as
    parameterized by I0 and Rg, as a function of q."""
    from numpy import exp
    return I0*exp(-(q*Rg)**2/3)

def flat_to_image_old(image_flat,mask,sort_indices):
    """Reconstructs 2D image from image_flat using as input 'mask' and
    'sort_indices'. For example, qpix is flattened and sorted according to:

        q_select = where(~mask)
        sort_indices = argsort(qpix[q_select])
        q_flat = qpix[q_select][sort_indices]

    This function effectively reverses this process and regenerates qpix."""
    from numpy import argsort,zeros_like,where
    reverse = argsort(sort_indices)
    datatype = image_flat.dtype
    image = zeros_like(mask,dtype=datatype)
    image[where(~mask)] = image_flat[reverse]
    return image

def unique(values):
    """Returns list of unique entries in values."""
    unique = []
    for value in values:
        if not value in unique:
            unique.append(value)
    return unique

def VPC_dict(N,Dvar,Imean,Ivar):
    """Generates VPC and VPCvar from Imean and Ivar data given N and
    Dvar. It is assumed the background offset (Dmean) has been subtracted from
    Imean. VPC corresponds to variance per count; when mulitplied by counts,
    this scale factor converts counts to variance."""
    from numpy import isnan,maximum,seterr
    seterr(divide='ignore',invalid='ignore')
    VPC = maximum((Ivar-Dvar),Dvar)/Imean
    VPCvar = VPC**2*((Ivar+Dvar)/Imean**2 + 2/(N-1))
    VPC[isnan(VPC)] = 0
    VPCvar[isnan(VPCvar)] = 0
    seterr(divide='warn',invalid='warn')
    VPC_dict = {'VPC':VPC,'VPCvar':VPCvar}
    return VPC_dict

def NPC_dict(N,Dvar,Imean,Ivar):
    """Generates NPC and NPCvar from Imean and Ivar data given N and
    Dvar. It is assumed the background offset has been subtracted from Imean.
    NPC corresponds to the number of x-ray photons required to generate one
    count in the corresponding pixel (assuming variance is determined by
    photon-couting statistics)."""
    from numpy import isnan,maximum,seterr
    seterr(divide='ignore',invalid='ignore')
    NPC = Imean/maximum((Ivar-Dvar),Dvar)
    NPCvar = (Ivar+Dvar)/maximum((Ivar-Dvar),Dvar)**2 + 2*NPC**2/(N-1)
    NPC[isnan(NPC)] = 0
    NPCvar[isnan(NPCvar)] = 0
    seterr(divide='warn',invalid='warn')
    NPC_dict = {'NPC':NPC,'NPCvar':NPCvar}
    return NPC_dict

def image_symmetrize_zeros(image,qdict):
    from numpy import ones_like,zeros_like
    X0 = qdict['X0']
    Y0 = qdict['Y0']
    row,col = image.shape
    perimeter_mask = ones_like(image,dtype=bool)
    perimeter_mask[2:-2,2:-2] = False
    xr = min(X0,col-1-X0)
    yr = min(Y0,row-1-Y0)
    r = min(xr,yr)
    ZP = (image == 0) & ~perimeter_mask
    ZP_T = zeros_like(image,dtype=bool)
    ZP_T[Y0-r:Y0+r+1,X0-r:X0+r+1] = ZP[Y0-r:Y0+r+1,X0-r:X0+r+1].T
    sym = ZP | ZP_T
    image[sym] = 0
    return image

def image_SVD(I):
    """Performs SVD on array of images (N,row,column). Returns U,s,V.
    """
    from scipy.linalg import svd
    from numpy import reshape
    from time import time
    t0 = time()
    N_images,w,h = I.shape
    print("PERFORMING SINGULAR VAULE DECOMPOSITION OF {} IMAGES...".format(N_images))
    I = reshape(I,(N_images,w*h))       # Reshape 3D array to 2D.
    U,s,V = svd(I.T,full_matrices=0)
    U = reshape(U.T,(N_images,w,h))     # Reshape 2D array to 3D.
    for i in range(N_images):
        if V[i].max() < abs(V[i].min()):
            U[i] *= -1
            V[i] *= -1
    t1=time()
    print('{:0.3f} seconds to SVD {} images.'.format(t1-t0,N_images))
    return U,s,V

def ptbs_scale(I,A,X0,Y0):
    """Find optimal scale factor 'a' for subtracting A from I."""
    from analysis_functions import beamcenter_footprint
    from numpy import where
    footprint=beamcenter_footprint()
    offset = int(len(footprint)/2)
    X1 = X0 - offset
    X2 = X0 + offset+1
    Y1 = Y0 - offset
    Y2 = Y0 + offset+1
    Ipix = I[Y1:Y2,X1:X2][where(footprint)]
    Apix = A[Y1:Y2,X1:X2][where(footprint)]
    Ipix_mean = Ipix.mean()
    Apix_mean = Apix.mean()
    return (Ipix*Apix-Apix*Ipix_mean-Ipix*Apix_mean+Ipix_mean*Apix_mean).sum()/(Apix**2-2*Apix*Apix_mean+Apix_mean**2).sum()

def mask_perimeter(image,N_pixels=2):
    from numpy import ones_like
    mask = ones_like(image,dtype=bool)
    mask[N_pixels:-N_pixels,N_pixels:-N_pixels] = False
    return mask

def defective_pixel_mask(image,npix=1):
    """Generates mask of regions with zeros, including 1 pixel around
    the zero regions, OR perimeter mask with npix border pixels."""
    from numpy import ones_like
    perimeter = ones_like(image,dtype=bool)
    perimeter[npix:-npix,npix:-npix] = False
    zeros = (image == 0)
    return grow_mask(zeros*~perimeter,1) | perimeter

def stats_mode(array,binsize = 0.01):
    """Returns mode for array after binning the data according to binsize."""
    from numpy import cast,int32,rint
    from scipy.stats import mode
    return mode(cast[int32](rint(array[array>0]/binsize)))[0][0]*binsize

def find_filenames_walk(topdir=None,pattern='*.*',exclude=['xray_images','xray_traces','laser_traces','xray_images_unassigned']):
    """Returns filenames found under 'topdir' that match 'pattern', which must
    be preceded with an '*'. The search excludes subfolder names specified by
    the 'exclude' list."""
    import re
    from os import walk
    from numpy import sort
    from time import time
    t0 = time()
    excluded = set(['.DS_Store','.AppleDouble','backup']) | set(exclude)
    re_pattern = re.compile(("^"+pattern).replace(".", "\.").replace("*", ".*").replace("?", ".")+"$")
    f = []
    for root, dirs, files in walk(topdir, topdown=True):
        [dirs.remove(d) for d in list(dirs) if d in excluded]
        for file in files:
            if re_pattern.match(file):
                f += [root+'/'+file]
    # Remove hidden files
    hidden = ['.DS_Store','.AppleDouble','.adxv_beam_center','._']
    filenames = f.copy()
    [filenames.remove(file) for file in f if any([name in file for name in hidden])]
    print('{:0.3f} seconds to find {} files containing "{}"'.format(time()-t0,len(filenames),pattern))
    return sort(filenames)

def find_filenames_path(path_name=None,pattern='*.*'):
    """Returns filenames found in 'path' which match 'pattern', which must be
    preceded with an '*'. The search excludes filenames specified by the
    hard-coded set 'excluded', as well as hidden files with '._' in the filename."""
    import re
    from os import listdir,path
    from numpy import sort
    f = [path.join(path_name,file) for file in listdir(path_name)]
    # Remove hidden files
    hidden = ['.DS_Store','.AppleDouble','.adxv_beam_center','._']
    filenames = f.copy()
    [filenames.remove(file) for file in f if any([name in file for name in hidden])]
    # Remove files that don't match pattern
    f = filenames.copy()
    re_pattern = re.compile(("^"+pattern).replace(".", "\.").replace("*", ".*").replace("?", ".")+"$")
    [filenames.remove(file) for file in f if not re_pattern.match(file)]
    return sort(filenames)

def save_npy(beamtime,name,np_array):
    """Appends '.npy' to 'name' and saves np_array in '/common files/'
    directory. Since name is assumed to be a string, it must be enclosed in
    quotes."""
    from numpy import save
    analysis_dir = '//femto/C/SAXS-WAXS Analysis/beamtime/'+beamtime
    common_dir = '/common files/'
    name = name+'.npy'
    save(analysis_dir+common_dir+name,np_array)

def load_npy(beamtime,name):
    """Appends '.npy' to 'name' and loads it from '/common files/' directory.
    Since name is assumed to be a string, it must be enclosed in quotes."""
    from numpy import load
    analysis_dir = '//femto/C/SAXS-WAXS Analysis/beamtime/'+beamtime
    common_dir = '/common files/'
    name = name+'.npy'
    return load(analysis_dir+common_dir+name)

def save_to_file(filename,object,):
    """This function saves data to a file, e.g. a dictionary object. It is used
    in conjuction with load_from_file."""
    from pickle import dump
    with open(filename,"wb") as f:
        dump(object,f)

def load_from_file(filename):
    """This function loads data from a file, e.g. a dictionary object. It is
    used in conjuction with save_to_file."""
    from pickle import load
    with open(filename,'rb') as f:
        data = load(f, encoding='bytes')
    return data

def lmfit_summary(basename,N,out):
    """Returns tab-separated header_str and data_str. The header_str spans two
    rows with the first row specifiying the fit variable names and the second
    row identifying (value,stderr) pairs. The first three columns in the
    data_str are reserved for basename, N, and redchi; subsequent columns
    contain interleaved values and stderrs for the fit."""
    from numpy import ravel,column_stack
    params = out.params
    names = [par.name for name, par in params.items()]
    values = [str(par.value) for name, par in params.items()]
    stderrs = [str(par.stderr) for name, par in params.items()]
    header0 = ['$','N','redchi'] + list(ravel(column_stack((names,names))))
    header0_str = '\t'.join(header0) + '\n'
    header1 = ['$','',''] + ['value','stderr']*len(names)
    header1_str = '\t'.join(header1) + '\n'
    header_str = header0_str + header1_str
    data = [basename,str(N),str(out.redchi)]+list(ravel(column_stack((values,stderrs))))
    data_str = '\t'.join(data) + '\n'
    return header_str,data_str

def file_write_append(filename,header,data):
    """If filename doesn't exist, create it and write both header string and
    data string; if exists, append data string. An exception is created if
    the folder doesn't exist."""
    from os.path import exists
    if not exists(filename):
        with open(filename, 'w') as file:
            file.write(header)
            file.write(data)
    else:
        with open(filename, 'a') as file:
            file.write(data)

def images_scale(images,offset = 9.71):
    """Returns images_s and scale, where images_s corresponds to the rescaled
    images, and scale is the normalized vector used to rescale the images.
    The offset is subtracted before rescaling, and added back to the scaled
    result."""
    from numpy import cast,float32
    images = cast[float32](images)
    scale = (images-offset).sum(axis=1).sum(axis=1)
    scale = scale/scale.mean()
    images_s = ((images-offset).T/scale).T + offset
    return images_s,scale

def images_zinger_replaced(images,sigma=5.730729):
    """Returns images_zr after replacing zingers in images with median values.
    Zingers are identified according to statistical criteria. Also returns the
    indices for images in which zingers were found. Note that sigma = 5.730729
    corresponds to one part in 100 million."""
    from analysis_functions import unique
    from numpy import median,std,cast,float32,any,where
    images = cast[float32](images)
    images_median = median(images,axis=0)
    images_std = std(images,axis=0)
    zingers = (images-images_median)/images_std > sigma
    images_zr = where(zingers,images_median,images)
    #images_zr = ~zingers*images + zingers*images_median
    # Recalculate images_std using zinger replaced images
    images_std = std(images_zr,axis=0)
    zingers = (images-images_median)/images_std > sigma
    zingers_any = any(zingers,axis=0)
    images_zr = where(zingers,images_median,images)
    #images_zr = ~zingers*images + zingers*images_median
    indices = unique(where(zingers)[0])
    print('Found {} outliers involving {} pixels in images {}'.format(zingers.sum(),zingers_any.sum(),indices))
    return images_zr,indices

def z_statistic(N):
    """Returns z, which corresponds to the threshold in units of sigma where
    the probability of measuring a value greater than z*sigma above the mean
    is 1 out of N."""
    from numpy import sqrt
    from scipy.special import erfinv
    return sqrt(2)*erfinv(1-2/N)

def poisson_array(M1,N,sort=True):
    """Returns aray of N numbers according to poisson distribution
    parameterized by M1 and N (mean and # values); if sort=True, values are
    sorted."""
    from numpy import sort
    from numpy.random import poisson
    dist = poisson(M1,N)
    if sort: dist = sort(dist)
    return dist

def process_data(root,terms,sub=0,overwrite=False,persist=False, erase = False):
    """Finds files containing list of terms (including either '.raw.hdf5' or
    '.data.hdf5'), loads or generates stats, then generates *.roi.hdf5 files
    from each source file found. If sub=0, processes all files found;
    if sub=1, processes odd files only; if sub=2, processes even files only.
    If persist=False, exits when all files found have been processed;
    if persist=True, awaits appearance of new files. If overwrite=True,
    existing *.roi.hdf5 files are overwritten."""
    from analysis_functions import find_pathnames,stats_from_chunk,\
        images_from_file,data_to_roi,roi_to_file
    from numpy import fromstring,array
    from time import time,sleep
    from os import path, remove

    def files_to_process(sub,overwrite):
        """Finds files defined by terms; If overwrite=False, removes those for
        which corresponding *.roi.hdf5 files already exist, so they are not
        overwritten. This allows process_data to be terminated and restarted
        without repeating the analysis."""
        path_names = find_pathnames(root,terms)
        if not sub == 0:
            repeat =[]
            [repeat.append(fromstring(path_name.split('_')[-1].split('.')[0],dtype='int16',sep=' ')[0]) for path_name in path_names]
            repeat = array(repeat)
            r_odd = repeat%2 == 1
            if sub == 1: path_names = array(path_names)[r_odd]
            if sub == 2: path_names = array(path_names)[~r_odd]
        f = path_names.copy()
        path_names = list(path_names)
        if not overwrite:
            if '.data.hdf5' in path_names[0]:
                [path_names.remove(file) for file in f if path.exists(file.replace('.data.hdf5','.roi.hdf5'))]
            if '.raw.hdf5' in path_names[0]:
                [path_names.remove(file) for file in f if path.exists(file.replace('.raw.hdf5','.roi.hdf5'))]
        return path_names

    # Load or compute stats for first chunk in dataset.
    median,mean,var,threshold = stats_from_chunk(find_pathnames(root,terms)[0])

    while True:
         # Find path_names for which a corresponding *.roi.hdf5 file doesn't exist
         path_names = files_to_process(sub,overwrite)
         #  To avoid repeatedly processing the same dataset, set overwrite = False
         #     (unnecessary if *.raw.hdf5 files were deleted after processing)
         if overwrite == True: overwrite = False
         if len(path_names)>0:
             for path_name in path_names:
                t0 = time()
                print('Processing {} ...'.format(path_name))
                images = images_from_file(path_name)
                hits0,hits1,hits2,hits_coord,mask,roi,dt_process = data_to_roi(images,median,threshold,dilation=10)
                roi_to_file(path_name,hits0,hits1,hits2,hits_coord,mask,roi,dt_process)
                print('Total time to process images[s]: {:0.3f}'.format(time()-t0))
                if erase:
                    print(f"pathname to erase {path_name} ...")
                    remove(path_name)
                # If wish to delete *.raw.hdf5 files after processing, put that line of code here
         else:
             if persist:
                 print('\r Waiting for new files...',end='\r')
                 sleep(60)
             else:
                 print('No files left to process.')
                 return

def images_from_file(path_name,frame=-1):
    """Returns images from file specified by 'path_name'. If frame=-1, returns
    an array of images; if frame is a non-negative integer, returns a single
    image specified by frame. If '.raw.hdf5' is found in 'path_name', the file
    is converted from a flat mono12p format to 16-bit image format."""
    from numpy import empty
    from time import time
    import h5py
    t0 = time()
    f = h5py.File(path_name,'r')
    if '.raw.hdf5' in path_name:
        row = f['image height'][()]
        col = f['image width'][()]
        raw = f['images']
        if frame == -1:
            N = len(raw)
            images = empty((N,row,col),dtype='int16')
            for i in range(N):
                images[i] = mono12p_to_image(raw[i],row,col)
        else:
            images = mono12p_to_image(raw[frame],row,col)
    else:
        if frame == -1:
            images = f['images'][()]
        else:
            images = f['images'][frame]
    print('time to generate images [s]: {:0.3f}'.format(time()-t0))
    return images

def mono12p_to_image_ver1(rawdata,row,col):
    """Converts FLIR Mono12p flat data format (uint8) to 16 bit image (uint16)."""
    from numpy import right_shift,bitwise_and,empty,uint16
    arr = rawdata.reshape(-1,3)
    byte_even = arr[:,0] + 256*(bitwise_and(arr[:,1],15))
    byte_odd = right_shift(arr[:,1],4) + right_shift(256*arr[:,2],4)
    img_flat = empty((row*col),uint16)
    img_flat[0::2] = byte_even
    img_flat[1::2] = byte_odd
    return img_flat.reshape((row,col))

def mono12p_to_image(arr,row,col):
    """Converts FLIR Mono12p flat data format (uint8) to 16 bit image (uint16)."""
    from numpy import right_shift,bitwise_and,empty,uint8,uint16
    image_flat = empty(int(len(arr)*4/3),uint8)
    image_flat[0::4] = arr[0::3]
    image_flat[1::4] = bitwise_and(arr[1::3],15)
    image_flat[2::4] = bitwise_and(arr[1::3],240)
    image_flat[3::4] = arr[2::3]
    image_flat = image_flat.view(uint16)
    image_flat[1::2] = right_shift(image_flat[1::2],4)
    return image_flat.reshape((row,col))

def mono12p_to_image_ver2(rawdata,row,col):
    """Converts FLIR Mono12p flat data format (uint8) to 16 bit image (uint16)."""
    from numpy import right_shift,bitwise_and,empty,uint8,uint16
    arr = rawdata.reshape(-1,3)
    N = 2*len(arr)
    b0 = empty(N,uint8)
    b1 = empty(N,uint8)
    b0[0::2] = arr[:,0]
    b0[1::2] = bitwise_and(arr[:,1],15)
    b1[0::2] = bitwise_and(arr[:,1],240)
    b1[1::2] = arr[:,2]
    image_flat = empty(N,uint16)
    image_flat[0::2] = b0.view(uint16)
    image_flat[1::2] = right_shift(b1.view(uint16),4)
    return image_flat.reshape((row,col))

def stats_from_chunk(reference,sigma=6):
    """Returns mean, var, and threshold (in counts) for reference. The mean
    and var are calculated after omitting the largest and smallest values
    found for each pixel, which assumes few particles in the laser
    sheet. The threshold statistic corresponds to the specified sigma level.
    Adding 0.5 to var helps compensate for digitization error so that false
    positives in the light sheet approximately match that outside the light
    sheet. The threshold for pixels defined by 'mask' are reset to 4095, which
    ensures they won't contribute to hits."""
    from analysis_functions import poisson_array,dm16_mask,images_from_file,save_to_file
    from numpy import sqrt,ceil,cast,array,zeros_like
    from time import time
    from os import path
    t0=time()
    if '.data.' in reference: stats_name = reference.replace('.data.hdf5','.stats.pkl')
    if '.raw.'  in reference: stats_name = reference.replace('.raw.hdf5','.stats.pkl')
    if  path.exists(stats_name):
        stats = load_from_file(stats_name)
        median = stats['median']
        mean = stats['mean']
        var = stats['var']
        threshold = stats['threshold']
    else:
        print('Processing {} ... please wait'.format(reference))
        # Load images and sort in place to minmimze memory footprint
        images = images_from_file(reference)
        images.sort(axis=0)
        mask = zeros_like(images[0])
        if 'dm16' in reference: mask = dm16_mask()
        # Compute median, then mean and var after omitting smallest and largest values.
        N = len(images)
        M = int(N/2)
        median = images[M]
        mean = images[1:-1].mean(axis=0,dtype='float32')
        var = images[1:-1].var(axis=0,dtype='float32')
        # Compute std_ratio; used to rescale stdev.
        std_ratio = []
        for i in range(10000):
            dist = poisson_array(3,N,sort=True)
            std_ratio.append(dist.std()/dist[1:-1].std())
        std_ratio_mean = array(std_ratio).mean()
        # Compute threshold to nearest larger integer; recast as int16.
        threshold = ceil(mean + sigma*std_ratio_mean*sqrt(var+0.5))
        threshold = cast['int16'](threshold) - median
        threshold[mask] = 4095
        save_to_file(stats_name,{'median':median,'mean':mean,'var':var,'threshold':threshold})
    print('time to execute stats_from_chunk [s]: {:0.1f}'.format(time()-t0))
    return median,mean,var,threshold

def dm16_mask():
    from numpy import zeros,indices
    mask = zeros((3000,4096),bool)
    y_indices,x_indices = indices((mask.shape))
    mask0 = y_indices > 1068-(94/979)*(x_indices-2921)
    mask1 = y_indices < 1792+(94/979)*(x_indices-2921)
    mask2 = x_indices > 2892
    mask3 = x_indices > 3900
    return ((mask0 & mask1) & mask2) | mask3

def find_pathnames(root,terms):
    """Searches 'root' and finds files that contains 'terms', which is a list
    of strings, i.e., ['dm16','.data.hdf5']. Returns list os sorted according
    to file timestamps."""
    from numpy import argsort,array
    from os import path,listdir
    from os.path import getmtime
    files = [path.join(root,file) for file in listdir(root)]
    select = terms
    selected = []
    [selected.append(file) for file in files if all([term in file for term in select])]
    path_names = selected.copy()
    if len(path_names) > 0:
        creation_times = [getmtime(file) for file in path_names]
        sort_order = argsort(creation_times)
        path_names = array(path_names)[sort_order]
    return path_names

def data_to_roi(images,median,threshold,dilation=10):
    from skimage.morphology import binary_dilation
    from numpy import uint16,uint32,where,around
    from time import time
    t0 = time()
    hits3 = images > (threshold + median)
    hits2 = hits3.sum(axis=0,dtype=uint16)
    hits1 = hits3.sum(axis=(1,2),dtype=uint32)
    hits0 = hits1.sum(dtype=uint32)
    t1 = time()
    print('time to find hits [s]: {:0.3f}'.format(t1-t0))
    hits_coord = where(hits3)
    t2 = time()
    print('time to compute hits_coord = where(hits3) [s]: {:0.3f}'.format(t2-t1))
    mask = hits2>0
    for i in range(dilation):
        mask = binary_dilation(mask)
    t3 = time()
    print('time to generate mask for roi [s]: {:0.3f}'.format(t3-t2))
    roi = (images-median)[:,mask]
    t4 = time()
    print('time to compute roi = (images-median)[:,mask] [s]: {:0.3f}'.format(t4-t3))
    dt_process = around(t4-t0,3)
    return hits0,hits1,hits2,hits_coord,mask,roi,dt_process

def roi_to_file(path_name,hits0,hits1,hits2,hits_coord,mask,roi,dt_process):
    """Extract information from path_name and include in *.roi.hdf5 file."""
    from time import time
    from os import utime,rename
    from os.path import getmtime
    import h5py
    t0 = time()
    if '.data.hdf5' in path_name: save_name = path_name.replace('.data.hdf5','.roi.hdf5')
    if '.raw.hdf5'  in path_name: save_name = path_name.replace('.raw.hdf5','.roi.hdf5')
    tmp_name = save_name.replace('roi','tmp')
    timestamp = getmtime(path_name)
    fraw = h5py.File(path_name,'r')
    row,col = mask.shape
    N = len(hits1)
    shape = (N,row,col)
    with h5py.File(tmp_name, 'w') as f:
        for key in fraw.keys():
            if key != 'images':
                f.create_dataset(key, data = fraw[key])
        f.create_dataset('shape', data = shape, dtype = 'uint32')
        f.create_dataset('hits0', data = hits0, dtype = 'uint32')
        f.create_dataset('hits1', data = hits1, dtype = 'uint32')
        f.create_dataset('hits2', data = hits2, dtype = 'uint16')
        f.create_dataset('hits_coord', data = hits_coord, dtype = 'int16')
        f.create_dataset('mask', data = mask, dtype = 'bool')
        f.create_dataset('roi', data = roi, dtype = 'int16')
        f.create_dataset('dt_process', data = dt_process, dtype = 'float32')
    rename(tmp_name,save_name)
    utime(save_name,(timestamp, timestamp))
    print('time to save hdf5 file [s]: {:0.3f}'.format(time()-t0))

def images_hits_reconstruct(roi_name,frame=-1):
    """Reconstructs images and hits from roi and hits_coord. If frame = -1,
    returns 3D versions; if a non-negative integer, returns image and hits for a
    single frame specified by 'frame'."""
    from numpy import zeros
    #from time import time
    import h5py
    #t0 = time()
    with h5py.File(roi_name,'r') as f:
        shape = f['shape'][()]
        mask = f['mask'][()]
        hits_coord = f['hits_coord'][()]
        roi = f['roi'][()]
    hits = zeros(shape,bool)
    hits[tuple(hits_coord)] = True
    if frame == -1:
        images = zeros(shape,dtype='int16')
        images[:,mask] = roi
        images = images.reshape(shape)
    else:
        hits = hits[frame]
        images = zeros(mask.shape,dtype='int16')
        images[mask] = roi[frame]
    #print('time to reconstruct images,hits [s]: {:0.3f}'.format(time()-t0))
    return images,hits

def roi_to_stats(root,terms):
    """Finds *.roi.hdf5 files that contain terms and writes a *.stats.hdf5
    file that is structured as a set of arrays including chunk, frame,
    row0, col0, counts, median, peak, and M, where M contains moments for hits
    that are also peaks. To qualify as a peak, the hit must be larger than
    the surrounding pixels, and the count difference between the peak and the
    median of the surrounding pixels must exceed 'threshold', which comes from
    the corresponding *.stats.pkl file."""
    from analysis_functions import find_pathnames,load_from_file
    from numpy import sort,nan,array,empty
    from numpy.random import random_sample
    from skimage.measure import moments
    from time import time
    from os import path
    import h5py
    median_footprint = array([[1,1,1],[1,0,1],[1,1,1]],dtype=bool)
    nan_matrix = empty((4,4))
    nan_matrix[:] = nan
    t0 = time()
    path_names = find_pathnames(root,terms)
    stats_name = path_names[0].replace('_0.roi.hdf5','.stats.hdf5')
    pkl_stats = path_names[0].replace('.roi.hdf5','.stats.pkl')
    if  path.exists(pkl_stats):
        stats = load_from_file(pkl_stats)
        threshold = stats['threshold']
        stats_median = stats['median']
    chunk = []
    frame = []
    row0 = []
    col0 = []
    counts = []
    median = []
    M = []
    peak = []
    sat = []
    for path_name in path_names:
        t1 = time()
        chunk_number = int((path_name.split('.roi.hdf5')[0].split('_')[-1]))
        with h5py.File(path_name,'r') as f:
            # Generate random numbers in shape of image (do it once)
            if path_name == path_names[0]:
                N,row,col = f['shape'][()]
                rnd = 0.1*random_sample((row,col))
            hits0 = f['hits0'][()]
            hits_coord = f['hits_coord'][()]
        images,hits = images_hits_reconstruct(path_name,frame=-1)
        # Step through hits0 to compute median, peak, and moments for peak
        frame_i = hits_coord[0]
        row0_i = hits_coord[1]
        col0_i = hits_coord[2]
        counts_i = []
        median_i = []
        M_i = []
        peak_i = []
        sat_i = []

        for i in range(hits0):
            r = row0_i[i]
            c = col0_i[i]
            f = frame_i[i]
            # Find if hit is a saturated pixel.
            counts_i += [images[f,r,c]]
            sat_i += [(counts_i[-1] + stats_median[r,c]) == 4094]
            # Construct 3x3 sub image with rnd offset
            sub3 = images[f,r-1:r+2,c-1:c+2] + rnd[r-1:r+2,c-1:c+2]
            # Omit stats calculations for hits found on image perimeter
            if sub3.size == 9:
                sub3_sort = sort(sub3[median_footprint])
                median_i += [int(sub3_sort[4])]
                peak_i += [(sub3[1,1] > sub3_sort[-1]) & (sub3[1,1]-median_i[-1] >= threshold[r,c])]
                # If a peak, calculate its moments
                if peak_i[-1]:
                    sub5 = images[f,r-2:r+3,c-2:c+3]
                    # Omit moments calculations for hits within 2 pixels of image perimeter
                    if sub5.size == 25:
                        M_i += [moments(sub5)]
                    else:
                       M_i += [nan_matrix]
                else:
                    M_i += [nan_matrix]
            else:
                median_i += [nan]
                peak_i += [nan]
                M_i += [nan_matrix]
        chunk += hits0*[chunk_number]
        frame += list(frame_i)
        row0 += list(row0_i)
        col0 += list(col0_i)
        counts += list(counts_i)
        median += list(median_i)
        M += list(M_i)
        peak += list(peak_i)
        sat += list(sat_i)

        print('\r Time to process {}: {:0.3f} seconds'.format(path_name.split(root)[1],time()-t1),end='\r')
    chunk = array(chunk)
    frame = array(frame)
    row0 = array(row0)
    col0 = array(col0)
    counts = array(counts)
    median = array(median)
    M = array(M)
    peak = array(peak).astype(bool)
    sat = array(sat).astype(bool)
    # Calculate running frame number from chunk and frame
    rfn = N*chunk+frame
    with h5py.File(stats_name, 'w') as f:
        f.create_dataset('shape', data = (N,row,col), dtype = 'uint32')
        f.create_dataset('rfn', data = rfn, dtype = 'uint32')
        f.create_dataset('chunk', data = chunk, dtype = 'uint32')
        f.create_dataset('frame', data = frame, dtype = 'uint16')
        f.create_dataset('row0', data = row0, dtype = 'uint16')
        f.create_dataset('col0', data = col0, dtype = 'uint16')
        f.create_dataset('counts', data = counts, dtype = 'int16')
        f.create_dataset('median', data = median, dtype = 'int16')
        f.create_dataset('M', data = M)
        f.create_dataset('peak', data = peak)
        f.create_dataset('sat', data = sat)
    print('\n Time to process dataset [s]: {:0.3f}  '.format(time()-t0))
    chart_stats(root,stats_name.split(root)[1])

def chart_stats(root,filename):
    """Reads *.stats.hdf5 file and charts relevant parameters. The spot
    intensity I_spot is the larger of the zeroth moment for the spot, or the
    difference between counts and median, which compensates for some cases
    where the zeroth moment is artifically lower because of a small background
    offset."""
    from charting_functions import chart_vector,chart_xy_symbol,chart_histogram,chart_image_mask,chart_image
    from numpy import maximum,isnan,zeros
    import h5py
    with h5py.File(root+filename,'r') as f:
        N,row,col = f['shape'][()]
        rfn = f['rfn'][()]
        row0 = f['row0'][()]
        col0 = f['col0'][()]
        counts = f['counts'][()]
        median = f['median'][()]
        M = f['M'][()]
        peak = f['peak'][()].astype(bool)
        sat = f['sat'][()].astype(bool)

    I_spot = maximum(counts-median,M[:,0,0])
    N_spots = (~isnan(I_spot)).sum()
    chart_xy_symbol(rfn,I_spot,'I_spot ({} spots)\n {}'.format(N_spots,filename),logy=True,x_label='running frame number')
    chart_histogram(I_spot,'I_spot \n {}'.format(filename),xmax=400)
    hits_image = zeros((row,col))
    hits_image[(row0,col0)] = counts
    peak_image = zeros((row,col))
    peak_image[(row0[peak],col0[peak])] = True
    chart_image_mask(hits_image,peak_image,'counts for hits; peak \n {}'.format(filename),vmax=100)
    sat_image = zeros((row,col))
    sat_image[(row0[sat],col0[sat])] = True
    chart_image_mask(hits_image,sat_image,'counts for hits; sat \n {}'.format(filename),vmax=100)

def spot_moments(I):
    """Computes moments for spot found in I, which is assumed to have zero
    background. Returns integrated counts, the x and y positions of the spot
    center, and the x and y sigmas."""
    #conda install -c conda-forge scikit-image
    from skimage.measure import moments
    from numpy import sqrt,arctan
    M = moments(I,order=3)
    mu00 = M[0,0]
    mup10 = M[1,0]/M[0,0]
    mup01 = M[0,1]/M[0,0]
    mup11 = M[1,1]/M[0,0] - M[1,0]*M[0,1]/M[0,0]**2
    mup20 = M[2,0]/M[0,0] - (M[1,0]/M[0,0])**2
    mup02 = M[0,2]/M[0,0] - (M[0,1]/M[0,0])**2
    mup30 = M[3,0]/M[0,0] - 3*M[1,0]*M[2,0]/M[0,0]**2 + 2*(M[1,0]/M[0,0])**3
    mup03 = M[0,3]/M[0,0] - 3*M[0,1]*M[0,2]/M[0,0]**2 + 2*(M[0,1]/M[0,0])**3
    integral = mu00
    angle = 0.5*arctan(2*mup11/(mup20-mup02))
    a = 0.5*(mup20+mup02)
    b = 0.5*sqrt(4*mup11**2 + (mup20-mup02)**2)
    sigma_major = sqrt(a+b)
    sigma_minor = sqrt(a-b)
    r = mup10
    c = mup01
    return integral,r,c,sigma_major,sigma_minor,angle


def chart_dataset(root,stats_name):
    from charting_functions import chart_vector,chart_xy_symbol,chart_histogram
    from numpy import array
    import h5py
    filename = stats_name.split(root)[1]
    with h5py.File(stats_name,'r') as f:
        counts = array(f['counts'])
        frame = array(f['frame'])
        chunk = array(f['chunk'])
    chart_vector(counts,'counts \n {}'.format(filename),x_label = 'hit number')
    rfn = 256*chunk+frame
    chart_xy_symbol(rfn,counts,'counts \n {}'.format(filename),logy=True,x_label = 'running frame number')
    chart_histogram(counts,'counts \n {}'.format(filename),xmax=500)
