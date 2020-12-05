"""
The library of codes used to work with .peaks.hdf5 file generate by scattering experiments

## Dataset related
- examine_peaks_file - examine PEAKS file and returns short summary of relavant information in it.
- read_peaks_file - read peaks file and returns dictionary with the file content



## Image reconstruction
- reconstruct_image_from_frames - creates an image from a list of frames
- image_from_selects - creates an image with given list of selectors.



## XY Analysis
- xy_peaks_vs_frame - returns xy of peaks per frame
- xy_particle_length - return particle length vs particle number


## Data Analysis
- get_eigencoordinate - for every particle assigns eigencoordinate to each peak based on time vector.

## Plotting function
- report_particle - plots a full report on a given particle

## Peaks to Particles


"""
def report_peaks_dataset(dpeaks):
    """
    examines the peaks datasets and report on several aspects.

    - number of assigned peaks
    - number of assigned peaks with d0==0
    - number of unassigned peaks
    - distance is 4 but numbers are odd
    - number of peaks with id -1

    """
    result = {}
    result['number of assigned peaks'] = (dpeaks['Mp']>=0).sum()

    result['number of assigned peaks with d0==0'] = ((dpeaks['Mp']>0) * (dpeaks['d0']==0)).sum()

    result['number of unassigned peaks'] = (dpeaks['Mp']==0).sum()

    result['number of peaks -1'] = (dpeaks['Mp']<0).sum()

    return result

def examine_peaks_file(filename, verbose = False):
    """
    checks the peaks file and returns information about its' content. This is useful to inspect datafile before loading data from it.
    """
    from h5py import File
    import os
    res = {}
    with File(filename,'r') as f:
        res['keys'] = list(f.keys())
        res['frames'] = (f['frame'][0],f['frame'][-1])
        res['particles'] = (f['Mp'][0],f['Mp'][-1])
        res['shape'] = f['shape'][()]
        res['N of peaks'] = f['col'].shape
        res['Memory on the drive, MB'] = round(os.path.getsize(filename)/8/1024/1024,3)
        res['N of peaks assigned to particles'] = (dpeaks_breathing['Mp']>0).sum()
        res['N of peaks with no nearby neibhours'] = (dpeaks_breathing['d0'][dpeaks_breathing['Mp']>0]>5).sum()
    return res

def read_peaks_file(filename, frame_ids = None, verbose = False):
    """
    reads the peaks file and returns a dictionary with different entries. input parameters like max_frame_ids and max_peak_ids can be used to select a range of data from the file.

    """
    from h5py import File
    res = {}
    from numpy import where
    import numpy as np

    with File(filename,'r') as f:
        selector = f['frame'][()] > -1
        if frame_ids is not None:
            selector *= f['frame'][()] >= frame_ids[0]
            selector *= f['frame'][()] <= frame_ids[1]
        (start,end) = (where(selector)[0][0],where(selector)[0][-1])
        for key in list(f.keys()):
            if verbose:
                from time import time, ctime
                print(f'{ctime(time())}: reading key {key}')
            if key not in ['frameID0']:
                res[key] = f[key][start:end+1]
            else:
                res[key] = f[key][()]
        if 'S coeff' not in res.keys():
            res['S coeff'] = f['M0'][()]*np.nan
        if 'coordinate' not in res.keys():
            res['coordinate'] = f['M0'][()]*np.nan
        res['filename'] = filename
    return res


## IMAGE reconstruction

def reconstruct_image_from_frames(dpeaks,frames):
    """
    creates a frame from dpeaks dictionary and list of frames.
    frames is a list of frames.
    """
    from scipy import stats
    from numpy import nonzero, zeros,nan, ones, argwhere, mean, nanmean, maximum

    #create empty selector which can be populated later.
    select = dpeaks['frame'] == -100
    for frame in frames:
        select |= dpeaks['frame'] == frame

    roi_size = 21

    height = dpeaks['shape'][1]
    width = dpeaks['shape'][2]

    image = zeros((height, width))

    for i in range(0,len(dpeaks['col'][select])):
        r = dpeaks['row'][select][i]
        c = dpeaks['col'][select][i]
        image[r-10:r+11,c-10:c+11] = maximum(image[r-10:r+11,c-10:c+11], dpeaks['roi'][select][i])

    image_fit = zeros((height, width))
    if 'img_fit' in list(dpeaks.keys()):
        for i in range(0,len(dpeaks['col'][select])):
            r = dpeaks['row'][select][i]
            c = dpeaks['col'][select][i]
            image_fit[r-10:r+11,c-10:c+11] = maximum(image_fit[r-10:r+11,c-10:c+11], dpeaks['img_fit'][select][i])

    return image, image_fit

def image_from_selects(dpeaks,selects):
    """
    creates a frame from dpeaks dictionary and list of selectors. The list of selectors can be used to encode different particles on the image.
    """
    from scipy import stats
    from numpy import nonzero, zeros,nan, ones, argwhere, mean, nanmean, maximum
    import numpy as np

    #create empty selector which can be populated later.
    if type(selects) != list:
        return None
    roi_size = 21

    height = dpeaks['shape'][1]
    width = dpeaks['shape'][2]
    X,Y,Z = [],[],[]
    image = zeros((height, width))
    for select in selects:
        for i in range(0,len(dpeaks['col'][select])):
            r = dpeaks['row'][select][i]
            c = dpeaks['col'][select][i]
            image[r-10:r+11,c-10:c+11] = maximum(image[r-10:r+11,c-10:c+11], dpeaks['roi'][select][i])

        image_fit = zeros((height, width))
        if 'img_fit' in list(dpeaks.keys()):
            for i in range(0,len(dpeaks['col'][select])):
                r = dpeaks['row'][select][i]
                c = dpeaks['col'][select][i]
                image_fit[r-10:r+11,c-10:c+11] = maximum(image_fit[r-10:r+11,c-10:c+11], dpeaks['img_fit'][select][i])

        X.append(dpeaks['col'][select])
        Y.append(dpeaks['frame'][select])
        Z.append(dpeaks['row'][select])

    return image, image_fit, np.array(X).astype('float64'), np.array(Y).astype('float64'), np.array(Z).astype('float64')

def update_dpeaks_s_coeff_slow_moving(dpeaks,particle):
    """
    get selector for a given particle
    """
    THREASHOLD = 6
    ROI_EXTRA = 20

    import numpy as np
    from lcp_video.analysis import grow_mask
    from lcp_video import peaks_files
    selector = dpeaks['Mp'] == particle
    frames = dpeaks['frame'][selector]
    M0 = dpeaks['M0'][selector]
    (unique,count) = np.unique(frames, return_counts=True)
    intensity = unique*0.0
    coeff = unique*0.0
    s_coeff = frames*0.0

    for frame in list(frames):
        index = list(unique).index(frame)
        tselector = selector*(dpeaks['frame']==frame)
        image, image_fit, X, Y, Z = peaks_files.image_from_selects(dpeaks,[tselector])
        img = image[int(np.min(Z)-ROI_EXTRA):int(np.max(Z)+ROI_EXTRA),int(np.min(X)-ROI_EXTRA):int(np.max(X)+ROI_EXTRA)]
        mask = grow_mask(img > THREASHOLD,1)
        intensity[index] = (img*mask).sum()
        coeff[index] = (count[index])*(1/51.2)

    for i in range(len(frames)):
        index = list(unique).index(list(frames)[i])
        sss = frames == frames[i]
        s_coeff[i] = coeff[index]*intensity[index]/M0[sss].sum()

    return s_coeff

def get_dpeaks_s_coeff_d0_zero(dpeaks):
    """
    get selector for a given particle

    """
    THREASHOLD = 6
    ROI_EXTRA = 20

    import numpy as np
    from lcp_video.analysis import grow_mask
    from lcp_video import peaks_files
    select = (dpeaks['d0'] == 0)*(dpeaks['Mp'] > 0)

    peak_ids = np.where(select)
    intensity = np.where(select)[0]*np.nan
    peaks_list = list(peak_ids[0])
    for i in range(len(peaks_list)):
        roi = dpeaks['roi'][peaks_list[i]]
        mask = grow_mask(roi > THREASHOLD,1)
        intensity[i] = (roi*mask).sum()


    frames = dpeaks['frame'][selector]
    M0 = dpeaks['M0'][selector]
    (unique,count) = np.unique(frames, return_counts=True)
    intensity = unique*0.0
    coeff = unique*0.0
    s_coeff = frames*0.0

    for frame in list(frames):
        index = list(unique).index(frame)
        tselector = selector*(dpeaks['frame']==frame)
        image, image_fit, X, Y, Z = peaks_files.image_from_selects(dpeaks,[tselector])
        img = image[int(np.min(Z)-ROI_EXTRA):int(np.max(Z)+ROI_EXTRA),int(np.min(X)-ROI_EXTRA):int(np.max(X)+ROI_EXTRA)]
        mask = grow_mask(img > THREASHOLD,1)
        intensity[index] = (img*mask).sum()
        coeff[index] = (count[index])*(1/51.2)

    for i in range(len(frames)):
        index = list(unique).index(list(frames)[i])
        sss = frames == frames[i]
        s_coeff[i] = coeff[index]*intensity[index]/M0[sss].sum()

    return s_coeff

## XY analysis

def xy_peaks_vs_frame(dpeaks, select = None, verbose = False):
    """
    return x, y set where
    x - frame index
    y - number of peaks per frame
    """
    if select is None:
        #this simply selects all frames
        if verbose:
            print('the select is None, hence selecting all peaks')
        select = dpeaks['frame'] >= 0

    import numpy as np
    frame_max = np.nanmax(dpeaks['frame'])
    x = np.arange(0,frame_max,1)
    y = np.copy(x)*0
    for frame in range(frame_max):
        temp_select = select*(dpeaks['frame'] == frame)
        y[frame] = temp_select.sum()

    return x,y

def xy_particle_length_frames(dpeaks, select = None, verbose = False):
    """
    return x, y set where
    x - frame index
    y - number of peaks per frame
    """
    if select is None:
        #this simply selects all frames
        if verbose:
            print('the select is None, hence selecting all peaks')
        select = dpeaks['frame'] >= 0

    import numpy as np
    particle_max = np.nanmax(dpeaks['Mp'])
    x = np.arange(0,particle_max,1)
    y = np.copy(x)*0
    for particle in range(particle_max):
        temp_select = select*(dpeaks['Mp'] == particle)
        idx = np.unique(dpeaks['frame'][temp_select])
        y[particle] = idx.shape[0]

    return x,y

def xy_particle_length_peaks(dpeaks, select = None, verbose = False):
    """
    return x, y set where
    x - frame index
    y - number of peaks per frame
    """
    if select is None:
        #this simply selects all frames
        select = dpeaks['frame'] >= 0

    import numpy as np
    particle_max = np.nanmax(dpeaks['Mp'])
    x = np.arange(0,particle_max,1)
    y = np.copy(x)*0
    for particle in range(particle_max):
        temp_select = select*(dpeaks['Mp'] == particle)
        y[particle] = temp_select.sum()

    return x,y

## Analysis of Dataset

def normalize_slow_particles(dpeaks, particle_list = None):
    from lcp_video import peaks_files
    for particle in list(particle_list):
        dpeaks['S coeff'][dpeaks['Mp'] == particle] = peaks_files.update_dpeaks_s_coeff_slow_moving(dpeaks,particle)
    return dpeaks

def get_eigencoordinate(dpeaks, particle):
    """
    Np - time index
    M0 - intensity of a peaks
    row - row position
    """
    import numpy as np
    select = dpeaks['Mp'] == particle
    index = np.where(select)[0]
    dist = index*0.0
    sorted = np.argsort(dpeaks['Np'][select])
    dist[0] = 0
    for i in range(len(dpeaks['Np'][select][sorted])-1):
        r_i1 = dpeaks['r0'][select][sorted][i]
        r_i2 = dpeaks['r0'][select][sorted][i+1]
        c_i1 = dpeaks['c0'][select][sorted][i]
        c_i2 = dpeaks['c0'][select][sorted][i+1]
        dist[i+1] = dist[i]+((r_i1-r_i2)**2 + (c_i1-c_i2)**2)**0.5
    return dist, index[sorted]

def get_local_speed(dpeaks,select):
    """
    """
    import numpy as np
    index = np.where(select)[0]
    sorted = np.argsort(dpeaks['Np'][index])

    speed = np.diff(dpeaks['coordinate'][index[sorted]])/np.diff(dpeaks['Np'][index[sorted]])

    return np.round(speed,3)

def get_speed(dpeaks, particle):
    """
    """
    import numpy as np
    select_particle = dpeaks['Mp'] == particle
    full_frames = np.unique(dpeaks['frame'][select_particle])[1:-1]
    select_frame = dpeaks['Mp'] <0
    for frame in list(full_frames):
        select_frame+=(dpeaks['frame'] == frame)
    indices = np.argsort(dpeaks['Np'][select_frame*select_particle])
    r_s = dpeaks['r0'][select_frame*select_particle][indices][0]
    r_e = dpeaks['r0'][select_frame*select_particle][indices][-1]
    c_s = dpeaks['c0'][select_frame*select_particle][indices][0]
    c_e = dpeaks['c0'][select_frame*select_particle][indices][-1]
    len_r = r_e-r_s
    len_c = c_e-c_s
    speed_r = len_r/len(full_frames)
    speed_c = len_c/len(full_frames)

    return speed_r, speed_c

def get_peaks_per_frame(dpeaks,select):
    import numpy as np
    lst_frames = list(np.unique(dpeaks['frame'][select]))
    lst_peaks = np.unique(dpeaks['frame'][select])*0
    for i in range(len(lst_frames)):
        lst_peaks[i] = np.sum(select*(dpeaks['frame'] == lst_frames[i]))
    return np.array(lst_peaks)

## Plotting functions

def report_particle(dpeaks, particle = 0):

    ROI_EXTRA = 20
    from matplotlib import pyplot as plt
    import numpy as np
    from lcp_video import peaks_files

    select = dpeaks['Mp'] == particle

    image, image_fit, X, Y, Z = peaks_files.image_from_selects(dpeaks,[select])
    select = dpeaks['Mp'] == particle

    indices = np.argsort(dpeaks['Np'][select])


    plot_image_2d(dpeaks,particle, markers = 'frame')

    plot_M0_vs_Np(dpeaks,particle)

    plt.figure()
    plt.plot(dpeaks['coordinate'][select][indices],dpeaks['M0'][select][indices],'-o', label = 'not scaled')
    plt.plot(dpeaks['coordinate'][select][indices],dpeaks['M0'][select][indices]*dpeaks['S coeff'][select][indices],'-o', label = 'scaled')
    plt.title(f'M0 vs eigencoordinate particle# {particle}')
    plt.legend()
    plt.grid()

    plot_image_3d(dpeaks,particle)

    peaks_per_frame = peaks_files.get_peaks_per_frame(dpeaks,select)

    index = np.where(select)[0]
    sorted = np.argsort(dpeaks['Np'][index])
    print(f"rows: {np.min(Z),np.max(Z)}, cols: {np.min(X),np.max(X)}")
    print(f"Np = {dpeaks['Np'][index[sorted]]}")
    print(f"frames = {np.unique(dpeaks['frame'][index[sorted]])}")

    print(f"peaks per frame = {peaks_per_frame}")
    print(f"distance pixels = {np.diff(dpeaks['coordinate'][index[sorted]])}")
    print(f"distance times = {np.diff(dpeaks['Np'][index[sorted]])}")
    speed = peaks_files.get_local_speed(dpeaks,select)
    print(f'speed pixels per tick: {speed}')
    print(f"M0 = {dpeaks['M0'][index[sorted]]}")
    print(f"d0 = {dpeaks['d0'][index[sorted]]}")
    print(f"number of peaks = {dpeaks['M0'][index[sorted]].shape[0]}")


def plot_image_2d(dpeaks, particle, markers = None, verbose = False):
    ROI_EXTRA = 20
    from matplotlib import pyplot as plt
    import numpy as np
    from lcp_video import peaks_files

    selector = dpeaks['Mp'] == particle

    image, image_fit, X, Y, Z = peaks_files.image_from_selects(dpeaks,[selector])
    fig = plt.figure(figsize=plt.figaspect(0.5))
    grid = plt.GridSpec(1, 1, hspace=0.025, wspace=0.025)
    ax1 = fig.add_subplot(grid[0,0])
    imshow = ax1.imshow(image[int(np.maximum(np.min(Z-ROI_EXTRA),0)):int(np.max(Z)+ROI_EXTRA),int(np.maximum(np.min(X-ROI_EXTRA),0)):int(np.max(X)+ROI_EXTRA)])
    fig.colorbar(imshow, shrink=0.5, aspect=10, orientation='vertical')
    if markers is 'dot':
        ax1.scatter(X-(np.min(X)-ROI_EXTRA),Z-(np.min(Z)-ROI_EXTRA), s = 10, color = 'red')
    elif markers is 'frame':
        lst_x = ((X - np.maximum(np.min(X-ROI_EXTRA),0))[0])
        lst_y = ((Z - np.maximum(np.min(Z-ROI_EXTRA),0))[0])
        lst_z = list(dpeaks['frame'][selector])
        for i in range(len(lst_z)):
            ax1.scatter(lst_x[i],lst_y[i], s = 500, marker =f'${lst_z[i]}$', color = 'red')
        #plt.scatter(lst_x[0],lst_y[0],lst_z[0], s =10, marker =f'${lst_z[0]}$', color = 'red')

    #ax.set_xlabel([80,122])
    fig.suptitle(f'particle# {particle}')

def plot_image_3d(dpeaks, particle):
    ROI_EXTRA = 20

    from lcp_video import peaks_files

    import matplotlib.pyplot as plt
    from matplotlib import cm
    import numpy as np

    from mpl_toolkits.mplot3d.axes3d import get_test_data
    # This import registers the 3D projection, but is otherwise unused.
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

    selector = dpeaks['Mp'] == particle

    fig = plt.figure(figsize=plt.figaspect(0.5))
    grid = plt.GridSpec(1, 1, hspace=0.025, wspace=0.025)
    #===============
    #  First subplot
    #===============
    # set up the axes for the first plot
    ax1 = fig.add_subplot(grid[0,0], projection='3d')

    image, image_fit, X, Y, Z = peaks_files.image_from_selects(dpeaks,[selector])
    # plt.figure()
    # plt.imshow(image[np.min(Z)-ROI_EXTRA:np.max(Z)+ROI_EXTRA,np.min(X)-ROI_EXTRA:np.max(X)+ROI_EXTRA])
    # plt.colorbar()
    # plt.scatter(X-(np.min(X)-ROI_EXTRA),Z-(np.min(Z)-ROI_EXTRA), s = 10, color = 'red')
    #
    local_image = image[int(np.maximum(np.min(Z-ROI_EXTRA),0)):int(np.max(Z+ROI_EXTRA)),int(np.maximum(np.min(X-ROI_EXTRA),0)):int(np.max(X+ROI_EXTRA))]

    x = np.arange(int(np.maximum(np.min(Z-ROI_EXTRA),0)), np.max(Z)+ROI_EXTRA, 1)
    y = np.arange(int(np.maximum(np.min(X-ROI_EXTRA),0)), np.max(X)+ROI_EXTRA, 1)
    x, y = np.meshgrid(x, y)

    surf = ax1.plot_surface(x, y, local_image.T, rstride=1, cstride=1, cmap=cm.coolwarm,linewidth=5, antialiased=False)
    fig.colorbar(surf, shrink=0.5, aspect=10)
    fig.suptitle(f'particle# {particle}')

    plt.show()

def plot_M0_vs_Np(dpeaks, particle):

    ROI_EXTRA = 20
    from matplotlib import pyplot as plt
    import numpy as np
    from lcp_video import peaks_files

    select = dpeaks['Mp'] == particle

    indices = np.argsort(dpeaks['Np'][select])
    # dpeaks['coordinate'][sorted] = dist

    plt.figure()
    plt.plot(dpeaks['Np'][select][indices],dpeaks['M0'][select][indices],'-o', label = 'not scaled')
    plt.plot(dpeaks['Np'][select][indices],dpeaks['M0'][select][indices]*dpeaks['S coeff'][select][indices],'-o', label = 'scaled')
    plt.title(f'M0 vs Np for particle# {particle}')
    plt.legend()
    plt.grid()

def plot_frames_per_particle(dpeaks):
    """
    """
    from lcp_video import peaks_files
    from matplotlib import pyplot as plt
    import numpy as np
    x,y = peaks_files.xy_particle_length_frames(dpeaks)
    plt.figure()
    plt.plot(x[1:],y[1:])
    plt.show()
    print(f'number of zeros {y[0]}')
    print(f'longest particle ID {np.argmax(y[1:])+1} with {np.max(y[1:])}')

## Peaks to Particles

def get_number_of_frames_per_particle(dpeaks,select = None):
    """
    return number of frames per each particle.

    ToDO: add selector and allow extraction only a part of the data
    """
    import numpy as np
    particle_max = dpeaks['Mp'].max()
    length_frames = []
    particle_ids = []
    for particle in range(particle_max):
        particle_ids.append(particle)
        length_frames.append(len(np.unqiue(dpeaks['frame'][dpeaks['Mp']==particle])))
    return np.array(particle_ids),np.array(length_frames)


def get_number_of_peaks_per_particle(dpeaks,select = None):
    """
    return number of peaks per each particle.

    ToDO: add selector and allow extraction only a part of the data
    """
    import numpy as np
    #if select is None:
    #    select = dpeaks['Mp'] >= 0

    particle_max = dpeaks['Mp'][select].max()
    length_peaks = []
    particle_ids = []
    for particle in range(particle_max):
        particle_ids.append(particle)
        length_peaks.append(len(dpeaks['frame'][dpeaks['Mp']==particle]))
    return np.array(particle_ids),np.array(length_peaks)


### Procedures

def normalize_no_neibhours(dpeaks):
    """
    """
    import numpy as np
    select = np.isnan(dpeaks['S coeff']) * (dpeaks['d0'] > 5)
    indices = np.where(select)[0]

    for peak in list(indices):
        if dpeaks['d0'][peak] > 5:
            if dpeaks['Np'][peak]%2 == 0:
                dpeaks['S coeff'][peak] = 1
            else:
                dpeaks['S coeff'][peak] = 1/0.6
            dpeaks['norm S'][peak] = 1
    return dpeaks

def update_very_slow_moving_particle(dpeaks, verbose = False):
    from lcp_video import peaks_files
    from time import ctime, time
    very_slow_particles = peaks_files.get_slow_particles(dpeaks)
    for particle in list(very_slow_particles):
        if verbose:
            print(ctime(time()),particle)
        dpeaks['S coeff'][dpeaks['Mp'] == particle] = peaks_files.update_dpeaks_s_coeff_slow_moving(dpeaks,particle)
    return dpeaks


def update_eigencoordinate(dpeaks):
    import numpy as np
    max_particle = dpeaks['Mp'].max()
    for particle in range(1,max_particle+1):
        dist, index = get_eigencoordinate(dpeaks, particle)
        dpeaks['coordinate'][index] = np.round(dist,3)
    return dpeaks

def get_slow_particles(dpeaks):
    """
    """
    from lcp_video import peaks_files
    import numpy as np
    x, y = peaks_files.xy_particle_length_frames(dpeaks)
    very_slow_particles = np.where(y[1:]>3)[0] + 1
    return very_slow_particles
