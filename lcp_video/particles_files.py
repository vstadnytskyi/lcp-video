"""
The library of codes used to work with .particle.hdf5 file generate by scattering experiments

## Dataset related

## Image reconstruction

## XY Analysis

## Data Analysis

## Plotting function

"""
def create_new_dparticles(dpeaks, verbose = False):
    """
    """
    import numpy as np
    from lcp_video import peaks_files
    res = {}
    res['peaks filename'] = dpeaks['filename']
    res['ID'] = np.unique(dpeaks['Mp'])
    x, res['len(peak)'] = peaks_files.xy_particle_length_peaks(dpeaks)
    x, res['len(frame)'] = peaks_files.xy_particle_length_frames(dpeaks)

    res['speed_r'] = np.unique(dpeaks['Mp'])*np.nan
    res['speed_c'] = np.unique(dpeaks['Mp'])*np.nan
    res['scattering max'] = np.unique(dpeaks['Mp'])*np.nan

    return res

def examine_particles_file(filename, verbose = False):
    """
    checks the peaks file and returns information about its' content. This is useful to inspect datafile before loading data from it.

    In [561]: dparticles.keys()
Out[561]: dict_keys(['IDs', 'frames', 'peaks', 'peaks filename', 'scattering value', 'scattering error'])
    """
    from h5py import File
    import os
    res = {}
    with File(filename,'r') as f:
        res['peaks filename'] = f['peaks filename'][()]
        res['N of particles'] = f['scattering value'].shape

    return res

def read_particle_file(filename, particle_ids = None, verbose = False):
    """
    reads the particles file and returns a dictionary with different entries. input parameters like max_frame_ids and max_peak_ids can be used to select a range of data from the file.

    """
    from h5py import File
    res = {}
    from numpy import where

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
    return res

def write_particle_file(filename, dparticles, rewrite = False):
    """
    writes or appends an existing particles files
    """
    from h5py import File
    import os
    is_exist = os.path.exists(filename)
    if rewrite and is_exist:
        os.remove(filename)
    with File(filename,'w') as f:
        for key in list(dparticles.keys()):
            f.create_dataset(key, data = dparticles[key])

## IMAGE reconstruction

## XY analysis

## Analysis of Dataset

def update_speed_very_slow_particle(dpeaks,dparticles):
    """
    """

    from lcp_video import peaks_files

    very_slow_particles = peaks_files.get_slow_particles(dpeaks)
    for particle in list(very_slow_particles):
        select = dparticles['ID'] == particle
        speed_r,speed_c = peaks_files.get_speed(dpeaks,particle)
        dparticles['speed_r'][select] = speed_r
        dparticles['speed_c'][select] = speed_c

    return dparticles

def update_s_coeff_very_slow_particle(dpeaks,dparticles):
    """
    """
    from lcp_video import peaks_files
    import numpy as np
    very_slow_particles = peaks_files.get_slow_particles(dpeaks)
    for particle in list(very_slow_particles):
        select = dpeaks['Mp'] == particle
        x = dpeaks['coordinate'][select]
        y = dpeaks['M0'][select] *dpeaks['S coeff'][select]
        z = np.polyfit(x,y,deg = 2)
        p = np.poly1d(z)

        x_new = np.linspace(x[0], x[-1], len(x)*100)
        y_new = p(x_new)
        max_value = y_new.max()
        dparticles['scattering max'][particle] = max_value

    return dparticles
## Plotting functions

def plot_particle_speed(dparticles):

    from matplotlib import pyplot as plt

    fig = plt.figure(figsize=plt.figaspect(0.5))
    grid = plt.GridSpec(2, 1, hspace=0.0, wspace=0.025)
    ax1 = fig.add_subplot(grid[0,0])
    plot1 = ax1.plot(dparticles['speed_c'],'o')
    ax1.axhline(0,color = 'darkred')
    ax1.set_ylabel('horizontal velocity')
    ax1.grid()

    ax2 = fig.add_subplot(grid[1,0], sharex = ax1)
    plot2 = ax2.plot(dparticles['speed_r'],'o')
    ax2.axhline(0,color = 'darkred')
    ax2.set_ylabel('falling velocity')
    ax2.grid()
