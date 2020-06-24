"""
Plotting library
"""
def plot_framea_as_RGB(frames = (None,None,None)):
    """
    """
    from matplotlib import pyplot as plt
    from numpy import zeros
    img = zeros((3000,4096,3),dtype = 'uint8')
    for i in range(3):
        img[:,:,i] = frames[i]
    fig = plt.figure(figsize=(4, 4))
    grid = plt.GridSpec(1, 1, hspace=0.025, wspace=0.025)
    ax1 = fig.add_subplot(grid[0,0])
    ax1.imshow(img)


def plot_images(lst = [], vmax = [1,20], vmin = [0,15], titles = ['1','2']):
    from matplotlib import pyplot as plt
    fig = plt.figure(figsize=(7, 3))
    grid = plt.GridSpec(1, len(lst), hspace=0.025, wspace=0.025)
    ax = []
    ax.append(fig.add_subplot(grid[0,0]))
    ax[0].imshow(lst[0],vmax = vmax[0], vmin = vmin[0])
    ax[0].set_title(titles[0])
    for i in range(1,len(lst)):
        ax.append(fig.add_subplot(grid[0,i], sharex = ax[0], sharey = ax[0]))
        ax[i].imshow(lst[i],vmax = vmax[i], vmin = vmin[i])
        ax[i].set_title(titles[i])


def plot_histogram(arr, start, end, step, yscale = 'log',xscale = 'log'):
    """
    """
    from numpy import zeros,where
    length = int((end-start)/step) -1
    bins_r = zeros((length,))
    bins_l = zeros((length,))
    bins_x = zeros((length,))
    bins_y = zeros((length,))
    for i in range(length):
        bins_l[i] = start + step*i
        bins_r[i] = start + step*(i+1)
        bins_x[i] = start + step*(i+1/2)
    less_x = bins_l[0] - step/2
    more_x = bins_r[-1] + step/2
    less_y = 0
    more_y = 0

    for i in range(length):
        bins_y[i] = ((arr<=bins_r[i]) * (arr>=bins_l[i])).sum()
    less_y = (arr<=less_x).sum()
    more_y = (arr>=more_x).sum()

    fig = plt.figure(figsize=(7, 3))
    grid = plt.GridSpec(1, 2, hspace=0.025, wspace=0.025)
    ax1 = fig.add_subplot(grid[0,0])

    ax1.plot(bins_x,bins_y,'-', linewidth = 1, color = 'darkgreen')

    ax1.plot(less_x,less_y,'o', color = 'darkred')
    ax1.plot(more_x,more_y,'o', color = 'darkred')
    ax1.set_yscale(yscale)
    ax1.set_xscale(xscale)


def get_and_plot_frame(filename, N):
    """
    """
    from cv2 import VideoCapture
    from lcp_video import video
    frame = video.get_frame(filename,video)

def plot_range(vec_temp, N_avg = 24):
    """
    """
    def moving_average(a, n=3) :
        from numpy import cumsum
        ret = cumsum(a, dtype=float)
        ret[n:] = ret[n:] - ret[:-n]
        return ret[n - 1:] / n
    from matplotlib import pyplot as plt
    plt.figure()
    lst = list(vec_temp.keys())[::-1]
    print(f'range: a, tau, offset')
    for key in lst:
        x = moving_average(vec_temp[key][:,0],N_avg)
        y = moving_average(vec_temp[key][:,1],N_avg)
        plt.plot(x,y,'-', linewidth = 1, label = key+' counts');

        from scipy.optimize import curve_fit
        from numpy import exp
        t0 = 20000
        xf = x[t0:] - x[t0]
        yf = y[t0:]
        coeff = curve_fit(lambda t,a,tau, offset: a*exp(-(t)/tau) +offset,  xf,  yf, p0 = [5,12.4,0.5])
        a = coeff[0][0]
        tau = coeff[0][1]
        offset = coeff[0][2]
        plt.plot(x[t0:],a*exp(-xf/tau) + offset,'-', linewidth = 2, color = 'darkred')
        print(f'{key}: {round(a,2)}, {round(tau/(24*60),2)}, {round(offset,2)}')

    plt.xlabel('frame number')
    plt.ylabel('count per frame')
    plt.title('PNAS paper data: count/frame vs frame for several size ranges')
    plt.legend()
    plt.show()

def plot_timestamp_difference(camera_lst, exposure_time):
    length = vec.shape[0]
    from numpy import arange
    vec0 = arange(0,length,1)*exposure_time-vec[0]


    fig = plt.figure(figsize=(7, 3))
    grid = plt.GridSpec(3, 1, hspace=0.025, wspace=0.025)
    ax1 = fig.add_subplot(grid[0,0])

    ax1.plot(bins_x,bins_y,'-', linewidth = 1, color = 'darkgreen')

    ax1.plot(less_x,less_y,'o', color = 'darkred')
    ax1.plot(more_x,more_y,'o', color = 'darkred')
    ax1.set_yscale(yscale)
    ax1.set_xscale(xscale)

def plot_image_and_projection(image, vrow = None, vcol = None):
    """

    """
    from matplotlib import pyplot as plt
    from numpy import arange
    fig = plt.figure(figsize=(4, 4))
    grid = plt.GridSpec(3, 3, hspace=0.025, wspace=0.025)

    ax1 = fig.add_subplot(grid[0:2,0:2])
    ax1.imshow(image)
    if vrow is None:
        vrow = image.sum(axis = 1)

    y = arange(0,vrow.shape[0])
    axv = fig.add_subplot(grid[0:2,2], sharey = ax1)
    axv.plot(vrow,y)

    axh = fig.add_subplot(grid[2,0:2], sharex = ax1 )
    if vcol is None:
        vcol = image.sum(axis = 0)
    x = arange(0,vcol.shape[0])
    axh.plot(x,vcol)
