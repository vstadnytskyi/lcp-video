"""
Plotting library
"""


def get_and_plot_frame(filename, N):
    """
    """
    from cv2 import VideoCapture
    from lcp_video import video
    frame = video.get_frame(filename,video)

def plot_range(vec_temp, N_avg = 24):
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
