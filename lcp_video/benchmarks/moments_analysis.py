import numpy as np
from matplotlib import pyplot as plt

###
###
###
###

###
### Create test image
###
shape = (31,31)

mean = np.ones((shape))*15
std = np.ones((shape))*1
noise = np.random.normal(loc = 0, scale = std).astype('int16')

import numpy as np

# define normalized 2D gaussian
def gauss2d(x=0, y=0, a = 10, mx=0, my=0, sx=1, sy=1):
    z = np.exp(-((x - mx)**2. / (2. * sx**2.) + (y - my)**2. / (2. * sy**2.)))
    return (z*a/z.max()).astype('int64')

x = np.linspace(0, shape[0],shape[0])
y = np.linspace(0, shape[1],shape[1])
x, y = np.meshgrid(x, y) # get 2D variables instead of 1D
gauss_img = gauss2d(x=x, y=y, a = 10, mx = 15, my = 15, sx = 2, sy = 4 )

img = (gauss_img + noise).astype('int16')

plt.figure()
plt.imshow(img)

from lcp_video import analysis
m = analysis.get_moments_simple(img)
print(m)

m_vec = {}
for key in list(m.keys()):
    m_vec[key] = np.zeros((int(shape[0]*shape[1],)))*np.nan

for i in range(int(shape[0]*shape[1])):
    noise1 = np.random.normal(loc = mean, scale = std).astype('int16')
    noise2 = np.random.normal(loc = mean, scale = std).astype('int16')
    try:
        m = analysis.get_moments_simple(img-mean+noise1-noise2)
        for key in list(m.keys()):
            m_vec[key][i] = m[key]
    except:
        print(i)
