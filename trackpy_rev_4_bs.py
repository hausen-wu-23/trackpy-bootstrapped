from __future__ import division, unicode_literals, print_function
from email.mime import image 
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os
import progressbar
import warnings
warnings.filterwarnings("ignore")

# Optionally, tweak styles.
mpl.rc('figure',  figsize=(10, 5))
mpl.rc('image', cmap='gray')


import numpy as np
import pandas as pd

import pims
import trackpy as tp

from PIL import Image

def func(x, a, b, c):
    return a*x**b + c

# def func2(x, a, b):
#     return a*x**b 

# def func3(x, a, c):
#     return a*x**alpha + c

fps = 50
pixel_per_micron = 10.72
file_dir = '/Volumes/AndersonLab/Hausen W/Data/2022_07_08_1um_nyo_beads_14dex/for tracking/dex_22_MMStack_Pos0.ome.tif'
work_dir = '/'.join(file_dir.split('/')[:-1]) + '/'
print(work_dir)
frames = pims.open(file_dir)

img = Image.open(file_dir)
n_frame1 = img.n_frames
print(n_frame1)

plt.imshow(frames[0])
plt.show()

inv = True if input('Invert Image (Y/N)').lower() == 'y' else False

f = tp.locate(frames[0], 11, invert=inv)

tp.annotate(f, frames[0])

fig, ax = plt.subplots()
ax.hist(f['mass'], bins=100)

# Optionally, label the axes.
ax.set(xlabel='mass', ylabel='count');

plt.show()


par_d = 15
test_f = 0


go = 'n'
while go != 'y':
    min_mass = int(input('Enter your minmass: '))

    f = tp.locate(frames[test_f], par_d, invert=inv, minmass=min_mass)

    tp.annotate(f, frames[test_f])

    go = input('Go on? (Y/N)').lower()




ax  = tp.annotate(f, frames[test_f])
#len(f.x)  # give the number of the fitures = size of x in f

ax.set_xticks([])
ax.set_yticks([])
ax.set_xlabel("x", fontweight ='bold' , fontsize = 14)
ax.set_ylabel('y', fontweight ='bold', fontsize = 14)

tp.subpx_bias(f)


tp.quiet(suppress=False) # Turn on progress reports
f = tp.batch(frames[:], par_d, minmass=min_mass, invert=inv, processes=1);



tp.quiet() # Turn off progress reports for best performance (same as tp.quiet(suppress=True))
#tp.quiet(suppress=False) # Turn on progress reports
t = tp.link_df(f, 7, memory=10)



plt.figure()
ax = plt.gca() #you first need to get the axis handle
ax.set_aspect(1) #sets the height to width ratio. 
tp.plot_traj(t)


t1 = tp.filter_stubs(t, 50)
# Compare the number of particles in the unfiltered and filtered data.
print('Before:', t['particle'].nunique())
print('After:', t1['particle'].nunique())


plt.figure()
tp.mass_size(t1.groupby('particle').mean()); # convenience function -- just plots size vs. mass


t2 = t1[((t1['size'] < 20) &
         (t1['ecc'] < .6))]



plt.figure()
tp.annotate(t2[t2['frame'] == 0], frames[0]);


plt.figure()
ax = plt.gca() #you first need to get the axis handle
ax.set_aspect(1) #sets the height to width ratio. 
tp.plot_traj(t2)

lod = []
loa = []

widgets = [' [',
         progressbar.Timer(format= 'elapsed time: %(elapsed)s'),
         '] ',
           progressbar.Bar('*'),' (',
           progressbar.ETA(), ') ',
          ]
  
bar = progressbar.ProgressBar(max_value=39, 
                              widgets=widgets).start()

for i in range(39):
    bar.update(i)
    rand = np.random.choice(t2['particle'].unique(), 80)
    # print(rand)
    t3 = pd.DataFrame(columns=t2.columns)
    # print(t3)
    alr_seen = []
    for i in rand:
        df_t = t2[t2['particle'] == i]
        # print(df_t)
        if i not in alr_seen:
            t3 = pd.concat([t3, df_t])
            alr_seen.append(i)
        else:
            k = np.random.randint(1000, 3939)
            while k in alr_seen:
                k = np.random.randint(1000, 3939)
            df_t = df_t.assign(particle = k)
            t3 = pd.concat([t3, df_t])
            alr_seen.append(k)
        # t3.append(df_t)
    t3.sort_values(by='frame', ascending=True)
    # print(t3['particle'].unique())
        
    em = tp.emsd(t3, 1/pixel_per_micron, fps) # microns per pixel = 100/285., frames per second = 24


    lim1 = 5
    lim2 = 60



    x = em.index[lim1:lim2]
    y = em.values[lim1:lim2]


    popt, pcov = curve_fit(func, x, y)

    # plt.plot(em.index, em.values, 'o', label='Exp data', markersize=8)
    # x2=em.index
    # #plt.plot(x2, m*x2 + c, 'r', label='Fitted line')
    # plt.loglog(x2, popt[0]*x2**popt[1] + popt[2], 'k', label='Fitted line ($K t^α +c$)')
    # #ax = em.plot(style='or', label='Trackpy')

    # plt.ylabel(r'$\langle \Delta r^2 \rangle$ [$\mu$m$^2$]',fontsize=14)
    # plt.xlabel('lag time $(s)$', fontsize=14);
    # plt.xticks(fontsize=14)
    # plt.yticks(fontsize=14)
    # plt.legend()
    # plt.show()

    # print(r'D_eff = {0:.5f} μm²/s^α'.format(popt[0]/4), end='\r')
    # print(r'α = {0:.5f} '.format(popt[1]), end='\r')
    # print(r'c = {0:.5f} μm²'.format(popt[2]), end='\r')
    lod.append(popt[0]/4)
    loa.append(popt[1])

print(lod)
print(loa)

print(np.average(lod))
print(np.average(loa))

print(np.std(lod))
print(np.std(loa))
