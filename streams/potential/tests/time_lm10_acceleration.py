# coding: utf-8
from __future__ import print_function

import os
import pstats, cProfile
import numpy as np
import time as pytime

from astropy.io import ascii
from streams.potential.lm10 import lm10_acceleration

plot_path = "plots/tests/potential"
if not os.path.exists(plot_path):
    os.makedirs(plot_path)

base,xx = os.path.split(__file__)
r_tbl = ascii.read(os.path.join(base, 'particle_data.txt'))
n_particles = len(r_tbl)

r = np.zeros((n_particles, 3))
r[:,0] = r_tbl['x']
r[:,1] = r_tbl['y']
r[:,2] = r_tbl['z']

r_0 = np.array([0.,0.,0.])
data = np.zeros((n_particles,3))

def time_function(Niter=1000):
    for ii in range(Niter):
        lm10_acceleration(r, n_particles, 1.3, 1.3, 1.69, 0.125, 1., 12., r_0, data)

Niter = 100000
a = pytime.time()
time_function(Niter=Niter)
print((pytime.time() - a)/float(Niter) * 1E6, "µs per call")

'''
prof_filename = os.path.join(plot_path, "lm10_acceleration.prof")
cmd = "time_function()"

cProfile.runctx(cmd, globals(), locals(), prof_filename)

s = pstats.Stats(prof_filename)
s.strip_dirs().sort_stats("time").print_stats()
'''