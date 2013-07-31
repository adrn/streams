# coding: utf-8
from __future__ import print_function

import os
import pstats, cProfile
import numpy as np
import time as pytime

from streams.potential._lm10_acceleration import lm10_acceleration

plot_path = "plots/tests/potential"
if not os.path.exists(plot_path):
    os.makedirs(plot_path)

#prof_filename = os.path.join(plot_path, "lm10_acceleration.prof")

n_particles = 100
r = np.random.random(size=(n_particles, 3))
acceleration = np.zeros((n_particles, 3))

def time_function(Niter=1000):
    for ii in range(Niter):
        acc = lm10_acceleration(r, n_particles, acceleration, 1.3, 1.3, 1.69, 0.125, 1., 12.)

Niter = 100000
a = pytime.time()
time_function(Niter=Niter)
print((pytime.time() - a)/float(Niter) * 1E6, "µs per call")

'''
cmd = "time_function()"

cProfile.runctx(cmd, globals(), locals(), prof_filename)

s = pstats.Stats(prof_filename)
s.strip_dirs().sort_stats("time").print_stats()
'''