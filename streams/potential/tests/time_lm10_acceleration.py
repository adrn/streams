# coding: utf-8
from __future__ import print_function

import os
import pstats, cProfile
import numpy as np
import time as pytime

from streams.potential.lm10 import lm10_acceleration

plot_path = "plots/tests/potential"
if not os.path.exists(plot_path):
    os.makedirs(plot_path)

prof_filename = os.path.join(plot_path, "lm10_acceleration.prof")

r = np.random.random(size=(100, 3))
r_0 = np.array([10.,0.,0.])

def time_function(Niter=1000):
    for ii in range(Niter):
        lm10_acceleration(r, 1.3, 1.3, 1.69, 0.125, 1., 12., r_0)

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