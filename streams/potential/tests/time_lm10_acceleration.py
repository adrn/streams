import os
import pstats, cProfile
import numpy as np
import time as pytime

import pyximport
pyximport.install()

from streams.potential._lm10_acceleration import lm10_acceleration

plot_path = "plots/tests/potential"
if not os.path.exists(plot_path):
    os.makedirs(plot_path)

prof_filename = os.path.join(plot_path, "lm10_acceleration.prof")

r = np.random.random(size=(100, 3))
r_0 = np.array([10.,0.,0.])

def time_function():
    for ii in range(10000):
        lm10_acceleration(r, 1.3, 1.3, 1.69, 0.125, 1., 12., r_0)

a = pytime.time()
time_function()
print(pytime.time() - a, "(sec) total time")

'''
cmd = "time_function()"

cProfile.runctx(cmd, globals(), locals(), prof_filename)

s = pstats.Stats(prof_filename)
s.strip_dirs().sort_stats("time").print_stats()
'''