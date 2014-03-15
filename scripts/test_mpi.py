# coding: utf-8

""" Test MPI on a given system. """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys
import time

# Third-party
import numpy as np
from emcee.utils import MPIPool

def f(x):
    time.sleep(0.1)
    return np.sqrt(x)

# Initialize the MPI pool
pool = MPIPool()

# Make sure the thread we're running on is the master
if not pool.is_master():
    pool.wait()
    sys.exit(0)

v = np.random.random(size=1000)

a = time.time()
results = pool.map(f, v)
pool.close()
print(time.time() - a, "MPI map")

# now try in serial
a = time.time()
map(f, v)
print(time.time() - a, "map")
