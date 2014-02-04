# encoding: utf-8
"""
Laptop:
cython -a _leapfrog_cython.pyx; gcc -shared -pthread -fPIC -fwrapv -O2 -Wall -fno-strict-aliasing -I /usr/include/python2.7 -L /usr/lib/python2.7 -l python -I/System/Library/Frameworks/Python.framework/Versions/2.7/Extras/lib/python/numpy/core/include -o _leapfrog_cython.so _leapfrog_cython.c
"""
from __future__ import division

import sys
import numpy as np
cimport numpy as np
cimport cython

#@cython.profile(True)
@cython.boundscheck(False) # turn of bounds-checking for entire function
@cython.cdivision(True)
@cython.wraparound(False)
@cython.nonecheck(False)
def _step(double[:, ::1] r_im1 not None,
          double[:, ::1] v_im1_2 not None,
          double[:, ::1] a_i not None,
          float dt,
          int n_particles,
          np.ndarray[double, ndim=2] r_i,
          np.ndarray[double, ndim=2] v_i,
          np.ndarray[double, ndim=2] v_ip1_2):

    cdef double half_dt = dt / 2.



    return np.array(r_i), np.array(v_i), np.array(v_ip1_2)