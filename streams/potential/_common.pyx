# cython -a _common.pyx; gcc -shared -pthread -fPIC -fwrapv -O2 -Wall -fno-strict-aliasing -I /usr/include/python2.7 -L /usr/lib/python2.7 -l python -I/System/Library/Frameworks/Python.framework/Versions/2.7/Extras/lib/python/numpy/core/include -o _common.so _common.c
from __future__ import division

import numpy as np
cimport numpy as np
cimport cython

cdef extern from "math.h":
    double fmod(double, double)
    double floor(double)
    double fmax(double, double)
    double fmin(double, double)
    double sqrt(double)
    int isnan(double)
    double fabs(double)

# This really only helps for >10^6 particles!

@cython.boundscheck(False) # turn of bounds-checking for entire function
def _miyamoto_nagai_dx(double G, double M, double a, double b, np.ndarray[double, ndim=1] _x, np.ndarray[double, ndim=1] _y, np.ndarray[double, ndim=1] _z, long length):

    cdef np.ndarray[double, ndim=1] data
    data = np.zeros(length, dtype=np.float64)

    for ii in range(length):
        data[ii] = G * M * _x[ii] / (_x[ii]*_x[ii] + _y[ii]*_y[ii] + (a + sqrt(_z[ii]*_z[ii] + b*b))**2)**1.5

    return data

@cython.boundscheck(False) # turn of bounds-checking for entire function
def _miyamoto_nagai_dy(double G, double M, double a, double b, np.ndarray[double, ndim=1] _x, np.ndarray[double, ndim=1] _y, np.ndarray[double, ndim=1] _z, long length):

    cdef np.ndarray[double, ndim=1] data
    data = np.zeros(length, dtype=np.float64)

    for ii in range(length):
        data[ii] = G * M * _y[ii] / (_x[ii]*_x[ii] + _y[ii]*_y[ii] + (a + sqrt(_z[ii]*_z[ii] + b*b))**2)**1.5

    return data

@cython.boundscheck(False) # turn of bounds-checking for entire function
def _miyamoto_nagai_dz(double G, double M, double a, double b, np.ndarray[double, ndim=1] _x, np.ndarray[double, ndim=1] _y, np.ndarray[double, ndim=1] _z, long length):

    cdef double zsqrt
    cdef np.ndarray[double, ndim=1] data
    data = np.zeros(length, dtype=np.float64)

    for ii in range(length):
        zsqrt = sqrt(_z[ii]*_z[ii] + b*b)
        data[ii] = G * M * _z[ii] * (1. + a / zsqrt) / (_x[ii]*_x[ii] + _y[ii]*_y[ii] + (a + zsqrt)**2)**1.5

    return data