"""
Deimos:
cython -a _common.pyx
gcc -shared -pthread -fPIC -fwrapv -O2 -Wall -fno-strict-aliasing -I/usr/local/lib/python2.7/dist-packages/numpy/core/include -o _common.so _common.c


gcc -shared -pthread -fPIC -fwrapv -O2 -Wall -fno-strict-aliasing -I /usr/include/python2.7 -L /usr/lib/python2.7 -l python -I/System/Library/Frameworks/Python.framework/Versions/2.7/Extras/lib/python/numpy/core/include -o _common.so _common.c
"""
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
def _cartesian_point_mass_df(double G, double m, long length, 
                             np.ndarray[double, ndim=1] origin, 
                             np.ndarray[double, ndim=1] x, 
                             np.ndarray[double, ndim=1] y, 
                             np.ndarray[double, ndim=1] z):
                             
    cdef np.ndarray[double, ndim=2] data
    data = np.zeros((3,length), dtype=np.float64)
    cdef double xx, yy, zz
    
    for ii in range(length):
        xx = (x[ii]-origin[0])
        yy = (y[ii]-origin[1])
        zz = (z[ii]-origin[2]) 
        
        fac = G * m * (xx*xx + yy*yy + zz*zz)**-1.5
        
        data[0][ii] = fac*xx
        data[1][ii] = fac*yy
        data[2][ii] = fac*zz

    return data