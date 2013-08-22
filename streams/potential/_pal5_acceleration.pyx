# encoding: utf-8
# filename: _pal5_acceleration.pyx
"""
Deimos:
cython -a _integrate_lm10.pyx
gcc -shared -pthread -fPIC -fwrapv -O2 -Wall -fno-strict-aliasing -I/usr/local/lib/python2.7/dist-packages/numpy/core/include -o _integrate_lm10.so _integrate_lm10.c

cd -; cython -a _integrate_lm10.pyx; gcc -shared -pthread -fPIC -fwrapv -O2 -Wall -fno-strict-aliasing -I/usr/local/lib/python2.7/dist-packages/numpy/core/include -o _integrate_lm10.so _integrate_lm10.c; cd -

Laptop:
cython -a _leapfrog_cython.pyx; gcc -shared -pthread -fPIC -fwrapv -O2 -Wall -fno-strict-aliasing -I /usr/include/python2.7 -L /usr/lib/python2.7 -l python -I/System/Library/Frameworks/Python.framework/Versions/2.7/Extras/lib/python/numpy/core/include -o _leapfrog_cython.so _leapfrog_cython.c

Hotfoot:
gcc -shared -pthread -fPIC -fwrapv -O2 -Wall -fno-strict-aliasing -I/hpc/astro/users/amp2217/yt-x86_64/lib/python2.7/site-packages/numpy/core/include -I/hpc/astro/users/amp2217/yt-x86_64/include/python2.7  -o _integrate_lm10.so _integrate_lm10.c
"""
from __future__ import division

import sys
import numpy as np
cimport numpy as np
cimport cython

cdef extern from "math.h":
    double sqrt(double x)
    double sin(double x)
    double cos(double x)
    double log(double x)
    double exp(double x)

#DTYPE = np.double
#ctypedef np.double_t DTYPE_t

#@cython.profile(True)
@cython.boundscheck(False) # turn of bounds-checking for entire function
@cython.cdivision(True) 
@cython.wraparound(False)
@cython.nonecheck(False)
def pal5_acceleration(double[:, ::1] r not None, int n_particles, 
                      np.ndarray[double, ndim=2] data,
                      double qz, double log_m, double Rs):
    
    #cdef double[:, ::1] data = np.empty((n_particles, 3))
    
    cdef double fac1, fac2, fac3, _tmp, _tmp_1, _tmp_2, R_pl_c
    cdef double G, a, b_sq, c, m_disk, m_bulge
    G = 4.49975332435e-12 # kpc^3 / Myr^2 / M_sun
    
    # Miyamoto-Nagai
    a = 6.5 # kpc
    b_sq = 0.26*0.26 # kpc^2
    Gm_disk = G*1.E11 # M_sun
    
    # Hernquist
    c = 0.7 # kpc
    Gm_bulge = G*3.4E10 # M_sun
    
    # Halo
    cdef double qz_sq
    qz_sq = qz*qz
    Gm_halo = G*exp(log_m)
    
    cdef double x, y, z
    cdef double xx, yy, zz
    
    for ii in range(n_particles):
        x = r[ii,0]
        y = r[ii,1]
        z = r[ii,2]
        
        xx = x*x
        yy = y*y
        zz = z*z
    
        # Disk
        _tmp_1 = (a + sqrt(zz + b_sq))
        _tmp_2 = xx + yy + _tmp_1*_tmp_1
        fac1 = -Gm_disk / (_tmp_2 * sqrt(_tmp_2))
        _tmp = a/(sqrt(zz + b_sq))
        
        # Bulge
        R = sqrt(xx + yy + zz)
        R_pl_c = R + c
        fac2 = -Gm_bulge / (R_pl_c*R_pl_c * R)
        
        # Halo
        R = sqrt(x*x + y*y + z*z/qz_sq)
        
        term1 = 1./(R*R*(Rs+R))
        term2 = -log(1. + R/Rs) / (R*R*R)
        fac3 = Gm_halo * (term1 + term2)
        
        data[ii,0] = fac1*x + fac2*x + fac3*x
        data[ii,1] = fac1*y + fac2*y + fac3*y
        data[ii,2] = fac1*z*(1.+_tmp) + fac2*z + fac3*z/qz_sq
            
    return np.array(data)

@cython.boundscheck(False) # turn of bounds-checking for entire function
@cython.cdivision(True) 
@cython.wraparound(False)
@cython.nonecheck(False)
def pal5_logarithmic(double[:, ::1] r not None, int n_particles, 
                      np.ndarray[double, ndim=2] data,
                      double qz, double log_m, double Rs):
    
    #cdef double[:, ::1] data = np.empty((n_particles, 3))
    
    cdef double G, fac3, _tmp, _tmp_1, _tmp_2, R_pl_c
    G = 4.49975332435e-12 # kpc^3 / Myr^2 / M_sun
    
    # Halo
    cdef double qz_sq
    qz_sq = qz*qz
    Gm_halo = G*exp(log_m)
    
    cdef double x, y, z
    cdef double xx, yy, zz
    
    for ii in range(n_particles):
        x = r[ii,0]
        y = r[ii,1]
        z = r[ii,2]
        
        xx = x*x
        yy = y*y
        zz = z*z
        
        # Halo
        R = sqrt(x*x + y*y + z*z/qz_sq)
        
        term1 = 1./(R*R*(Rs+R))
        term2 = -log(1. + R/Rs) / (R*R*R)
        fac3 = Gm_halo * (term1 + term2)
        
        data[ii,0] = fac3*x
        data[ii,1] = fac3*y
        data[ii,2] = fac3*z/qz_sq
            
    return np.array(data)