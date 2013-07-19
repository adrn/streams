# encoding: utf-8
# cython: profile=True
# filename: _lm10_acceleration.pyx
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
    double fmod(double, double)
    double floor(double)
    double fmax(double, double)
    double fmin(double, double)
    double sqrt(double)
    double log(double)
    double sin(double)
    double cos(double)
    int isnan(double)
    double fabs(double)

DTYPE = np.double
ctypedef np.double_t DTYPE_t

@cython.boundscheck(False) # turn of bounds-checking for entire function
@cython.cdivision(True) 
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.profile(True)
def lm10_acceleration(double[:, ::1] r not None, 
                      double q1, double qz, double phi, double v_halo, 
                      double q2, double r_halo, np.ndarray[double, ndim=1] r_0):
    
    n_particles = r.shape[0]
    cdef double[:, ::1] data = np.empty((n_particles, 3))
    #cdef double[:] fac1 = np.empty((n_particles,))
    #cdef double[:] fac2 = np.empty((n_particles,))
    #cdef double[:] fac3 = np.empty((n_particles,))
    #cdef double[:] _tmp = np.empty((n_particles,))
    
    cdef double fac1, fac2, fac3, _tmp
    cdef double G, a, b, c, m_disk, m_bulge
    G = 4.49975332435e-12 # kpc^3 / Myr^2 / M_sun
    
    # Miyamoto-Nagai
    a = 6.5 # kpc
    b = 0.26 # kpc
    m_disk = 1.E11 # M_sun
    
    # Hernquist
    c = 0.7 # kpc
    m_bulge = 3.4E10 # M_sun
    
    # Halo
    cdef double q1_sq, q2_sq, qz_sq, r_halo_sq, v_halo_sq
    q1_sq = q1*q1
    q2_sq = q2*q2
    qz_sq = qz*qz
    r_halo_sq = r_halo*r_halo # kpc
    v_halo_sq = v_halo*v_halo
    
    cdef double xx, yy, zz
    
    for ii in xrange(n_particles):    
        xx = r[ii,0]*r[ii,0]
        yy = r[ii,1]*r[ii,1]
        zz = r[ii,2]*r[ii,2]
    
        # Disk
        fac1 = -G*m_disk*((xx + yy) + (a + sqrt(zz + b**2))**2)**-1.5
        _tmp = a/(sqrt(zz + b**2))
        
        # Bulge
        R = sqrt(xx + yy + zz)
        fac2 = -G*m_bulge / ((R + c)**2 * R)
        
        # Halo
        C1 = cos(phi)**2/q1_sq + sin(phi)**2/q2_sq
        C2 = cos(phi)**2/q2_sq + sin(phi)**2/q1_sq
        C3 = 2.*sin(phi)*cos(phi)*(1./q1_sq - 1./q2_sq)
    
        fac3 = -v_halo_sq / (C1*xx + C2*yy + C3*r[ii,0]*r[ii,1] + zz/qz_sq + r_halo_sq)
        
        data[ii,0] = fac1*r[ii,0] + fac2*r[ii,0] + fac3*(2.*C1*r[ii,0] + C3*r[ii,1])
        data[ii,1] = fac1*r[ii,1] + fac2*r[ii,1] + fac3*(2.*C2*r[ii,1] + C3*r[ii,0])
        data[ii,2] = fac1*r[ii,2]*(1.+_tmp) + fac2*r[ii,2] + 2.*fac3*r[ii,2]/qz_sq
            
    return data
