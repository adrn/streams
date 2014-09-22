# encoding: utf-8

""" Fast transformation from heliocentric (Galactic) coordinates to
    galacocentric cartesian coordinates.
"""

from __future__ import division

import sys
import numpy as np
cimport numpy as np

import cython
cimport cython

cdef extern from "math.h":
    double sqrt(double x)
    double atan2(double x, double x)
    double acos(double x)
    double sin(double x)
    double cos(double x)
    double log(double x)
    double abs(double x)

def gal_to_hel(np.ndarray[double, ndim=2] X,
               double Rsun=8.,
               double Vcirc=(220*0.0010227121650537077)): # km/s to kpc/Myr
    """ Assumes Galactic units: kpc, Myr, radian, M_sun """

    cdef int nparticles = X.shape[0]
    cdef double[:,:] O = np.empty((nparticles, 6))

    cdef double x, y, z, vx, vy, vz, d_xy
    cdef double l,b,d,mul,mub,vr
    cdef double twopi = np.pi*2

    for ii in range(nparticles):
        # transform to heliocentric cartesian
        x = X[ii,0] + Rsun
        y = X[ii,1]
        z = X[ii,2]
        vx = X[ii,3]
        vy = X[ii,4] - Vcirc
        vz = X[ii,5]

        # transform from cartesian to spherical
        d = sqrt(x**2 + y**2 + z**2)
        d_xy = sqrt(x**2 + y**2)
        l = atan2(y, x)
        # b = 1.5707963267948966 - acos(z/d)
        b = atan2(z, d_xy)

        if l < 0:
            l = l+twopi

        # transform cartesian velocity to spherical
        vr = (vx*x + vy*y + vz*z) / d # kpc/Myr
        mul = -(vx*y - x*vy) / d_xy**2 # rad / Myr
        mub = -(z*(x*vx + y*vy) - d_xy**2*vz) / (d**2 * d_xy) # rad / Myr

        O[ii,0] = l
        O[ii,1] = b
        O[ii,2] = d
        O[ii,3] = mul
        O[ii,4] = mub
        O[ii,5] = vr

    return np.array(O)

def hel_to_gal(np.ndarray[double, ndim=2] O,
               double Rsun=8.,
               double Vcirc=(220*0.0010227121650537077)): # km/s to kpc/Myr
    """ Assumes Galactic units: kpc, Myr, radian, M_sun """

    cdef int nparticles = O.shape[0]
    cdef double[:,:] X = np.empty((nparticles, 6))

    cdef double l,b,d,mul,mub,vr
    cdef double x, y, z, vx, vy, vz, d_xy

    for ii in range(nparticles):
        l = O[ii,0]
        b = O[ii,1]
        d = O[ii,2]
        mul = O[ii,3]
        mub = O[ii,4]
        vr = O[ii,5]

        # transform from spherical to cartesian
        x = d*np.cos(b)*np.cos(l)
        y = d*np.cos(b)*np.sin(l)
        z = d*np.sin(b)

        # transform spherical velocity to cartesian
        mul = -mul
        mub = -mub

        vx = x/d*vr + y*mul + z*cos(l)*mub
        vy = y/d*vr - x*mul + z*sin(l)*mub
        vz = z/d*vr - d*cos(b)*mub

        x = x - Rsun
        vy = vy + Vcirc

        X[ii,0] = x
        X[ii,1] = y
        X[ii,2] = z
        X[ii,3] = vx
        X[ii,4] = vy
        X[ii,5] = vz

    return np.array(X)
