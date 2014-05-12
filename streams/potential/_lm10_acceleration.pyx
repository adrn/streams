# encoding: utf-8
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

import cython
cimport cython

cdef extern from "math.h":
    double sqrt(double x)
    double sin(double x)
    double cos(double x)
    double log(double x)

#DTYPE = np.double
#ctypedef np.double_t DTYPE_t

@cython.boundscheck(False) # turn of bounds-checking for entire function
@cython.cdivision(True)
@cython.wraparound(False)
@cython.nonecheck(False)
def lm10_potential(np.ndarray[double, ndim=2] r, int n_particles,
                   np.ndarray[double, ndim=1] pot,
                   double q1, double qz, double phi, double v_halo,
                   double q2, double R_halo):

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
    cdef double q1_sq, q2_sq, qz_sq, R_halo_sq, v_halo_sq, sinphi, cosphi, C1, C2, C3
    q1_sq = q1*q1
    q2_sq = q2*q2
    qz_sq = qz*qz
    R_halo_sq = R_halo*R_halo # kpc
    v_halo_sq = v_halo*v_halo
    sinphi = sin(phi)
    cosphi = cos(phi)
    C1 = cosphi**2/q1_sq + sinphi**2/q2_sq
    C2 = cosphi**2/q2_sq + sinphi**2/q1_sq
    C3 = 2.*sinphi*cosphi*(1./q1_sq - 1./q2_sq)

    cdef double x, y, z
    cdef double xx, yy, zz

    for ii in range(n_particles):
        x = r[ii,0]
        y = r[ii,1]
        z = r[ii,2]

        xx = x*x
        yy = y*y
        zz = z*z
        R = sqrt(xx + yy + zz)

        _disk = -Gm_disk / np.sqrt(xx + yy + (a + sqrt(zz+b_sq))**2)
        _spher = -Gm_bulge / (R+c)
        _halo = v_halo_sq*log(C1*xx + C2*yy + C3*x*y + zz/qz_sq + R_halo_sq)

        pot[ii] = _disk + _spher + _halo

    return np.array(pot)

#@cython.profile(True)
@cython.boundscheck(False) # turn of bounds-checking for entire function
@cython.cdivision(True)
@cython.wraparound(False)
@cython.nonecheck(False)
def lm10_acceleration(np.ndarray[double, ndim=2] r, int n_particles,
                      np.ndarray[double, ndim=2] acc,
                      double q1, double qz, double phi, double v_halo,
                      double q2, double R_halo):

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
    cdef double q1_sq, q2_sq, qz_sq, R_halo_sq, v_halo_sq, sinphi, cosphi, C1, C2, C3
    q1_sq = q1*q1
    q2_sq = q2*q2
    qz_sq = qz*qz
    R_halo_sq = R_halo*R_halo # kpc
    v_halo_sq = v_halo*v_halo
    sinphi = sin(phi)
    cosphi = cos(phi)
    C1 = cosphi**2/q1_sq + sinphi**2/q2_sq
    C2 = cosphi**2/q2_sq + sinphi**2/q1_sq
    C3 = 2.*sinphi*cosphi*(1./q1_sq - 1./q2_sq)

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
        fac3 = -v_halo_sq / (C1*xx + C2*yy + C3*x*y + zz/qz_sq + R_halo_sq)

        acc[ii,0] = fac1*x + fac2*x + fac3*(2.*C1*x + C3*y)
        acc[ii,1] = fac1*y + fac2*y + fac3*(2.*C2*y + C3*x)
        acc[ii,2] = fac1*z*(1.+_tmp) + fac2*z + 2.*fac3*z/qz_sq

    return np.array(acc)

@cython.boundscheck(False) # turn of bounds-checking for entire function
@cython.cdivision(True)
@cython.wraparound(False)
@cython.nonecheck(False)
def lm10_variational_acceleration(np.ndarray[double, ndim=2] W, int n_particles,
                                  np.ndarray[double, ndim=2] acc,
                                  double q1, double qz, double phi, double v_halo,
                                  double q2, double R_halo):

    cdef double G, a, b_2, c
    G = 4.49975332435e-12 # kpc^3 / Myr^2 / M_sun

    # Miyamoto-Nagai
    a = 6.5 # kpc
    b_2 = 0.26*0.26 # kpc^2
    GMD = G*1.E11 # M_sun

    # Hernquist
    c = 0.7 # kpc
    GMB = G*3.4E10 # M_sun

    # Halo
    cdef double q1_2, q2_2, qz_2, rh_2, vh_2, sinphi, cosphi, C1, C2, C3
    q1_2 = q1*q1
    q2_2 = q2*q2
    qz_2 = qz*qz
    rh_2 = R_halo*R_halo # kpc
    vh_2 = v_halo*v_halo
    sinphi = sin(phi)
    cosphi = cos(phi)
    C1 = cosphi**2/q1_2 + sinphi**2/q2_2
    C2 = cosphi**2/q2_2 + sinphi**2/q1_2
    C3 = 2.*sinphi*cosphi*(1./q1_2 - 1./q2_2)

    cdef double x, y, z
    cdef double xx, yy, zz
    cdef double dx, dy, dz
    cdef double H, bz, abz, RD, r_2, r, rcr

    for ii in range(n_particles):
        x = W[ii,0]
        y = W[ii,1]
        z = W[ii,2]

        dx = W[ii,6]
        dy = W[ii,7]
        dz = W[ii,8]

        xx = x*x
        yy = y*y
        zz = z*z

        # some quantities for speed
        H = C1*xx + C2*yy + C3*x*y + zz/qz_2 + rh_2
        bz = sqrt(zz + b_2)
        abz = a + bz
        RD = xx + yy + abz*abz
        r_2 = xx + yy + zz
        r = sqrt(r_2)
        rcr = r*(c+r)

        # -----------------------------------------
        # acceleration terms for the regular orbit

        # Bulge
        fB = GMB / (rcr*(r+c))

        # Disk
        fD = GMD / RD**1.5

        # Halo
        fH = vh_2 / H

        # acc. is -dPhi/d{xyz}
        acc[ii,0] = -(fB*x + fD*x + fH*(2.*C1*x + C3*y))
        acc[ii,1] = -(fB*y + fD*y + fH*(2.*C2*y + C3*x))
        acc[ii,2] = -(fB*z + fD*z*abz/bz + 2*fH*z/qz_2)

        # -----------------------------------------
        # acceleration terms for the deviation orbit

        XX = 2*C1*fH - GMB*xx/(r*rcr**2) + fB - 2*fB*xx/rcr + fD - 3*fD*xx/RD + fH*(-2*C1*x - C3*y)*(2*C1*x + C3*y)/H

        XY = C3*fH - GMB*x*y/(r*rcr**2) - 2*fB*x*y/rcr - 3*fD*x*y/RD + fH*(2*C1*x + C3*y)*(-2*C2*y - C3*x)/H

        XZ = -GMB*x*z/(r*rcr**2) - 2*fB*x*z/rcr - 3*fD*x*z*(a + bz)/(RD*bz) - 2*fH*z*(2*C1*x + C3*y)/(H*qz_2)

        YY = 2*C2*fH - GMB*yy/(r*rcr**2) + fB - 2*fB*yy/rcr + fD - 3*fD*yy/RD + fH*(-2*C2*y - C3*x)*(2*C2*y + C3*x)/H

        YZ = -GMB*y*z/(r*rcr**2) - 2*fB*y*z/rcr - 3*fD*y*z*(a + bz)/(RD*bz) - 2*fH*z*(2*C2*y + C3*x)/(H*qz_2)

        ZZ = -GMB*zz/(r*rcr**2) + fB - 2*fB*zz/rcr + 2*fH/qz_2 + fD*(a + bz)/bz + fD*zz/bz**2 - fD*zz*(a + bz)/bz**3 - 3*fD*zz*(a + bz)**2/(RD*bz**2) - 4*fH*zz/(H*qz_2**2)

        acc[ii,3] = -(XX*dx + XY*dy + XZ*dz)
        acc[ii,4] = -(XY*dx + YY*dy + YZ*dz)
        acc[ii,5] = -(XZ*dx + YZ*dy + ZZ*dz)

    return np.array(acc)
