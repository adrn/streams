# coding: utf-8

""" Cython-based Potential classes """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Third-party
import numpy as np
cimport numpy as np
np.import_array()

import cython
cimport cython

cdef extern from "math.h":
    double sqrt(double x)
    double atan2(double x, double x)
    double acos(double x)
    double sin(double x)
    double cos(double x)
    double log(double x)
    double fabs(double x)
    double exp(double x)

# ==============================================================================

cdef class Potential:

    cpdef evaluate(self, double[:,::1] r):
        cdef int nparticles, ndim
        nparticles = r.shape[0]
        ndim = r.shape[1]

        cdef double [::1] pot = np.empty(nparticles)
        self._evaluate(r, pot, nparticles)
        return np.array(pot)

    @cython.boundscheck(False)
    @cython.cdivision(True)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    cdef public void _evaluate(self, double[:,::1] r, double[::1] pot, int nparticles):
        for i in range(nparticles):
            pot[i] = 0.

    cpdef acceleration(self, double[:,::1] r):
        cdef int nparticles, ndim
        nparticles = r.shape[0]
        ndim = r.shape[1]

        cdef double [:,::1] acc = np.empty((nparticles,ndim))
        self._acceleration(r, acc, nparticles)
        return np.array(acc)

    @cython.boundscheck(False)
    @cython.cdivision(True)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    cdef public void _acceleration(self, double[:,::1] r, double[:,::1] acc, int nparticles):
        for i in range(nparticles):
            acc[i,0] = 0.
            acc[i,1] = 0.
            acc[i,2] = 0.

cdef class LM10Potential(Potential):

    # here need to cdef all the attributes
    cdef public double G
    cdef public double M_disk, a, b
    cdef public double M_bulge, c
    cdef public double q1, q2, qz, phi, v_halo, r_halo
    cdef double C1, C2, C3, GM_disk, GM_bulge
    cdef double b2, qz2, r_halo2, v_halo2

    def __init__(self,
                 double M_disk, double a, double b,
                 double M_bulge, double c,
                 double q1, double q2, double qz, double phi,
                 double v_halo, double r_halo):
        """ Units of everything should be in the system:
                kpc, Myr, radian, M_sun
        """
        cdef double G = 4.499753324353494927e-12 # kpc^3 / Myr^2 / M_sun
        cdef double sinphi, cosphi
        self.G = G

        # disk parameters
        self.GM_disk = G*M_disk
        self.M_disk = M_disk
        self.a = a
        self.b = b

        # bulge parameters
        self.GM_bulge = G*M_bulge
        self.M_bulge = M_bulge
        self.c = c

        # halo parameters
        self.q1 = q1
        self.q2 = q2
        self.qz = qz
        self.phi = phi
        self.v_halo = v_halo
        self.r_halo = r_halo

        # helpers
        self.b2 = self.b*self.b
        self.qz2 = self.qz*self.qz
        self.r_halo2 = self.r_halo*self.r_halo
        self.v_halo2 = self.v_halo*self.v_halo

        sinphi = sin(phi)
        cosphi = cos(phi)
        self.C1 = cosphi*cosphi/(q1*q1) + sinphi*sinphi/(q2*q2)
        self.C2 = cosphi*cosphi/(q2*q2) + sinphi*sinphi/(q1*q1)
        self.C3 = 2.*sinphi*cosphi*(1./(q1*q1) - 1./(q2*q2))

    @cython.boundscheck(False)
    @cython.cdivision(True)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    cdef public inline void _evaluate(self, double[:,::1] r,
                                      double[::1] pot, int nparticles):

        cdef double zd
        cdef double x, y, z
        cdef double xx, yy, zz

        for i in range(nparticles):
            x = r[i,0]
            y = r[i,1]
            z = r[i,2]

            xx = x*x
            yy = y*y
            zz = z*z

            R = sqrt(xx + yy + zz)

            zd = (self.a + sqrt(zz+self.b2))
            _disk = -self.GM_disk / np.sqrt(xx + yy + zd*zd)
            _spher = -self.GM_bulge / (R+self.c)
            _halo = self.v_halo2*log(self.C1*xx + self.C2*yy + self.C3*x*y +
                                        zz/self.qz2 + self.r_halo2)

            pot[i] = _disk + _spher + _halo

    @cython.boundscheck(False)
    @cython.cdivision(True)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    cdef public inline void _acceleration(self, double[:,::1] r,
                                          double[:,::1] acc, int nparticles):

        cdef:
            double facb, rb, rb_c
            double facd, zd, rd2, ztmp
            double fach
            double x, y, z
            double xx, yy, zz

        for i in range(nparticles):
            x = r[i,0]
            y = r[i,1]
            z = r[i,2]

            xx = x*x
            yy = y*y
            zz = z*z

            # Disk
            ztmp = sqrt(zz + self.b2)
            zd = self.a + ztmp
            rd2 = xx + yy + zd*zd
            facd = -self.GM_disk / (rd2*sqrt(rd2))

            # Bulge
            rb = sqrt(xx + yy + zz)
            rb_c = rb + self.c
            facb = -self.GM_bulge / (rb_c*rb_c*rb)

            # Halo
            fach = -self.v_halo2 / (self.C1*xx + self.C2*yy + self.C3*x*y + \
                                    zz/self.qz2 + self.r_halo2)

            acc[i,0] = facd*x + facb*x + fach*(2.*self.C1*x + self.C3*y)
            acc[i,1] = facd*y + facb*y + fach*(2.*self.C2*y + self.C3*x)
            acc[i,2] = facd*z*(1.+self.a/ztmp) + facb*z + 2.*fach*z/self.qz2

    @cython.boundscheck(False)
    @cython.cdivision(True)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    cdef inline double tidal_radius(double m_sat, double R):

        # Radius of Sgr center relative to galactic center
        cdef double GM_halo, m_enc, dlnM_dlnR, f
        cdef double G = 4.499753324353494927e-12 # kpc^3 / Myr^2 / M_sun

        GM_halo = (2*R*R*R*v_halo*v_halo) / (R*R + R_halo*R_halo)
        m_enc = (GM_disk + GM_bulge + GM_halo) / G

        dlnM_dlnR = (3*R_halo*R_halo + R*R)/(R_halo*R_halo + R*R)
        f = (1 - dlnM_dlnR/3.)

        return R * (m / (3*m_enc*f))**(0.3333333333333)