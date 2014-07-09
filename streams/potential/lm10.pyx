# coding: utf-8

""" Law & Majewski 2010 potential, both Cython and Python Potential classes """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Third-party
import numpy as np
cimport numpy as np
np.import_array()

import cython
cimport cython

from .basepotential cimport Potential

cdef extern from "math.h":
    double sqrt(double x)
    double atan2(double x, double x)
    double acos(double x)
    double sin(double x)
    double cos(double x)
    double log(double x)
    double fabs(double x)
    double exp(double x)

cdef class _LM10Potential(Potential):

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

    @cython.boundscheck(False) # turn of bounds-checking for entire function
    @cython.cdivision(True)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    cdef public inline void _var_acceleration(self, double[:,::1] w,
                                              double[:,::1] acc, int nparticles):

        cdef double x, y, z
        cdef double xx, yy, zz
        cdef double dx, dy, dz
        cdef double H, bz, abz, RD, r_2, r, rcr

        for ii in range(nparticles):
            x = w[ii,0]
            y = w[ii,1]
            z = w[ii,2]
            acc[ii,0] = w[ii,3]
            acc[ii,1] = w[ii,4]
            acc[ii,2] = w[ii,5]

            dx = w[ii,6]
            dy = w[ii,7]
            dz = w[ii,8]
            acc[ii,6] = w[ii,9]
            acc[ii,7] = w[ii,10]
            acc[ii,8] = w[ii,11]

            xx = x*x
            yy = y*y
            zz = z*z

            # some quantities for speed
            H = self.C1*xx + self.C2*yy + self.C3*x*y + zz/self.qz2 + self.r_halo2
            bz = sqrt(zz + self.b2)
            abz = self.a + bz
            RD = xx + yy + abz*abz
            r_2 = xx + yy + zz
            r = sqrt(r_2)
            rcr = r*(self.c+r)

            # -----------------------------------------
            # acceleration terms for the regular orbit

            # Bulge
            fB = self.GM_bulge / (rcr*(r+self.c))

            # Disk
            fD = self.GM_disk / RD**1.5

            # Halo
            fH = self.v_halo2 / H

            # acc. is -dPhi/d{xyz}
            acc[ii,3] = -(fB*x + fD*x + fH*(2.*self.C1*x + self.C3*y))
            acc[ii,4] = -(fB*y + fD*y + fH*(2.*self.C2*y + self.C3*x))
            acc[ii,5] = -(fB*z + fD*z*abz/bz + 2*fH*z/self.qz2)

            # -----------------------------------------
            # acceleration terms for the deviation orbit

            XX = 2*self.C1*fH - self.GM_bulge*xx/(r*rcr**2) + fB - 2*fB*xx/rcr + fD - 3*fD*xx/RD + fH*(-2*self.C1*x - self.C3*y)*(2*self.C1*x + self.C3*y)/H

            XY = self.C3*fH - self.GM_bulge*x*y/(r*rcr**2) - 2*fB*x*y/rcr - 3*fD*x*y/RD + fH*(2*self.C1*x + self.C3*y)*(-2*self.C2*y - self.C3*x)/H

            XZ = -self.GM_bulge*x*z/(r*rcr**2) - 2*fB*x*z/rcr - 3*fD*x*z*(self.a + bz)/(RD*bz) - 2*fH*z*(2*self.C1*x + self.C3*y)/(H*self.qz2)

            YY = 2*self.C2*fH - self.GM_bulge*yy/(r*rcr**2) + fB - 2*fB*yy/rcr + fD - 3*fD*yy/RD + fH*(-2*self.C2*y - self.C3*x)*(2*self.C2*y + self.C3*x)/H

            YZ = -self.GM_bulge*y*z/(r*rcr**2) - 2*fB*y*z/rcr - 3*fD*y*z*(self.a + bz)/(RD*bz) - 2*fH*z*(2*self.C2*y + self.C3*x)/(H*self.qz2)

            ZZ = -self.GM_bulge*zz/(r*rcr**2) + fB - 2*fB*zz/rcr + 2*fH/self.qz2 + fD*(self.a + bz)/bz + fD*zz/bz**2 - fD*zz*(self.a + bz)/bz**3 - 3*fD*zz*(self.a + bz)**2/(RD*bz**2) - 4*fH*zz/(H*self.qz2**2)

            acc[ii,9] = -(XX*dx + XY*dy + XZ*dz)
            acc[ii,10] = -(XY*dx + YY*dy + YZ*dz)
            acc[ii,11] = -(XZ*dx + YZ*dy + ZZ*dz)

    @cython.boundscheck(False)
    @cython.cdivision(True)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    cdef public double _tidal_radius(self, double m_sat, double R):
        """ This is a very crude estimate of the tidal radius... """

        # Radius of Sgr center relative to galactic center
        cdef double GM_halo, m_enc, dlnM_dlnR, f

        GM_halo = (2*R*R*R*self.v_halo2) / (R*R + self.r_halo2)
        m_enc = self.M_disk + self.M_bulge + GM_halo / self.G

        dlnM_dlnR = (3*self.r_halo2 + R*R)/(self.r_halo2 + R*R)
        f = (1 - dlnM_dlnR/3.)

        return R * (m_sat / (3*m_enc*f))**(0.3333333333333)