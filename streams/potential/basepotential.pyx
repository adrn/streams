# coding: utf-8

""" Cython base Potential class """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Third-party
import numpy as np
cimport numpy as np
np.import_array()

import cython
cimport cython

class Potential(object):

    """ When subclassing, either need to override these methods, or write your own
        Cython _Potential subclass that implements the required methods. See lm10.pyx
        for an example.
    """

    def evaluate(self, xyz):
        return self.c_instance.evaluate(xyz)

    def acceleration(self, xyz):
        return self.c_instance.acceleration(xyz)

    def var_acceleration(self, w):
        return self.c_instance.var_acceleration(w)

    def tidal_radius(self, m_sat, r):
        return self.c_instance.tidal_radius(m_sat, r)

# ==============================================================================

cdef class _Potential:

    cpdef evaluate(self, double[:,::1] xyz):
        cdef int nparticles, ndim
        nparticles = xyz.shape[0]
        ndim = xyz.shape[1]

        cdef double [::1] pot = np.empty(nparticles)
        self._evaluate(xyz, pot, nparticles)
        return np.array(pot)

    @cython.boundscheck(False)
    @cython.cdivision(True)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    cdef public void _evaluate(self, double[:,::1] xyz, double[::1] pot, int nparticles):
        for i in range(nparticles):
            pot[i] = 0.

    # -------------------------------------------------------------
    cpdef acceleration(self, double[:,::1] xyz):
        cdef int nparticles, ndim
        nparticles = xyz.shape[0]
        ndim = xyz.shape[1]

        cdef double [:,::1] acc = np.empty((nparticles,ndim//2))
        self._acceleration(xyz, acc, nparticles)
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

    # -------------------------------------------------------------
    cpdef var_acceleration(self, double[:,::1] w):
        cdef int nparticles, ndim
        nparticles = w.shape[0]
        ndim = w.shape[1]

        cdef double [:,::1] acc = np.empty((nparticles,ndim))
        self._var_acceleration(w, acc, nparticles)
        return np.array(acc)

    @cython.boundscheck(False)
    @cython.cdivision(True)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    cdef public void _var_acceleration(self, double[:,::1] w, double[:,::1] acc, int nparticles):
        for i in range(nparticles):
            acc[i,0] = 0.
            acc[i,1] = 0.
            acc[i,2] = 0.

    # -------------------------------------------------------------
    cpdef tidal_radius(self, double m_sat, double[::1] r):
        cdef int nparticles
        nparticles = r.shape[0]

        cdef double [::1] rt = np.empty((nparticles,))
        for i in range(nparticles):
            rt[i] = self._tidal_radius(m_sat, r[i])

        return np.array(rt)

    @cython.boundscheck(False)
    @cython.cdivision(True)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    cdef public double _tidal_radius(self, double m_sat, double r):
        return 0.
