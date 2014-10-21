# encoding: utf-8
# cython: boundscheck=False
import sys

import numpy as np
cimport numpy as np
np.import_array()

import cython
cimport cython
# from cython.parallel import prange

cimport streamteam.potential.cpotential as pot
import streamteam.potential.cpotential as pot

cdef extern from "math.h":
    double sqrt(double x) nogil
    double cbrt(double x) nogil
    double acos(double x) nogil
    double sin(double x) nogil
    double cos(double x) nogil
    double log(double x) nogil
    double fabs(double x) nogil
    double exp(double x) nogil
    double pow(double x, double n) nogil

cdef extern from "likelihood_help.c":
    void set_basis(double*, double*, double*, double*, double*)
    double test(double*, long)
    void ln_likelihood_helper(double, double,
                              double*, double*, long,
                              double*, double*, double*,
                              double*, double*,
                              double, double*, double, double,
                              double*)

@cython.cdivision(True)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef inline void leapfrog_init(double[:,::1] r, double[:,::1] v,
                               double[:,::1] v_12,
                               double[:,::1] acc,  # return
                               unsigned int nparticles, double dt,
                               pot._CPotential potential) nogil:

    potential._gradient(r, acc, nparticles);

    for i in range(nparticles):
        v_12[i,0] = v[i,0] - 0.5*dt*acc[i,0]
        v_12[i,1] = v[i,1] - 0.5*dt*acc[i,1]
        v_12[i,2] = v[i,2] - 0.5*dt*acc[i,2]

@cython.cdivision(True)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef inline void leapfrog_step(double[:,::1] r, double[:,::1] v,
                               double[:,::1] v_12,
                               double[:,::1] acc,
                               unsigned int nparticles, double dt,
                               pot._CPotential potential) nogil:
    """ Velocities need to be offset from positions by 1/2 step! To
        'prime' the integration, call

            leapfrog_init(r, v, v_12, ...)

        before looping over leapfrog_step!
    """
    cdef unsigned int i

    for i in range(nparticles):
        # incr. pos. by full-step
        r[i,0] = r[i,0] + dt*v_12[i,0]
        r[i,1] = r[i,1] + dt*v_12[i,1]
        r[i,2] = r[i,2] + dt*v_12[i,2]

    potential._gradient(r, acc, nparticles)

    for i in range(nparticles):
        # incr. synced vel. by full-step
        v[i,0] = v[i,0] - dt*acc[i,0]
        v[i,1] = v[i,1] - dt*acc[i,1]
        v[i,2] = v[i,2] - dt*acc[i,2]

        # incr. leapfrog vel. by full-step
        v_12[i,0] = v_12[i,0] - dt*acc[i,0]
        v_12[i,1] = v_12[i,1] - dt*acc[i,1]
        v_12[i,2] = v_12[i,2] - dt*acc[i,2]

@cython.cdivision(True)
@cython.wraparound(False)
@cython.nonecheck(False)
cpdef rewinder_likelihood(double dt, int nsteps,
                          pot._CPotential potential,
                          np.ndarray[double,ndim=2] prog_xv,
                          np.ndarray[double,ndim=2] star_xv,
                          double m0, double mdot,
                          double alpha, np.ndarray[double,ndim=1] _betas,
                          double theta):

    cdef unsigned int i, nparticles, ndim
    cdef double t1
    nparticles = len(star_xv)

    # Containers
    cdef double [::1] x1 = np.empty(3)
    cdef double [::1] x2 = np.empty(3)
    cdef double [::1] x3 = np.empty(3)
    cdef double [::1] dx = np.empty(3)
    cdef double [::1] dv = np.empty(3)
    cdef double [::1] menc_tmp = np.empty(3)
    cdef double [:,::1] menc_epsilon = np.empty((1,3))

    cdef double [::1] likelihoods = np.zeros(nparticles, dtype='d')
    cdef double [:,::1] ln_likelihood_tmp = np.zeros((nsteps,nparticles), dtype='d')
    cdef double [:,::1] v_12 = np.zeros((nparticles+1,3))

    cdef double [:,::1] x = np.vstack((prog_xv[:,:3],star_xv[:,:3]))
    cdef double [:,::1] v = np.vstack((prog_xv[:,3:],star_xv[:,3:]))
    cdef double [:,::1] acc = np.empty((nparticles+1,3))
    cdef double [::1] betas = _betas
    cdef double sintheta, costheta
    cdef double Menc

    # -- TURN THESE ON FOR DEBUGGING --
    # cdef double [:,:,::1] all_x = np.empty((nsteps,nparticles+1,3))
    # cdef double [:,:,::1] all_v = np.empty((nsteps,nparticles+1,3))
    # --

    sintheta = sin(theta)
    costheta = cos(theta)

    # mass
    t1 = fabs(dt*nsteps)
    mass = -mdot*t1 + m0

    # prime the accelerations
    leapfrog_init(x, v, v_12, acc, nparticles+1, dt, potential)

    # -- TURN THESE ON FOR DEBUGGING --
    # all_x[0] = x
    # all_v[0] = v
    # --

    Menc = potential._mass_enclosed(x[0], menc_epsilon, menc_tmp)
    rtide = cbrt(mass/Menc) * sqrt(x[0,0]*x[0,0]+x[0,1]*x[0,1]+x[0,2]*x[0,2])
    vdisp = cbrt(mass/Menc) * sqrt(v[0,0]*v[0,0]+v[0,1]*v[0,1]+v[0,2]*v[0,2])

    ln_likelihood_helper(rtide, vdisp,
                         &x[0,0], &v[0,0], nparticles,
                         &x1[0], &x2[0], &x3[0],
                         &dx[0], &dv[0],
                         alpha, &betas[0], sintheta, costheta,
                         &ln_likelihood_tmp[0,0])

    for i in range(1,nsteps-1):
        leapfrog_step(x, v, v_12, acc, nparticles+1, dt, potential)

        # -- TURN THESE ON FOR DEBUGGING --
        # all_x[i] = x
        # all_v[i] = v
        # --
        t1 += dt

        # mass of the satellite
        mass = -mdot*t1 + m0

        Menc = potential._mass_enclosed(x[0], menc_epsilon, menc_tmp)
        rtide = cbrt(mass/Menc) * sqrt(x[0,0]*x[0,0]+x[0,1]*x[0,1]+x[0,2]*x[0,2])
        vdisp = cbrt(mass/Menc) * sqrt(v[0,0]*v[0,0]+v[0,1]*v[0,1]+v[0,2]*v[0,2])

        ln_likelihood_helper(rtide, vdisp,
                             &x[0,0], &v[0,0], nparticles,
                             &x1[0], &x2[0], &x3[0],
                             &dx[0], &dv[0],
                             alpha, &betas[0], sintheta, costheta,
                             &ln_likelihood_tmp[i,0])

    leapfrog_step(x, v, v_12, acc, nparticles+1, dt, potential)
    t1 += dt
    mass = -mdot*t1 + m0
    Menc = potential._mass_enclosed(x[0], menc_epsilon, menc_tmp)
    rtide = cbrt(mass/Menc) * sqrt(x[0,0]*x[0,0]+x[0,1]*x[0,1]+x[0,2]*x[0,2])
    vdisp = cbrt(mass/Menc) * sqrt(v[0,0]*v[0,0]+v[0,1]*v[0,1]+v[0,2]*v[0,2])

    # -- TURN THESE ON FOR DEBUGGING --
    # all_x[i+1] = x
    # all_v[i+1] = v
    # --
    ln_likelihood_helper(rtide, vdisp,
                         &x[0,0], &v[0,0], nparticles,
                         &x1[0], &x2[0], &x3[0],
                         &dx[0], &dv[0],
                         alpha, &betas[0], sintheta, costheta,
                         &ln_likelihood_tmp[nsteps-1,0])

    # TODO: why am I getting NaNs???
    return np.array(ln_likelihood_tmp)
    # ln_likelihoods = logsumexp(ln_likelihood_tmp, axis=0, b=coeffs) + log(fabs(dt))
    # return ln_likelihoods

    # -- TURN THESE ON FOR DEBUGGING --
    # return np.log(fabs(dt)*np.array(ln_likelihoods)), np.array(all_x), np.array(all_v)
    # --
