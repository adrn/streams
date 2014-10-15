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
    double acos(double x) nogil
    double sin(double x) nogil
    double cos(double x) nogil
    double log(double x) nogil
    double fabs(double x) nogil
    double exp(double x) nogil
    double pow(double x, double n) nogil

cdef extern from "rewinder.c":
    void set_basis(double*, double*, double*, double*, double*)
    double test(double*, long)
    void ln_likelihood_helper(double, double,
                              double*, double*, long,
                              double*, double*, double*,
                              double*, double*,
                              double, double*,
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
cpdef _rewinder_likelihood():
    cdef double [::1] x1 = np.empty(3)
    cdef double [::1] x2 = np.empty(3)
    cdef double [::1] x3 = np.empty(3)
    cdef double [::1] dx = np.empty(3)
    cdef double [::1] dv = np.empty(3)

    N = 16
    cdef double [:,::1] x = np.vstack((np.array([1.,0,0]),np.ones((N,3))))
    cdef double [:,::1] v = np.vstack((np.array([0.,1,0]),-1*np.ones((N,3))))
    cdef double alpha = 1.
    cdef double [::1] betas = np.ones(N)

    cdef double [::1] ln_likelihoods = np.empty(N)

    # print(test(&x[0,0], 3))

    # ln_likelihood_helper(1., 1.,
    #                      &x[0,0], &v[0,0], N,
    #                      &x1[0], &x2[0], &x3[0],
    #                      &dx[0], &dv[0],
    #                      alpha, &betas[0],
    #                      &ln_likelihoods[0])

    # print(np.array(x1))
    # print(np.array(x2))
    # print(np.array(x3))

    import time
    t1 = time.time()
    for i in range(1000000):
        ln_likelihood_helper(1., 1.,
                             &x[0,0], &v[0,0], N,
                             &x1[0], &x2[0], &x3[0],
                             &dx[0], &dv[0],
                             alpha, &betas[0],
                             &ln_likelihoods[0])
    print("{} µs per call".format((time.time()-t1)))
    print("{} ms per 6000".format((time.time()-t1)/1e6*6000*1000))

    print(np.array(x1))
    print(np.array(x2))
    print(np.array(x3))


@cython.cdivision(True)
@cython.wraparound(False)
@cython.nonecheck(False)
cpdef rewinder_likelihood(double t1, double t2, double dt,
                          pot._CPotential potential,
                          np.ndarray[double,ndim=2] prog_xv,
                          np.ndarray[double,ndim=2] star_xv,
                          double m0, double mdot,
                          double alpha, np.ndarray[double,ndim=1] _betas,
                          double theta):

    cdef unsigned int i, nsteps, nparticles, ndim
    nparticles = len(star_xv)
    nsteps = int(fabs((t1-t2)/dt))

    # Containers
    cdef double [::1] x1 = np.empty(3)
    cdef double [::1] x2 = np.empty(3)
    cdef double [::1] x3 = np.empty(3)
    cdef double [::1] dx = np.empty(3)
    cdef double [::1] dv = np.empty(3)

    cdef double [::1] ln_likelihood_tmp = np.zeros(nparticles)
    cdef double [:,::1] v_12 = np.zeros((nparticles+1,3))

    cdef double [:,::1] x = np.vstack((prog_xv[:,:3],star_xv[:,:3]))
    cdef double [:,::1] v = np.vstack((prog_xv[:,3:],star_xv[:,3:]))
    cdef double [:,::1] acc = np.empty((nparticles+1,3))
    cdef double [::1] betas = _betas

    # -- TURN THESE ON FOR DEBUGGING --
    # cdef double [:,:,::1] all_x = np.empty((nsteps,nparticles+1,3))
    # cdef double [:,:,::1] all_v = np.empty((nsteps,nparticles+1,3))
    # --

    # mass
    mass = -mdot*t1 + m0

    # prime the accelerations
    leapfrog_init(x, v, v_12, acc, nparticles+1, dt, potential)

    # -- TURN THESE ON FOR DEBUGGING --
    # all_x[0] = x
    # all_v[0] = v
    # --

    # TODO: hack
    rtide = 1.
    vdisp = 1.

    ln_likelihood_helper(rtide, vdisp,
                         &x[0,0], &v[0,0], nparticles,
                         &x1[0], &x2[0], &x3[0],
                         &dx[0], &dv[0],
                         alpha, &betas[0],
                         &ln_likelihood_tmp[0])

    for i in range(1,nsteps):
        # HACK
        leapfrog_step(x, v, v_12, acc, nparticles+1, dt, potential)

        # -- TURN THESE ON FOR DEBUGGING --
        # all_x[i] = x
        # all_v[i] = v
        # --
        t1 += dt

        # mass of the satellite
        mass = -mdot*t1 + m0

        ln_likelihood_helper(rtide, vdisp,
                             &x[0,0], &v[0,0], nparticles,
                             &x1[0], &x2[0], &x3[0],
                             &dx[0], &dv[0],
                             alpha, &betas[0],
                             &ln_likelihood_tmp[0])

    #return np.array(ln_likelihoods)
    # -- TURN THESE ON FOR DEBUGGING --
    # return np.array(ln_likelihoods), np.array(all_x), np.array(all_v)
    # --



























# @cython.cdivision(True)
# @cython.wraparound(False)
# @cython.nonecheck(False)
# cdef inline double basis(double[:,::1] x, double[:,::1] v,
#                          double[::1] x1_hat, double[::1] x2_hat,
#                          double[::1] x3_hat, double theta):
#     """
#     Theta is an extra angle to rotate by to account for observed rotation
#     of lagrange points.
#     """

#     cdef double x1_norm, x2_norm, x3_norm

#     # instantaneous cartesian basis to project into
#     x1_hat[0] = x[0,0]
#     x1_hat[1] = x[0,1]
#     x1_hat[2] = x[0,2]

#     x3_hat[0] = x1_hat[1]*v[0,2] - x1_hat[2]*v[0,1]
#     x3_hat[1] = x1_hat[2]*v[0,0] - x1_hat[0]*v[0,2]
#     x3_hat[2] = x1_hat[0]*v[0,1] - x1_hat[1]*v[0,0]

#     x2_hat[0] = -x1_hat[1]*x3_hat[2] + x1_hat[2]*x3_hat[1]
#     x2_hat[1] = -x1_hat[2]*x3_hat[0] + x1_hat[0]*x3_hat[2]
#     x2_hat[2] = -x1_hat[0]*x3_hat[1] + x1_hat[1]*x3_hat[0]

#     x1_norm = sqrt(dot(x1_hat, x1_hat))
#     x2_norm = sqrt(dot(x2_hat, x2_hat))
#     x3_norm = sqrt(dot(x3_hat, x3_hat))

#     x1_hat[0] = x1_hat[0] / x1_norm * cos(theta)
#     x2_hat[0] = x2_hat[0] / x2_norm * -sin(theta)
#     x3_hat[0] = x3_hat[0] / x3_norm

#     x1_hat[1] = x1_hat[1] / x1_norm * sin(theta)
#     x2_hat[1] = x2_hat[1] / x2_norm * cos(theta)
#     x3_hat[1] = x3_hat[1] / x3_norm

#     x1_hat[2] = x1_hat[2] / x1_norm
#     x2_hat[2] = x2_hat[2] / x2_norm
#     x3_hat[2] = x3_hat[2] / x3_norm

# @cython.cdivision(True)
# @cython.wraparound(False)
# @cython.nonecheck(False)
# cdef inline void ln_likelihood_helper(double sat_mass,
#                                       double[:,::1] x, double[:,::1] v, int nparticles,
#                                       double alpha, double[::1] betas,
#                                       pot._CPotential potential,
#                                       double[::1] x1_hat, double[::1] x2_hat,
#                                       double[::1] x3_hat, double theta,
#                                       double[:,::1] ln_likelihoods, int ll_idx,
#                                       double[::1] dx, double[::1] dv):
#     cdef int i
#     cdef double x1,x2,x3,vx1,vx2,vx3
#     cdef double beta
#     cdef double ln_sigma_r, ln_sigma_v, sigma_r_sq, sigma_v_sq

#     cdef double jac, d, BB, cosb, Rsun = 8.
#     cdef double sat_R = sqrt(x[0,0]*x[0,0] + x[0,1]*x[0,1] + x[0,2]*x[0,2])
#     cdef double sat_V = sqrt(v[0,0]*v[0,0] + v[0,1]*v[0,1] + v[0,2]*v[0,2])

#     # HACK THIS STUFF SUXXXXXXXX
#     cdef double [::1] _r_tide = np.empty((1,))
#     potential._tidal_radius(sat_mass, np.array([x[0]]), _r_tide, 1)
#     cdef double r_tide = _r_tide[0]
#     cdef double v_disp = sat_V * r_tide / sat_R

#     ln_sigma_r = log(0.5*r_tide)
#     sigma_r_sq = 0.25*r_tide*r_tide
#     ln_sigma_v = log(v_disp)
#     sigma_v_sq = v_disp*v_disp

#     # compute instantaneous orbital plane coordinates
#     basis(x, v, x1_hat, x2_hat, x3_hat, theta)

#     for i in range(nparticles):
#         beta = betas[i]

#         # translate to satellite position
#         dx[0] = x[i+1,0] - x[0,0]
#         dv[0] = v[i+1,0] - v[0,0]
#         dx[1] = x[i+1,1] - x[0,1]
#         dv[1] = v[i+1,1] - v[0,1]
#         dx[2] = x[i+1,2] - x[0,2]
#         dv[2] = v[i+1,2] - v[0,2]

#         # hijacking these vars to use for jacobian calc.
#         #   this is basically the transformation from GC cartesian to
#         #   heliocentric spherical, but just the parts I need for the
#         #   determinant of the jacobian
#         x1 = x[i+1,0] + Rsun
#         x2 = x[i+1,1]
#         x3 = x[i+1,2]
#         d = sqrt(x1*x1 + x2*x2 + x3*x3)
#         BB = 1.5707963267948966 - acos(x3/d)
#         cosb = cos(BB)
#         jac = log(fabs(d*d*d*d*cosb))

#         # project into new frame (dot product)
#         x1 = dx[0]*x1_hat[0] + dx[1]*x1_hat[1] + dx[2]*x1_hat[2]
#         x1 = x1 - alpha*beta*r_tide
#         x2 = dx[0]*x2_hat[0] + dx[1]*x2_hat[1] + dx[2]*x2_hat[2]
#         x3 = dx[0]*x3_hat[0] + dx[1]*x3_hat[1] + dx[2]*x3_hat[2]

#         vx1 = dv[0]*x1_hat[0] + dv[1]*x1_hat[1] + dv[2]*x1_hat[2]
#         vx2 = dv[0]*x2_hat[0] + dv[1]*x2_hat[1] + dv[2]*x2_hat[2]
#         vx3 = dv[0]*x3_hat[0] + dv[1]*x3_hat[1] + dv[2]*x3_hat[2]

#         # position likelihood is gaussian at lagrange points
#         r_term = (2*ln_sigma_r + x1*x1/sigma_r_sq) + \
#             (2*(ln_sigma_r + 0.6931471805599452) + x2*x2/(4*sigma_r_sq)) + \
#             (2*ln_sigma_r + x3*x3/sigma_r_sq)

#         v_term = (2*ln_sigma_v + vx1*vx1/sigma_v_sq) + \
#             (2*ln_sigma_v + vx2*vx2/sigma_v_sq) + \
#             (2*ln_sigma_v + vx3*vx3/sigma_v_sq)

#         ln_likelihoods[ll_idx,i] = -0.5*(r_term + v_term) + jac

# cpdef rewinder_likelihood(double t1, double t2, double dt,
#                           pot._CPotential potential,
#                           np.ndarray[double,ndim=2] prog_gal,
#                           np.ndarray[double,ndim=2] star_gal,
#                           double m0, double mdot,
#                           double alpha, np.ndarray[double,ndim=1] _betas,
#                           double theta):
#     """ 0th entry of x,v is the satellite position, velocity """

#     cdef int i, nsteps, nparticles, ndim
#     nparticles = len(star_gal)
#     nsteps = int(fabs((t1-t2)/dt))

#     cdef double [:,::1] ln_likelihoods = np.zeros((nsteps, nparticles))
#     cdef double [::1] dx = np.zeros(3)
#     cdef double [::1] dv = np.zeros(3)
#     cdef double [:,::1] v_12 = np.zeros((nparticles+1,3))

#     cdef double [::1] x1_hat = np.empty(3)
#     cdef double [::1] x2_hat = np.empty(3)
#     cdef double [::1] x3_hat = np.empty(3)
#     cdef double [:,::1] x = np.vstack((prog_gal[:,:3],star_gal[:,:3]))
#     cdef double [:,::1] v = np.vstack((prog_gal[:,3:],star_gal[:,3:]))
#     cdef double [:,::1] acc = np.empty((nparticles+1,3))
#     cdef double [::1] betas = _betas

#     # -- TURN THESE ON FOR DEBUGGING --
#     # cdef double [:,:,::1] all_x = np.empty((nsteps,nparticles+1,3))
#     # cdef double [:,:,::1] all_v = np.empty((nsteps,nparticles+1,3))
#     # --

#     # mass
#     mass = -mdot*t1 + m0

#     # prime the accelerations
#     leapfrog_init(x, v, v_12, acc, nparticles+1, dt, potential)

#     # -- TURN THESE ON FOR DEBUGGING --
#     # all_x[0] = x
#     # all_v[0] = v
#     # --
#     ln_likelihood_helper(mass, x, v, nparticles,
#                          alpha, betas, potential,
#                          x1_hat, x2_hat, x3_hat, theta,
#                          ln_likelihoods, 0, dx, dv)

#     for i in range(1,nsteps):
#         leapfrog_step(x, v, v_12, acc, nparticles+1, dt, potential)

#         # -- TURN THESE ON FOR DEBUGGING --
#         # all_x[i] = x
#         # all_v[i] = v
#         # --
#         t1 += dt

#         # mass of the satellite
#         mass = -mdot*t1 + m0

#         ln_likelihood_helper(mass, x, v, nparticles,
#                              alpha, betas, potential,
#                              x1_hat, x2_hat, x3_hat, theta,
#                              ln_likelihoods, i, dx, dv)

#     return np.array(ln_likelihoods)
#     # -- TURN THESE ON FOR DEBUGGING --
#     # return np.array(ln_likelihoods), np.array(all_x), np.array(all_v)
#     # --
