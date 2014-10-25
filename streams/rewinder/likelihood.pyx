# encoding: utf-8
# cython: boundscheck=False
# cython: cdivision=True
# cython: nonecheck=False
# cython: wraparound=False
import sys

import numpy as np
cimport numpy as np
np.import_array()

import cython
cimport cython
from cython.parallel import prange, parallel

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

cdef double Rsun = 8.

cdef inline void leapfrog_init(double[:,::1] r, double[:,::1] v, double[:,::1] v_12,
                               double[:,::1] grad, unsigned int k, double dt,
                               pot._CPotential potential) nogil:

    potential._gradient(r, grad, k);
    v_12[k,0] = v[k,0] - 0.5*dt*grad[k,0]
    v_12[k,1] = v[k,1] - 0.5*dt*grad[k,1]
    v_12[k,2] = v[k,2] - 0.5*dt*grad[k,2]

cdef inline void leapfrog_step(double[:,::1] r, double[:,::1] v, double[:,::1] v_12,
                               double[:,::1] grad, unsigned int k, double dt,
                               pot._CPotential potential) nogil:
    """ Velocities need to be offset from positions by 1/2 step! To
        'prime' the integration, call

            leapfrog_init(r, v, v_12, ...)

        before looping over leapfrog_step!
    """

    # incr. pos. by full-step
    r[k,0] = r[k,0] + dt*v_12[k,0]
    r[k,1] = r[k,1] + dt*v_12[k,1]
    r[k,2] = r[k,2] + dt*v_12[k,2]

    potential._gradient(r, grad, k)

    # incr. synced vel. by full-step
    v[k,0] = v[k,0] - dt*grad[k,0]
    v[k,1] = v[k,1] - dt*grad[k,1]
    v[k,2] = v[k,2] - dt*grad[k,2]

    # incr. leapfrog vel. by full-step
    v_12[k,0] = v_12[k,0] - dt*grad[k,0]
    v_12[k,1] = v_12[k,1] - dt*grad[k,1]
    v_12[k,2] = v_12[k,2] - dt*grad[k,2]

    # if (selfgravity == 1):
    #     for i in range(1,nparticles):
    #         rel_x = r[i,0] - r[0,0]
    #         rel_y = r[i,1] - r[0,1]
    #         rel_z = r[i,2] - r[0,2]
    #         r_ip = sqrt(rel_x*rel_x + rel_y*rel_y + rel_z*rel_z)
    #         grad[i,0] += GMprog * rel_x / (r_ip*r_ip*r_ip)
    #         grad[i,1] += GMprog * rel_y / (r_ip*r_ip*r_ip)
    #         grad[i,2] += GMprog * rel_z / (r_ip*r_ip*r_ip)

# -------------------------------------------------------------------------------------------------

cdef void set_basis(double[:,::1] x, double[:,::1] v,
                    double[::1] x1_hat, double[::1] x2_hat, double[::1] x3_hat,
                    double sintheta, double costheta) nogil:
    """
        Cartesian basis defined by the orbital plane of the satellite to
        project the orbits of stars into.
    """

    cdef double x1_norm, x2_norm, x3_norm = 0.
    cdef unsigned int i

    x1_hat[0] = x[0,0]
    x1_hat[1] = x[0,1]
    x1_hat[2] = x[0,2]

    x3_hat[0] = x1_hat[1]*v[0,2] - x1_hat[2]*v[0,1]
    x3_hat[1] = x1_hat[2]*v[0,0] - x1_hat[0]*v[0,2]
    x3_hat[2] = x1_hat[0]*v[0,1] - x1_hat[1]*v[0,0]

    x2_hat[0] = -x1_hat[1]*x3_hat[2] + x1_hat[2]*x3_hat[1]
    x2_hat[1] = -x1_hat[2]*x3_hat[0] + x1_hat[0]*x3_hat[2]
    x2_hat[2] = -x1_hat[0]*x3_hat[1] + x1_hat[1]*x3_hat[0]

    x1_norm = sqrt(x1_hat[0]*x1_hat[0] + x1_hat[1]*x1_hat[1] + x1_hat[2]*x1_hat[2])
    x2_norm = sqrt(x2_hat[0]*x2_hat[0] + x2_hat[1]*x2_hat[1] + x2_hat[2]*x2_hat[2])
    x3_norm = sqrt(x3_hat[0]*x3_hat[0] + x3_hat[1]*x3_hat[1] + x3_hat[2]*x3_hat[2])

    for i in range(3):
        x1_hat[i] /= x1_norm
        x2_hat[i] /= x2_norm
        x3_hat[i] /= x3_norm

    if (sintheta != 0.):
        # using this as a temp variable
        x1_norm = x1_hat[0];
        x1_hat[0] = x1_hat[0]*costheta + x1_hat[1]*sintheta;
        x1_hat[1] = -x1_norm*sintheta + x1_hat[1]*costheta;

        x2_norm = x2_hat[0];
        x2_hat[0] = x2_hat[0]*costheta + x2_hat[1]*sintheta;
        x2_hat[1] = -x2_norm*sintheta + x2_hat[1]*costheta;

cdef double ln_likelihood_helper(double r_norm, double v_norm, double rtide, double sigma_r_sq, double sigma_v_sq,
                                 double[:,::1] x, double[:,::1] v,
                                 double[::1] x1_hat, double[::1] x2_hat, double[::1] x3_hat,
                                 double[::1] dx, double[::1] dv,
                                 double alpha, double beta,
                                 unsigned int k) nogil:

    # For Jacobian (spherical -> cartesian)
    cdef double R2, log_jac

    # Coordinates of stars in instantaneous orbital plane
    cdef double x1, x2, x3, v1, v2, v3

    # Likelihood terms
    cdef double r_term, v_term

    # Translate to be centered on progenitor
    dx[0] = x[k,0] - x[0,0]
    dv[0] = v[k,0] - v[0,0]
    dx[1] = x[k,1] - x[0,1]
    dv[1] = v[k,1] - v[0,1]
    dx[2] = x[k,2] - x[0,2]
    dv[2] = v[k,2] - v[0,2]

    # Hijacking these variables to use for Jacobian calculation
    x1 = x[k,0] + Rsun;
    R2 = x1*x1 + x[k,1]*x[k,1] + x[k,2]*x[k,2]
    x2 = x[k,2]*x[k,2]/R2
    log_jac = log(R2*R2*sqrt(1.-x2))

    # Project into new basis
    x1 = dx[0]*x1_hat[0] + dx[1]*x1_hat[1] + dx[2]*x1_hat[2]
    x2 = dx[0]*x2_hat[0] + dx[1]*x2_hat[1] + dx[2]*x2_hat[2]
    x3 = dx[0]*x3_hat[0] + dx[1]*x3_hat[1] + dx[2]*x3_hat[2]

    v1 = dv[0]*x1_hat[0] + dv[1]*x1_hat[1] + dv[2]*x1_hat[2]
    v2 = dv[0]*x2_hat[0] + dv[1]*x2_hat[1] + dv[2]*x2_hat[2]
    v3 = dv[0]*x3_hat[0] + dv[1]*x3_hat[1] + dv[2]*x3_hat[2]

    # Move to center of Lagrange point
    x1 += alpha*beta*rtide

    # Compute likelihoods for position and velocity terms
    r_term = r_norm - 0.5*(x1*x1 + x2*x2 + x3*x3)/sigma_r_sq
    v_term = v_norm - 0.5*(v1*v1 + v2*v2 + v3*v3)/sigma_v_sq

    return r_term + v_term + log_jac

cpdef rewinder_likelihood(double[:,::1] ln_likelihood,
                          double dt, int nsteps,
                          pot._CPotential potential,
                          np.ndarray[double,ndim=2] prog_xv,
                          np.ndarray[double,ndim=2] star_xv,
                          double m0, double mdot,
                          double alpha, double[::1] betas,
                          double theta, int selfgravity=1):

    cdef int i, k, nparticles, ndim
    cdef double t1
    nparticles = star_xv.shape[0]

    # Containers
    cdef double [::1] x1_hat = np.empty(3)
    cdef double [::1] x2_hat = np.empty(3)
    cdef double [::1] x3_hat = np.empty(3)
    cdef double [::1] dx = np.empty(3)
    cdef double [::1] dv = np.empty(3)
    cdef double [:,::1] menc_epsilon = np.empty((1,3))
    cdef double [:,::1] v_12 = np.zeros((nparticles+1,3))

    cdef double [:,::1] x = np.vstack((prog_xv[:,:3],star_xv[:,:3]))
    cdef double [:,::1] v = np.vstack((prog_xv[:,3:],star_xv[:,3:]))
    cdef double [:,::1] grad = np.empty((nparticles+1,3))
    cdef double sintheta, costheta
    cdef double E_scale, sat_mass
    cdef double Gee, GMprog, r_norm, v_norm, sigma_r_sq, sigma_v_sq

    # --------------  DEBUG  ------------------
    # cdef double [:,:,::1] all_x = np.empty((nsteps,nparticles+1,3))
    # cdef double [:,:,::1] all_v = np.empty((nsteps,nparticles+1,3))
    # --------------  DEBUG  ------------------

    Gee = potential.G
    sintheta = sin(theta)
    costheta = cos(theta)

    # mass
    t1 = fabs(dt*nsteps)
    sat_mass = -mdot*t1 + m0
    GMprog = Gee * sat_mass

    # prime the accelerations (progenitor)
    leapfrog_init(x, v, v_12, grad, 0, dt, potential)

    # compute approximations of tidal radius and velocity dispersion from mass enclosed
    E_scale = cbrt(sat_mass / potential._mass_enclosed(x, menc_epsilon, Gee, 0))
    rtide = E_scale * sqrt(x[0,0]*x[0,0]+x[0,1]*x[0,1]+x[0,2]*x[0,2])
    vdisp = E_scale * sqrt(v[0,0]*v[0,0]+v[0,1]*v[0,1]+v[0,2]*v[0,2])

    # define constants
    r_norm = -log(rtide) - 0.91893853320467267;
    v_norm = -log(vdisp) - 0.91893853320467267;
    sigma_r_sq = rtide*rtide;
    sigma_v_sq = vdisp*vdisp;
    set_basis(x, v, x1_hat, x2_hat, x3_hat, sintheta, costheta)

    # loop over stars
    with nogil:
        for k in range(1,nparticles+1):
            leapfrog_init(x, v, v_12, grad, k, dt, potential)
            ln_likelihood[0,k-1] = ln_likelihood_helper(r_norm, v_norm, rtide, sigma_r_sq, sigma_v_sq,
                                                        x, v, x1_hat, x2_hat, x3_hat, dx, dv,
                                                        alpha, betas[k-1], k)

    # --------------  DEBUG  ------------------
    # all_x[0,:,:] = x
    # all_v[0,:,:] = v
    # --------------  DEBUG  ------------------

    with nogil:
        for i in range(1,nsteps):
            # progenitor
            leapfrog_step(x, v, v_12, grad, 0, dt, potential)

            # mass of the satellite
            t1 += dt
            sat_mass = -mdot*t1 + m0
            GMprog = Gee * sat_mass

            # compute approximations of tidal radius and velocity dispersion from mass enclosed
            E_scale = cbrt(sat_mass / potential._mass_enclosed(x, menc_epsilon, Gee, 0))
            rtide = E_scale * sqrt(x[0,0]*x[0,0]+x[0,1]*x[0,1]+x[0,2]*x[0,2])
            vdisp = E_scale * sqrt(v[0,0]*v[0,0]+v[0,1]*v[0,1]+v[0,2]*v[0,2])

            # define constants
            r_norm = -log(rtide) - 0.91893853320467267;
            v_norm = -log(vdisp) - 0.91893853320467267;
            sigma_r_sq = rtide*rtide;
            sigma_v_sq = vdisp*vdisp;
            set_basis(x, v, x1_hat, x2_hat, x3_hat, sintheta, costheta)

            # loop over stars
            for k in prange(1,nparticles+1):
                leapfrog_step(x, v, v_12, grad, k, dt, potential)
                ln_likelihood[i,k-1] = ln_likelihood_helper(r_norm, v_norm, rtide, sigma_r_sq, sigma_v_sq,
                                                            x, v, x1_hat, x2_hat, x3_hat, dx, dv,
                                                            alpha, betas[k-1], k)

        # --------------  DEBUG  ------------------
        # all_x[i,:,:] = x
        # all_v[i,:,:] = v
        # --------------  DEBUG  ------------------

    # --------------  DEBUG  ------------------
    # return np.array(all_x), np.array(all_v)
    # --------------  DEBUG  ------------------


cpdef compute_dE(np.ndarray[double,ndim=2] w0,
                 double dt, int nsteps,
                 pot._CPotential potential,
                 double m0, double mdot):

    cdef unsigned int i
    cdef double E_scale = 0.
    cdef double [:,::1] grad = np.empty((1,3))
    cdef double [:,::1] x = np.array(w0[:,:3])
    cdef double [:,::1] v = np.array(w0[:,3:])
    cdef double [:,::1] v_12 = np.zeros((1,3))
    cdef double [:,::1] menc_epsilon = np.empty((1,3))
    cdef double t1, sat_mass
    cdef double Gee = potential.G

    # prime the accelerations
    t1 = fabs(dt*nsteps)
    sat_mass = -mdot*t1 + m0

    # prime the accelerations (progenitor)
    leapfrog_init(x, v, v_12, grad, 0, dt, potential)

    # compute approximations of tidal radius and velocity dispersion from mass enclosed
    E_scale += cbrt(sat_mass / potential._mass_enclosed(x, menc_epsilon, Gee, 0)) * \
                (v[0,0]*v[0,0] + v[0,1]*v[0,1] + v[0,2]*v[0,2])

    for i in range(1,nsteps):
        t1 += dt
        sat_mass = -mdot*t1 + m0

        leapfrog_step(x, v, v_12, grad, 0, dt, potential)
        E_scale += cbrt(sat_mass / potential._mass_enclosed(x, menc_epsilon, Gee, 0)) * \
                        (v[0,0]*v[0,0] + v[0,1]*v[0,1] + v[0,2]*v[0,2])

    return E_scale / float(nsteps)
