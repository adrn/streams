# encoding: utf-8
# cython: boundscheck=False
# cython: cdivision=True
# cython: nonecheck=False
# cython: wraparound=False
# cython: profile=False
import sys

import numpy as np
cimport numpy as np
np.import_array()

import cython
cimport cython
# from cython.parallel import prange, parallel

cimport gary.potential.cpotential as pot
import gary.potential.cpotential as pot

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

cdef inline void leapfrog_init(double *r, double *v, double *v_12, double *grad,
                               double dt, pot._CPotential potential,
                               double *prog_r, double GMprog, int selfgravity) nogil:
    cdef double rel_x, rel_y, rel_z, fac

    # zero out the gradient holder (calling _gradient adds to grad)
    grad[0] = 0.
    grad[1] = 0.
    grad[2] = 0.

    # compute the gradient of this potential, add to the list of vectors grad
    potential._gradient(r, grad)

    if (selfgravity == 1):
        # if accounting for self-gravity of the progenitor, assume it's a point mass
        #   orbiting as a test particle
        rel_x = r[0] - prog_r[0]
        rel_y = r[1] - prog_r[1]
        rel_z = r[2] - prog_r[2]
        fac = GMprog / pow(rel_x*rel_x + rel_y*rel_y + rel_z*rel_z, 1.5)
        grad[0] += fac*rel_x
        grad[1] += fac*rel_y
        grad[2] += fac*rel_z

    v_12[0] = v[0] - 0.5*dt*grad[0]
    v_12[1] = v[1] - 0.5*dt*grad[1]
    v_12[2] = v[2] - 0.5*dt*grad[2]

cdef inline void leapfrog_step(double *r, double *v, double *v_12, double *grad,
                               double dt, pot._CPotential potential,
                               double *prog_r, double GMprog, int selfgravity) nogil:
    """ Velocities need to be offset from positions by 1/2 step! To
        'prime' the integration, call

            leapfrog_init(r, v, v_12, ...)

        before looping over leapfrog_step!
    """
    cdef double rel_x, rel_y, rel_z, fac

    # incr. pos. by full-step
    r[0] = r[0] + dt*v_12[0]
    r[1] = r[1] + dt*v_12[1]
    r[2] = r[2] + dt*v_12[2]

    # zero out the gradient holder (calling _gradient adds to grad)
    grad[0] = 0.
    grad[1] = 0.
    grad[2] = 0.

    potential._gradient(r, grad)

    if (selfgravity == 1):
        rel_x = r[0] - prog_r[0]
        rel_y = r[1] - prog_r[1]
        rel_z = r[2] - prog_r[2]
        fac = GMprog / pow(rel_x*rel_x + rel_y*rel_y + rel_z*rel_z, 1.5)
        grad[0] += fac*rel_x
        grad[1] += fac*rel_y
        grad[2] += fac*rel_z

    # incr. synced vel. by full-step
    v[0] = v[0] - dt*grad[0]
    v[1] = v[1] - dt*grad[1]
    v[2] = v[2] - dt*grad[2]

    # incr. leapfrog vel. by full-step
    v_12[0] = v_12[0] - dt*grad[0]
    v_12[1] = v_12[1] - dt*grad[1]
    v_12[2] = v_12[2] - dt*grad[2]

# -------------------------------------------------------------------------------------------------

cdef void set_basis(double *prog_x, double *prog_v,
                    double *x1_hat, double *x2_hat, double *x3_hat,
                    double sintheta, double costheta) nogil:
    """
        Cartesian basis defined by the orbital plane of the satellite to
        project the orbits of stars into.
    """

    cdef double x1_norm, x2_norm, x3_norm = 0.
    cdef unsigned int i

    x1_hat[0] = prog_x[0]
    x1_hat[1] = prog_x[1]
    x1_hat[2] = prog_x[2]

    x3_hat[0] = x1_hat[1]*prog_v[2] - x1_hat[2]*prog_v[1]
    x3_hat[1] = x1_hat[2]*prog_v[0] - x1_hat[0]*prog_v[2]
    x3_hat[2] = x1_hat[0]*prog_v[1] - x1_hat[1]*prog_v[0]

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

cdef double ln_likelihood_helper(double rtide, double vdisp,
                                 double *x, double *v,
                                 double *prog_x, double *prog_v,
                                 double *x1_hat, double *x2_hat, double *x3_hat,
                                 double *dx, double *dv,
                                 double alpha, double beta) nogil:

    # For Jacobian (spherical -> cartesian)
    cdef double R2, log_jac

    # Coordinates of stars in instantaneous orbital plane
    cdef double x1, x2, x3, v1, v2, v3

    # Likelihood terms
    cdef double r_term, v_term

    # Translate to be centered on progenitor
    dx[0] = x[0] - prog_x[0]
    dx[1] = x[1] - prog_x[1]
    dx[2] = x[2] - prog_x[2]
    dv[0] = v[0] - prog_v[0]
    dv[1] = v[1] - prog_v[1]
    dv[2] = v[2] - prog_v[2]

    # Hijacking these variables to use for Jacobian calculation
    x1 = x[0] + Rsun;
    R2 = x1*x1 + x[1]*x[1] + x[2]*x[2]
    x2 = x[2]*x[2]/R2
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
    #   -3*log(σ) - 3/2*log(2π)
    r_term = -3*log(rtide) - 2.756815599614018 - 0.5*(x1*x1 + x2*x2 + x3*x3)/rtide/rtide
    v_term = -3*log(vdisp) - 2.756815599614018 - 0.5*(v1*v1 + v2*v2 + v3*v3)/vdisp/vdisp

    return r_term + v_term + log_jac

cpdef rewinder_likelihood(double[:,::1] ln_likelihood,
                          double dt, np.intp_t nsteps,
                          pot._CPotential potential,
                          np.ndarray[double,ndim=2] prog_xv,
                          np.ndarray[double,ndim=2] star_xv,
                          double m0, double mdot,
                          double alpha, double[::1] betas,
                          double theta, np.intp_t selfgravity=1):

    cdef np.intp_t i, k, nparticles, ndim
    cdef double t1 = 0.
    nparticles = star_xv.shape[0]

    # Containers
    cdef double [::1] x1_hat = np.empty(3)
    cdef double [::1] x2_hat = np.empty(3)
    cdef double [::1] x3_hat = np.empty(3)
    cdef double [::1] dx = np.empty(3)
    cdef double [::1] dv = np.empty(3)
    cdef double [::1] menc_epsilon = np.empty(3)
    cdef double [:,::1] v_12 = np.zeros((nparticles+1,3))

    cdef double [:,::1] x = np.vstack((prog_xv[:,:3],star_xv[:,:3]))
    cdef double [:,::1] v = np.vstack((prog_xv[:,3:],star_xv[:,3:]))
    cdef double [:,::1] grad = np.zeros((nparticles+1,3))
    cdef double sintheta, costheta
    cdef double E_scale, sat_mass
    cdef double Gee, GMprog, rtide, vdisp

    # --------------  DEBUG  ------------------
    # cdef double [:,:,::1] all_x = np.empty((nsteps,nparticles+1,3))
    # cdef double [:,:,::1] all_v = np.empty((nsteps,nparticles+1,3))
    # --------------  DEBUG  ------------------

    Gee = potential.G
    sintheta = sin(theta)
    costheta = cos(theta)

    # mass
    sat_mass = m0
    GMprog = Gee * sat_mass

    with nogil:

        # prime the progenitor for leapfrogging (set velocity at half-step)
        leapfrog_init(&x[0,0], &v[0,0], &v_12[0,0], &grad[0,0],
                      dt, potential,
                      &x[0,0], 0., 0) # no self gravity for the progenitor orbit

        # compute approximations of tidal radius and velocity dispersion from mass enclosed
        E_scale = cbrt(sat_mass / potential._mass_enclosed(&x[0,0], &menc_epsilon[0], Gee))
        rtide = E_scale * sqrt(x[0,0]*x[0,0]+x[0,1]*x[0,1]+x[0,2]*x[0,2])
        vdisp = E_scale * sqrt(v[0,0]*v[0,0]+v[0,1]*v[0,1]+v[0,2]*v[0,2])

        # set the instantaneous orbital plane basis, (x1,x2,x3)
        set_basis(&x[0,0], &v[0,0], &x1_hat[0], &x2_hat[0], &x3_hat[0], sintheta, costheta)

        # loop over stars and prime for leapfrog (set velocity at half-step)
        #   then compute likelihood at initial conditions
        for k in range(1,nparticles+1):
            leapfrog_init(&x[k,0], &v[k,0], &v_12[k,0], &grad[k,0],
                          dt, potential,
                          &x[0,0], GMprog, selfgravity)

            ln_likelihood[0,k-1] = ln_likelihood_helper(rtide, vdisp, &x[k,0], &v[k,0],
                                                        &x[0,0], &v[0,0],
                                                        &x1_hat[0], &x2_hat[0], &x3_hat[0],
                                                        &dx[0], &dv[0],
                                                        alpha, betas[k-1])

        # --------------  DEBUG  ------------------
        # all_x[0,:,:] = x
        # all_v[0,:,:] = v
        # --------------  DEBUG  ------------------

        for i in range(1,nsteps):

            # progenitor
            leapfrog_step(&x[0,0], &v[0,0], &v_12[0,0], &grad[0,0],
                          dt, potential,
                          &x[0,0], 0., 0) # no self gravity for the progenitor orbit


            # mass of the satellite
            t1 += dt
            sat_mass = -mdot*t1 + m0
            GMprog = Gee * sat_mass

            # compute approximations of tidal radius and velocity dispersion from mass enclosed
            E_scale = cbrt(sat_mass / potential._mass_enclosed(&x[0,0], &menc_epsilon[0], Gee))
            rtide = E_scale * sqrt(x[0,0]*x[0,0]+x[0,1]*x[0,1]+x[0,2]*x[0,2])
            vdisp = E_scale * sqrt(v[0,0]*v[0,0]+v[0,1]*v[0,1]+v[0,2]*v[0,2])

            # set the instantaneous orbital plane basis, (x1,x2,x3)
            set_basis(&x[0,0], &v[0,0], &x1_hat[0], &x2_hat[0], &x3_hat[0], sintheta, costheta)

            # loop over stars
            for k in range(1,nparticles+1):
                leapfrog_step(&x[k,0], &v[k,0], &v_12[k,0], &grad[k,0],
                              dt, potential,
                              &x[0,0], GMprog, selfgravity)

                ln_likelihood[i,k-1] = ln_likelihood_helper(rtide, vdisp, &x[k,0], &v[k,0],
                                                            &x[0,0], &v[0,0],
                                                            &x1_hat[0], &x2_hat[0], &x3_hat[0],
                                                            &dx[0], &dv[0],
                                                            alpha, betas[k-1])

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
    pass

#     cdef unsigned int i
#     cdef double E_scale = 0.
#     cdef double [:,::1] grad = np.empty((1,3))
#     cdef double [:,::1] x = np.array(w0[:,:3])
#     cdef double [:,::1] v = np.array(w0[:,3:])
#     cdef double [:,::1] v_12 = np.zeros((1,3))
#     cdef double [:,::1] menc_epsilon = np.empty((1,3))
#     cdef double t1, sat_mass
#     cdef double Gee = potential.G

#     # prime the accelerations
#     t1 = fabs(dt*nsteps)
#     sat_mass = -mdot*t1 + m0

#     # prime the accelerations (progenitor)
#     leapfrog_init(x, v, v_12, grad, 0, dt, potential, 0., 0)

#     # compute approximations of tidal radius and velocity dispersion from mass enclosed
#     E_scale += cbrt(sat_mass / potential._mass_enclosed(x, menc_epsilon, Gee, 0)) * \
#                 (v[0,0]*v[0,0] + v[0,1]*v[0,1] + v[0,2]*v[0,2])

#     for i in range(1,nsteps):
#         t1 += dt
#         sat_mass = -mdot*t1 + m0

#         leapfrog_step(x, v, v_12, grad, 0, dt, potential, 0., 0)
#         E_scale += cbrt(sat_mass / potential._mass_enclosed(x, menc_epsilon, Gee, 0)) * \
#                         (v[0,0]*v[0,0] + v[0,1]*v[0,1] + v[0,2]*v[0,2])

#     return E_scale / float(nsteps)
