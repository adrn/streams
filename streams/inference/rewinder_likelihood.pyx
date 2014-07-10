# encoding: utf-8

import numpy as np
cimport numpy as np
np.import_array()

from scipy.misc import logsumexp

import cython
cimport cython

cimport streams.potential.basepotential as pot
import streams.potential.basepotential as pot

cdef extern from "math.h":
    double sqrt(double x)
    double atan2(double x, double x)
    double acos(double x)
    double sin(double x)
    double cos(double x)
    double log(double x)
    double fabs(double x)
    double exp(double x)

@cython.boundscheck(False)
@cython.cdivision(True)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef inline void leapfrog_init(double[:,::1] r, double[:,::1] v,
                        double[:,::1] v_12,
                        double[:,::1] acc, # return
                        int nparticles, double dt,
                        pot._Potential potential):

    potential._acceleration(r, acc, nparticles);

    for i in range(nparticles):
        for j in range(3):
            v_12[i,j] = v[i,j] + 0.5*dt*acc[i,j]

@cython.boundscheck(False)
@cython.cdivision(True)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef inline void leapfrog_step(double[:,::1] r, double[:,::1] v,
                        double[:,::1] v_12,
                        double[:,::1] acc,
                        int nparticles, double dt,
                        pot._Potential potential):
    """ Velocities need to be offset from positions by 1/2 step! To
        'prime' the integration, call

            leapfrog_init(r, v, v_12, ...)

        before looping over leapfrog_step!
    """
    cdef int i,j

    for i in range(nparticles):
        for j in range(3):
            r[i,j] = r[i,j] + dt*v_12[i,j]; # incr. pos. by full-step

    potential._acceleration(r, acc, nparticles);

    for i in range(nparticles):
        for j in range(3):
            v[i,j] = v[i,j] + dt*acc[i,j]; # incr. synced vel. by full-step
            v_12[i,j] = v_12[i,j] + dt*acc[i,j]; # incr. leapfrog vel. by full-step

@cython.boundscheck(False)
@cython.cdivision(True)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef inline double dot(double[::1] a, double[::1] b):
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]

@cython.boundscheck(False)
@cython.cdivision(True)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef inline double basis(double[:,::1] x, double[:,::1] v,
                         double[::1] x1_hat, double[::1] x2_hat,
                         double[::1] x3_hat):

    cdef double x1_norm, x2_norm, x3_norm

    # instantaneous cartesian basis to project into
    x1_hat[0] = x[0,0]
    x1_hat[1] = x[0,1]
    x1_hat[2] = x[0,2]

    x3_hat[0] = x1_hat[1]*v[0,2] - x1_hat[2]*v[0,1]
    x3_hat[1] = x1_hat[2]*v[0,0] - x1_hat[0]*v[0,2]
    x3_hat[2] = x1_hat[0]*v[0,1] - x1_hat[1]*v[0,0]

    x2_hat[0] = -x1_hat[1]*x3_hat[2] + x1_hat[2]*x3_hat[1]
    x2_hat[1] = -x1_hat[2]*x3_hat[0] + x1_hat[0]*x3_hat[2]
    x2_hat[2] = -x1_hat[0]*x3_hat[1] + x1_hat[1]*x3_hat[0]

    x1_norm = sqrt(dot(x1_hat, x1_hat))
    x2_norm = sqrt(dot(x2_hat, x2_hat))
    x3_norm = sqrt(dot(x3_hat, x3_hat))

    x1_hat[0] = x1_hat[0] / x1_norm
    x2_hat[0] = x2_hat[0] / x2_norm
    x3_hat[0] = x3_hat[0] / x3_norm

    x1_hat[1] = x1_hat[1] / x1_norm
    x2_hat[1] = x2_hat[1] / x2_norm
    x3_hat[1] = x3_hat[1] / x3_norm

    x1_hat[2] = x1_hat[2] / x1_norm
    x2_hat[2] = x2_hat[2] / x2_norm
    x3_hat[2] = x3_hat[2] / x3_norm

@cython.boundscheck(False)
@cython.cdivision(True)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef inline void ln_likelihood_helper(double sat_mass,
                               double[:,::1] x, double[:,::1] v, int nparticles,
                               double alpha, double[::1] betas,
                               pot._Potential potential,
                               double[::1] x1_hat, double[::1] x2_hat,
                               double[::1] x3_hat,
                               double[:,::1] ln_likelihoods, int ll_idx,
                               double[::1] dx, double[::1] dv):
    cdef int i
    cdef double x1,x2,x3,vx1,vx2,vx3
    cdef double beta
    cdef double ln_sigma_r, ln_sigma_v, sigma_r_sq, sigma_v_sq

    cdef double jac, d, BB, cosb, Rsun = 8.
    cdef double sat_R = sqrt(x[0,0]*x[0,0] + x[0,1]*x[0,1] + x[0,2]*x[0,2])
    cdef double sat_V = sqrt(v[0,0]*v[0,0] + v[0,1]*v[0,1] + v[0,2]*v[0,2])
    cdef double r_tide = potential._tidal_radius(sat_mass, sat_R)
    cdef double v_disp = sat_V * r_tide / sat_R

    ln_sigma_r = log(0.5*r_tide)
    sigma_r_sq = 0.25*r_tide*r_tide
    ln_sigma_v = log(v_disp)
    sigma_v_sq = v_disp*v_disp

    # compute instantaneous orbital plane coordinates
    basis(x, v, x1_hat, x2_hat, x3_hat)

    for i in range(nparticles):
        beta = betas[i]

        # translate to satellite position
        dx[0] = x[i+1,0] - x[0,0]
        dv[0] = v[i+1,0] - v[0,0]
        dx[1] = x[i+1,1] - x[0,1]
        dv[1] = v[i+1,1] - v[0,1]
        dx[2] = x[i+1,2] - x[0,2]
        dv[2] = v[i+1,2] - v[0,2]

        # hijacking these vars to use for jacobian calc.
        #   this is basically the transformation from GC cartesian to
        #   heliocentric spherical, but just the parts I need for the
        #   determinant of the jacobian
        x1 = x[i+1,0] + Rsun
        x2 = x[i+1,1]
        x3 = x[i+1,2]
        d = sqrt(x1*x1 + x2*x2 + x3*x3)
        BB = 1.5707963267948966 - acos(x3/d)
        cosb = cos(BB)
        jac = log(fabs(d*d*d*d*cosb))

        # project into new frame (dot product)
        x1 = dx[0]*x1_hat[0] + dx[1]*x1_hat[1] + dx[2]*x1_hat[2]
        x1 = x1 - alpha*beta*r_tide
        x2 = dx[0]*x2_hat[0] + dx[1]*x2_hat[1] + dx[2]*x2_hat[2]
        x3 = dx[0]*x3_hat[0] + dx[1]*x3_hat[1] + dx[2]*x3_hat[2]

        vx1 = dv[0]*x1_hat[0] + dv[1]*x1_hat[1] + dv[2]*x1_hat[2]
        vx2 = dv[0]*x2_hat[0] + dv[1]*x2_hat[1] + dv[2]*x2_hat[2]
        vx3 = dv[0]*x3_hat[0] + dv[1]*x3_hat[1] + dv[2]*x3_hat[2]

        # position likelihood is gaussian at lagrange points
        r_term = -0.5*((2*ln_sigma_r + x1*x1/sigma_r_sq) + \
                       (2*(ln_sigma_r + 0.6931471805599452) + x2*x2/(4*sigma_r_sq)) + \
                       (2*ln_sigma_r + x3*x3/sigma_r_sq))

        v_term = -0.5*((2*ln_sigma_v + vx1*vx1/sigma_v_sq) + \
                       (2*ln_sigma_v + vx2*vx2/sigma_v_sq) + \
                       (2*ln_sigma_v + vx3*vx3/sigma_v_sq))

        ln_likelihoods[ll_idx,i] = r_term + v_term + jac

cpdef back_integration_likelihood(double t1, double t2, double dt,
                                  pot._Potential potential,
                                  np.ndarray[double,ndim=2] s_gc,
                                  np.ndarray[double,ndim=2] p_gc,
                                  double logm0, double logmdot,
                                  double alpha, np.ndarray[double,ndim=1] _betas):
    """ 0th entry of x,v is the satellite position, velocity """

    cdef int i, nsteps, nparticles
    nparticles = len(p_gc)
    nsteps = int(fabs((t1-t2)/dt))

    cdef double [:,::1] ln_likelihoods = np.empty((nsteps, nparticles))
    cdef double [::1] dx = np.zeros(3)
    cdef double [::1] dv = np.zeros(3)
    cdef double [:,::1] v_12 = np.zeros((nparticles+1,3))

    cdef double [::1] x1_hat = np.empty(3)
    cdef double [::1] x2_hat = np.empty(3)
    cdef double [::1] x3_hat = np.empty(3)
    cdef double [:,::1] x = np.vstack((s_gc[:,:3],p_gc[:,:3]))
    cdef double [:,::1] v = np.vstack((s_gc[:,3:],p_gc[:,3:]))
    cdef double [:,::1] acc = np.empty((nparticles+1,3))
    cdef double G = 4.499753324353494927e-12 # kpc^3 / Myr^2 / M_sun
    cdef double q1,q2,qz,phi,v_halo,R_halo,C1,C2,C3,sinphi,cosphi
    cdef double [::1] betas = _betas

    # mass
    m0 = exp(logm0)
    mdot = exp(logmdot)

    # potential parameters
    # potential = pot.LM10Potential(1.E11, 6.5, 0.26,
    #                               3.4E10, 0.7,
    #                               potential_params['q1'],
    #                               potential_params['q2'],
    #                               potential_params['qz'],
    #                               potential_params['phi'],
    #                               potential_params['v_halo'],
    #                               exp(potential_params['log_R_halo']))
    mass = -mdot*t1 + m0

    # prime the accelerations
    leapfrog_init(x, v, v_12, acc, nparticles+1, dt, potential)

    #all_x,all_v = np.empty((2,nsteps,nparticles+1,ndim))
    #all_x[0] = x
    #all_v[0] = v
    ln_likelihood_helper(mass, x, v, nparticles,
                         alpha, betas, potential,
                         x1_hat, x2_hat, x3_hat,
                         ln_likelihoods, 0, dx, dv)

    for i in range(1,nsteps):
        leapfrog_step(x, v, v_12, acc, nparticles+1, dt, potential)
        #all_x[ii] = x
        #all_v[ii] = v
        t1 += dt

        # mass of the satellite
        mass = -mdot*t1 + m0

        ln_likelihood_helper(mass, x, v, nparticles,
                             alpha, betas, potential,
                             x1_hat, x2_hat, x3_hat,
                             ln_likelihoods, i, dx, dv)

    return logsumexp(ln_likelihoods, axis=0)
