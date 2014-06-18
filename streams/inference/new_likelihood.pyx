# encoding: utf-8
# filename: .pyx

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
    double abs(double x)
    double exp(double x)

@cython.boundscheck(False)
@cython.cdivision(True)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef inline void lm10_acceleration(double[:,:] r, double[:,:] acc, int nparticles,
                            double GM_bulge, double c,
                            double GM_disk, double a, double b,
                            double C1, double C2, double C3, double qz,
                            double v_halo, double R_halo):

    cdef double facb, rb, rb_c
    cdef double facd, zd, rd2, ztmp
    cdef double fach
    cdef double b2, qz2, R_halo2, v_halo2
    cdef double x, y, z
    cdef double xx, yy, zz

    b2 = b*b
    qz2 = qz*qz
    R_halo2 = R_halo*R_halo
    v_halo2 = v_halo*v_halo

    for i in range(nparticles):
        x = r[i,0]
        y = r[i,1]
        z = r[i,2]

        xx = x*x
        yy = y*y
        zz = z*z

        # Disk
        ztmp = sqrt(zz + b2)
        zd = a + ztmp
        rd2 = xx + yy + zd*zd
        facd = -GM_disk / (rd2*sqrt(rd2))

        # Bulge
        rb = sqrt(xx + yy + zz)
        rb_c = rb + c
        facb = -GM_bulge / (rb_c*rb_c*rb)

        # Halo
        fach = -v_halo2 / (C1*xx + C2*yy + C3*x*y + zz/qz2 + R_halo2)

        acc[i,0] = facd*x + facb*x + fach*(2.*C1*x + C3*y)
        acc[i,1] = facd*y + facb*y + fach*(2.*C2*y + C3*x)
        acc[i,2] = facd*z*(1.+a/ztmp) + facb*z + 2.*fach*z/qz2

@cython.boundscheck(False)
@cython.cdivision(True)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef inline double lm10_tidal_radius(double m, double R,
                              double GM_bulge, double c,
                              double GM_disk, double a, double b,
                              double C1, double C2, double C3, double qz,
                              double v_halo, double R_halo):

    # Radius of Sgr center relative to galactic center
    cdef double GM_halo, m_enc, dlnM_dlnR, f
    cdef double G = 4.49975332435e-12 # kpc^3 / Myr^2 / M_sun

    GM_halo = (2*R*R*R*v_halo*v_halo) / (R*R + R_halo*R_halo)
    m_enc = (GM_disk + GM_bulge + GM_halo) / G

    dlnM_dlnR = (3*R_halo*R_halo + R*R)/(R_halo*R_halo + R*R)
    f = (1 - dlnM_dlnR/3.)

    return R * (m / (3*m_enc*f))**(0.3333333333333)

@cython.boundscheck(False)
@cython.cdivision(True)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef inline void leapfrog_step(double[:,:] r, double[:,:] v, double[:,:] acc,
                        int nparticles, double dt,
                        double GM_bulge, double c,
                        double GM_disk, double a, double b,
                        double C1, double C2, double C3, double qz,
                        double v_halo, double R_halo):
    """ Need to call
            lm10_acceleration(...)
        before looping over leapfrog_step to initialize!
    """
    cdef int i,j

    for i in range(nparticles):
        for j in range(3):
            v[i,j] = v[i,j] + 0.5*dt*acc[i,j]; # incr. vel. by half-step
            r[i,j] = r[i,j] + dt*v[i,j]; # incr. pos. by full-step

    lm10_acceleration(r, acc, nparticles,
                      GM_bulge, c,
                      GM_disk, a, b,
                      C1, C2, C3, qz, v_halo, R_halo);
    for i in range(nparticles):
        for j in range(3):
            v[i,j] = v[i,j] + 0.5*dt*acc[i,j]; # incr. vel. by half-step

@cython.boundscheck(False)
@cython.cdivision(True)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef inline double dot(double[:] a, double[:] b):
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]

@cython.boundscheck(False)
@cython.cdivision(True)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef inline double basis(double[:,:] x, double[:,:] v,
                         double[:] x1_hat, double[:] x2_hat, double[:] x3_hat):

    cdef double x1_norm, x2_norm, x3_norm

    # instantaneous cartesian basis to project into
    for i in range(3):
        x1_hat[i] = x[0,i]

    x3_hat[0] = x1_hat[1]*v[0,2] - x1_hat[2]*v[0,1]
    x3_hat[1] = x1_hat[2]*v[0,0] - x1_hat[0]*v[0,2]
    x3_hat[2] = x1_hat[0]*v[0,1] - x1_hat[1]*v[0,0]

    x2_hat[0] = -x1_hat[1]*x3_hat[2] + x1_hat[2]*x3_hat[1]
    x2_hat[1] = -x1_hat[2]*x3_hat[0] + x1_hat[0]*x3_hat[2]
    x2_hat[2] = -x1_hat[0]*x3_hat[1] + x1_hat[1]*x3_hat[0]

    x1_norm = sqrt(dot(x1_hat, x1_hat))
    x2_norm = sqrt(dot(x2_hat, x2_hat))
    x3_norm = sqrt(dot(x3_hat, x3_hat))

    for i in range(3):
        x1_hat[i] = x1_hat[i] / x1_norm
        x2_hat[i] = x2_hat[i] / x2_norm
        x3_hat[i] = x3_hat[i] / x3_norm

@cython.boundscheck(False)
@cython.cdivision(True)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef inline void ln_likelihood_helper(double sat_mass,
                               double[:,:] x, double[:,:] v, int nparticles,
                               double alpha, double[:] betas, double[:] jac,
                               double GM_bulge, double c,
                               double GM_disk, double a, double b,
                               double C1, double C2, double C3, double qz,
                               double v_halo, double R_halo,
                               double[:] x1_hat, double[:] x2_hat, double[:] x3_hat,
                               double[:] ln_likelihoods, double[:] dx, double[:] dv):
    cdef int i
    cdef double x1,x2,x3,vx1,vx2,vx3
    cdef double beta
    cdef double sigma_r, sigma_v

    # marginalizing over tub, use the full orbits
    #p_x_hel = _gc_to_hel(p_orbits.reshape(nparticles*ntimes,6)).reshape(ntimes,nparticles,6)
    #jac = xyz_sph_jac(p_x_hel).reshape(ntimes,nparticles)

    cdef double sat_R = sqrt(x[0,0]**2 + x[0,1]**2 + x[0,2]**2)
    cdef double sat_V = sqrt(v[0,0]**2 + v[0,1]**2 + v[0,2]**2)
    cdef double r_tide = lm10_tidal_radius(sat_mass, sat_R,
                                           GM_bulge, c,
                                           GM_disk, a, b,
                                           C1, C2, C3, qz, v_halo, R_halo)
    cdef double v_disp = sat_V * r_tide / sat_R

    # compute instantaneous orbital plane coordinates
    basis(x, v, x1_hat, x2_hat, x3_hat)

    for i in range(nparticles):
        beta = betas[i]
        # translate to satellite position
        for j in range(3):
            dx[j] = x[i+1,j] - x[0,j]
            dv[j] = v[i+1,j] - v[0,j]

        # project into new frame (dot product)
        x1 = dx[0]*x1_hat[0] + dx[1]*x1_hat[1] + dx[2]*x1_hat[2]
        x2 = dx[0]*x2_hat[0] + dx[1]*x2_hat[1] + dx[2]*x2_hat[2]
        x3 = dx[0]*x3_hat[0] + dx[1]*x3_hat[1] + dx[2]*x3_hat[2]

        vx1 = dv[0]*x1_hat[0] + dv[1]*x1_hat[1] + dv[2]*x1_hat[2]
        vx2 = dv[0]*x2_hat[0] + dv[1]*x2_hat[1] + dv[2]*x2_hat[2]
        vx3 = dv[0]*x3_hat[0] + dv[1]*x3_hat[1] + dv[2]*x3_hat[2]

        sigma_r = 0.5*r_tide
        sigma_v = v_disp

        # position likelihood is gaussian at lagrange points
        r_term = -0.5*((2*log(sigma_r) + (x1-alpha*beta*r_tide)**2/sigma_r**2) + \
                       (2*log(2*sigma_r) + x2**2/(2*sigma_r)**2) + \
                       (2*log(sigma_r) + x3**2/sigma_r**2))

        v_term = -0.5*((2*log(sigma_v) + vx1**2/sigma_v**2) + \
                       (2*log(sigma_v) + vx2**2/sigma_v**2) + \
                       (2*log(sigma_v) + vx3**2/sigma_v**2))

        ln_likelihoods[i] = r_term + v_term + jac[i]

def back_integration_likelihood(double t1, double t2, double dt,
                                object potential_params,
                                np.ndarray[double,ndim=2] s_gc,
                                np.ndarray[double,ndim=2] p_gc,
                                double logm0, double logmdot,
                                double alpha, np.ndarray[double,ndim=1] betas):
    """ 0th entry of x,v is the satellite position, velocity """

    cdef int i, nsteps, nparticles
    nparticles = len(p_gc)
    nsteps = int(abs((t1-t2)/dt)) # 6000

    cdef double [:,:] ln_likelihoods = np.empty((nsteps, nparticles)).astype(np.double)
    cdef double [:] dx = np.zeros(3).astype(np.double)
    cdef double [:] dv = np.zeros(3).astype(np.double)

    cdef double [:] jac = np.zeros(nparticles).astype(np.double)
    cdef double [:] x1_hat = np.empty(3).astype(np.double)
    cdef double [:] x2_hat = np.empty(3).astype(np.double)
    cdef double [:] x3_hat = np.empty(3).astype(np.double)
    cdef double [:,:] w, x, v, acc
    cdef double G = 4.49975332435e-12 # kpc^3 / Myr^2 / M_sun
    cdef double q1,q2,qz,phi,v_halo,R_halo,C1,C2,C3,sinphi,cosphi

    w = np.vstack((s_gc,p_gc))
    x = w[:,:3].copy()
    v = w[:,3:].copy()
    acc = np.empty((nparticles+1,3)).astype(np.double)

    # mass
    m0 = exp(logm0)
    mdot = exp(logmdot)

    # potential parameters
    GM_disk = G*1.E11 # M_sun
    a = 6.5
    b = 0.26
    GM_bulge = G*3.4E10 # M_sun
    c = 0.7

    q1 = potential_params['q1']
    q2 = potential_params['q2']
    qz = potential_params['qz']
    phi = potential_params['phi']
    R_halo = potential_params['R_halo']
    v_halo = potential_params['v_halo']

    sinphi = sin(phi)
    cosphi = cos(phi)
    C1 = cosphi*cosphi/(q1*q1) + sinphi*sinphi/(q2*q2)
    C2 = cosphi*cosphi/(q2*q2) + sinphi*sinphi/(q1*q1)
    C3 = 2.*sinphi*cosphi*(1./(q1*q1) - 1./(q2*q2))

    # prime the accelerations
    lm10_acceleration(x, acc, nparticles+1,
                      GM_bulge, c,
                      GM_disk, a, b,
                      C1, C2, C3, qz, v_halo, R_halo)

    #all_x,all_v = np.empty((2,nsteps,nparticles+1,ndim))
    #all_x[0] = x
    #all_v[0] = v
    ln_likelihood_helper(m0, x, v, nparticles,
                         alpha, betas, jac,
                         GM_bulge, c,
                         GM_disk, a, b,
                         C1, C2, C3, qz, v_halo, R_halo,
                         x1_hat, x2_hat, x3_hat,
                         ln_likelihoods[0], dx, dv)

    for i in range(1,nsteps):
        leapfrog_step(x, v, acc, nparticles, dt,
                      GM_bulge, c,
                      GM_disk, a, b,
                      C1, C2, C3, qz, v_halo, R_halo)
        #all_x[ii] = x
        #all_v[ii] = v
        t1 += dt

        # mass of the satellite
        mass = -mdot*t1 + m0
        #if mass < 0:
        #    return -np.inf

        ln_likelihood_helper(m0, x, v, nparticles,
                             alpha, betas, jac,
                             GM_bulge, c,
                             GM_disk, a, b,
                             C1, C2, C3, qz, v_halo, R_halo,
                             x1_hat, x2_hat, x3_hat,
                             ln_likelihoods[i], dx, dv)