# coding: utf-8

""" Contains likelihood function specific to back-integration and
    the Rewinder
"""

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys

# Third-party
import numpy as np
import astropy.units as u
from scipy.misc import logsumexp

# Project
from ..coordinates import _hel_to_gc, _gc_to_hel, _xyz_sph_jac
from ..dynamics import Particle
from ..integrate import LeapfrogIntegrator

__all__ = ["back_integration_likelihood"]

def xyz_sph_jac(hel):
    l,b,d,mul,mub,vr = hel.T
    cosl, sinl = np.cos(l), np.sin(l)
    cosb, sinb = np.cos(b), np.sin(b)

    Rsun = 8.
    dtmnt = d**2*(Rsun**2*cosb + Rsun*d*sinb**2*cosl - 2*Rsun*d*cosl + d**2*sinb**4*cosb - d**2*cosb**5 + 2*d**2*cosb**3)*cosb
    deet = np.log(np.abs(dtmnt))
    return deet

def back_integration_likelihood(t1, t2, dt, potential, p_gc, s_gc, logm0, logmdot,
                                tub, beta, p_shocked, alpha):

    """ Compute the likelihood of 6D heliocentric star positions today given the
        potential and position of the satellite.

        Parameters
        ----------
        t1, t2, dt : float
            Integration parameters
        potential : streams.Potential
            The potential class.
        p_gc : array_like
            A (nparticles, 6) shaped array containing 6D galactocentric coordinates
            for all stars.
        s_gc : array_like
            A (1, 6) shaped array containing 6D galactocentric coordinates
            for the satellite.
        logm0 : float
            Log of the initial mass of the satellite.
        logmdot : float
            Log of the mass loss rate.
        tub : array_like
            Array of unbinding times for each star.
        beta : array_like
            Array of tail bits (K) to specify leading/trailing tail for each star.
        p_shocked : array_like
            Probability the star was shocked.
        alpha : float
            Position of effective tidal radius.

    """

    K = np.sign(beta)
    p_shocked = np.array(p_shocked)

    gc = np.vstack((s_gc,p_gc)).copy()
    acc = np.zeros_like(gc[:,:3])
    integrator = LeapfrogIntegrator(potential._acceleration_at,
                                    np.array(gc[:,:3]), np.array(gc[:,3:]),
                                    args=(gc.shape[0], acc))

    ts, rs, vs = integrator.run(t1=t1, t2=t2, dt=dt)
    ntimes = len(ts)
    nparticles = gc.shape[0]-1

    # mass of the satellite vs. time
    m0 = np.exp(logm0)
    mdot = np.exp(logmdot)
    m_t = (-mdot*ts + m0)[:,np.newaxis]
    if np.any(m_t < 0):
        return -np.inf

    # separate out the orbit of the satellite from the orbit of the stars
    s_orbit = np.vstack((rs[:,0][:,np.newaxis].T, vs[:,0][:,np.newaxis].T)).T
    p_orbits = np.vstack((rs[:,1:].T, vs[:,1:].T)).T

    #########################################################################
    # if not marginalizing over unbinding time, get the orbit index closest to
    #   tub for each star
    t_idx = np.array([np.argmin(np.fabs(ts - t)) for t in tub])
    s_orbit = np.array([s_orbit[jj,0] for jj in t_idx])
    p_orbits = np.array([p_orbits[jj,ii] for ii,jj in enumerate(t_idx)])
    m_t = np.squeeze(np.array([m_t[jj] for jj in t_idx]))
    p_x_hel = _gc_to_hel(p_orbits)
    jac1 = _xyz_sph_jac(p_x_hel)
    #########################################################################

    #########################################################################
    # if marginalizing over tub, just use the full orbits
    # p_x_hel = _gc_to_hel(p_orbits.reshape(nparticles*ntimes,6))
    # jac1 = _xyz_sph_jac(p_x_hel).reshape(ntimes,nparticles)
    #########################################################################

    s_R = np.sqrt(np.sum(s_orbit[...,:3]**2, axis=-1))
    s_V = np.sqrt(np.sum(s_orbit[...,3:]**2, axis=-1))

    #r_tide = potential._tidal_radius(s_mass, s_orbit)
    r_tide = potential._tidal_radius(m_t, s_orbit[...,:3])
    v_disp = s_V * r_tide / s_R

    # instantaneous cartesian basis to project into
    x_hat = s_orbit[...,:3] / np.sqrt(np.sum(s_orbit[...,:3]**2, axis=-1))[...,np.newaxis]
    _y_hat = s_orbit[...,3:] / np.sqrt(np.sum(s_orbit[...,3:]**2, axis=-1))[...,np.newaxis]
    z_hat = np.cross(x_hat, _y_hat)
    y_hat = -np.cross(x_hat, z_hat)

    # translate to satellite position
    rel_orbits = p_orbits - s_orbit
    rel_pos = rel_orbits[...,:3]
    rel_vel = rel_orbits[...,3:]

    # project onto each
    X = np.sum(rel_pos * x_hat, axis=-1) / r_tide
    Y = np.sum(rel_pos * y_hat, axis=-1) / r_tide
    Z = np.sum(rel_pos * z_hat, axis=-1) / r_tide

    VX = np.sum(rel_vel * x_hat, axis=-1) / v_disp
    VY = np.sum(rel_vel * y_hat, axis=-1) / v_disp
    VZ = np.sum(rel_vel * z_hat, axis=-1) / v_disp

    # position likelihood is gaussian at lagrange points
    r_term = -0.5*((np.log(r_tide) + (X-alpha*K)**2) + \
                   (np.log(r_tide) + Y**2) + \
                   (np.log(r_tide) + Z**2))

    v_term = -0.5*((np.log(v_disp) + VX**2) + \
                   (np.log(v_disp) + VY**2) + \
                   (np.log(v_disp) + VZ**2))

    not_shocked = (r_term + v_term + jac1)

    # shocked
    var_r = 10.*r_tide**2
    r_term2 = -0.5*(3*np.log(var_r) + X/var_r + Y/var_r + Z/var_r)

    var_v = 10.*v_disp**2
    v_term2 = -0.5*(3*np.log(var_v) + VX/var_v + VY/var_v + VZ/var_v)

    shocked = (r_term2 + v_term2 + jac1)
    arg = np.vstack((shocked, not_shocked))

    scale = np.ones_like(arg)
    scale[0,:] = p_shocked
    scale[1,:] = 1-p_shocked

    lnlike = logsumexp(arg, b=scale, axis=0)

    return lnlike

    # import matplotlib.pyplot as plt
    # fig,axes = plt.subplots(1,3,figsize=(16,5),sharex=True)
    # bins = np.linspace(-3.,3.,50)
    # axes[0].hist(X - tail_bit, bins=bins, normed=True)
    # axes[1].hist(Y, bins=bins, normed=True)
    # axes[2].hist(Z, bins=bins, normed=True)
    # axes[0].hist(np.random.normal(0., np.sqrt(var_x),size=10000),
    #              bins=bins, alpha=0.3, normed=True)
    # axes[1].hist(np.random.normal(0., np.sqrt(var_y),size=10000),
    #              bins=bins, alpha=0.3, normed=True)
    # axes[2].hist(np.random.normal(0., np.sqrt(var_z),size=10000),
    #              bins=bins, alpha=0.3, normed=True)

    # for ii in range(nparticles):
    #     if shocked_bit[ii] == 0:
    #         c = 'k'
    #     else:
    #         c = 'r'
    #     axes[0].axvline(X[ii]-tail_bit[ii], c=c)
    #     axes[1].axvline(Y[ii], c=c)
    #     axes[2].axvline(Z[ii], c=c)
    # fig.savefig("/Users/adrian/Desktop/derp.png")

    # fig,axes = plt.subplots(1,3,figsize=(16,5),sharex=True)
    # bins = np.linspace(-3.,3.,50)
    # axes[0].hist(VX, bins=bins, normed=True)
    # axes[1].hist(VY, bins=bins, normed=True)
    # axes[2].hist(VZ, bins=bins, normed=True)
    # axes[0].hist(np.random.normal(0., np.sqrt(var_vx),size=10000),
    #              bins=bins, alpha=0.3, normed=True)
    # axes[1].hist(np.random.normal(0., np.sqrt(var_vy),size=10000),
    #              bins=bins, alpha=0.3, normed=True)
    # axes[2].hist(np.random.normal(0., np.sqrt(var_vz),size=10000),
    #              bins=bins, alpha=0.3, normed=True)

    # for ii in range(nparticles):
    #     if shocked_bit[ii] == 0:
    #         c = 'k'
    #     else:
    #         c = 'r'
    #     axes[0].axvline(VX[ii], c=c)
    #     axes[1].axvline(VY[ii], c=c)
    #     axes[2].axvline(VZ[ii], c=c)
    # fig.savefig("/Users/adrian/Desktop/derp2.png")
    # sys.exit(0)

    # return logsumexp(r_term + v_term + jac1, axis=0)
