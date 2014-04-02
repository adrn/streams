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
                                beta, alpha, tub):
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
        beta : array_like
            Array of tail bits (K) to specify leading/trailing tail for each star.
        alpha : float
            Position of effective tidal radius.

    """

    # the tail assignment for each star. Only need this to be +/- 1
    beta = np.sign(beta)

    # stack the coordinates -- we integrate everything as test particles
    gc = np.vstack((s_gc,p_gc)).copy()
    acc = np.zeros_like(gc[:,:3])
    integrator = LeapfrogIntegrator(potential._acceleration_at,
                                    np.array(gc[:,:3]), np.array(gc[:,3:]),
                                    args=(gc.shape[0], acc))

    ts, rs, vs = integrator.run(t1=t1, t2=t2, dt=dt)
    ntimes = len(ts)
    nparticles = len(p_gc)

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
    #jac = _xyz_sph_jac(p_x_hel)
    jac = xyz_sph_jac(p_x_hel)
    #########################################################################

    #########################################################################
    # if marginalizing over tub, use the full orbits
    # p_x_hel = _gc_to_hel(p_orbits.reshape(nparticles*ntimes,6)).reshape(ntimes,nparticles,6)
    # jac = xyz_sph_jac(p_x_hel).reshape(ntimes,nparticles)
    #########################################################################

    s_R = np.sqrt(np.sum(s_orbit[...,:3]**2, axis=-1))
    s_V = np.sqrt(np.sum(s_orbit[...,3:]**2, axis=-1))

    #r_tide = potential._tidal_radius(s_mass, s_orbit)
    r_tide = potential._tidal_radius(m_t, s_orbit[...,:3])
    v_disp = s_V * r_tide / s_R

    # instantaneous cartesian basis to project into
    x1_hat = s_orbit[...,:3] / np.sqrt(np.sum(s_orbit[...,:3]**2, axis=-1))[...,np.newaxis]
    _x2_hat = s_orbit[...,3:] / np.sqrt(np.sum(s_orbit[...,3:]**2, axis=-1))[...,np.newaxis]
    _x3_hat = np.cross(x1_hat, _x2_hat)
    _x2_hat = -np.cross(x1_hat, _x3_hat)
    # x2_hat = _x2_hat / np.sqrt(np.sum(_x2_hat**2, axis=-1))[...,np.newaxis]
    # x3_hat = _x3_hat /np.sqrt(np.sum(_x3_hat**2, axis=-1))[...,np.newaxis]
    x2_hat = _x2_hat / np.linalg.norm(_x2_hat, axis=-1)[...,np.newaxis]
    x3_hat = _x3_hat / np.linalg.norm(_x3_hat, axis=-1)[...,np.newaxis]

    # translate to satellite position
    rel_orbits = p_orbits - s_orbit
    rel_pos = rel_orbits[...,:3]
    rel_vel = rel_orbits[...,3:]

    # # project onto each
    # X = np.sum(rel_pos * x_hat, axis=-1) / r_tide
    # Y = np.sum(rel_pos * y_hat, axis=-1) / r_tide
    # Z = np.sum(rel_pos * z_hat, axis=-1) / r_tide

    # VX = np.sum(rel_vel * x_hat, axis=-1) / v_disp
    # VY = np.sum(rel_vel * y_hat, axis=-1) / v_disp
    # VZ = np.sum(rel_vel * z_hat, axis=-1) / v_disp

    # # position likelihood is gaussian at lagrange points
    # sigma_r = sigma_v = 1.0
    # r_term = -0.5*((np.log(sigma_r) + (X-alpha*beta)**2/sigma_r**2) + \
    #                (np.log(sigma_r) + Y**2/sigma_r**2) + \
    #                (np.log(sigma_r) + Z**2/sigma_r**2))

    # v_term = -0.5*((np.log(sigma_v) + VX**2/sigma_v**2) + \
    #                (np.log(sigma_v) + VY**2/sigma_v**2) + \
    #                (np.log(sigma_v) + VZ**2/sigma_v**2))

    # project into new frame
    x1 = np.sum(rel_pos * x1_hat, axis=-1)
    x2 = np.sum(rel_pos * x2_hat, axis=-1)
    x3 = np.sum(rel_pos * x3_hat, axis=-1)

    vx1 = np.sum(rel_vel * x1_hat, axis=-1)
    vx2 = np.sum(rel_vel * x2_hat, axis=-1)
    vx3 = np.sum(rel_vel * x3_hat, axis=-1)

    # position likelihood is gaussian at lagrange points
    sigma_r = 0.5*r_tide
    r_term = -0.5*((2*np.log(sigma_r) + (x1-alpha*beta*r_tide)**2/sigma_r**2) + \
                   (2*np.log(2*sigma_r) + x2**2/(2*sigma_r)**2) + \
                   (2*np.log(sigma_r) + x3**2/sigma_r**2))

    sigma_v = v_disp
    v_term = -0.5*((2*np.log(sigma_v) + vx1**2/sigma_v**2) + \
                   (2*np.log(sigma_v) + vx2**2/sigma_v**2) + \
                   (2*np.log(sigma_v) + vx3**2/sigma_v**2))

    # import matplotlib.pyplot as plt
    # plt.figure(figsize=(5,5))
    # plt.plot(x1 / r_tide, x2 / r_tide, marker='o',
    #          alpha=0.4, color='k', linestyle='none')
    # plt.xlim(-3,3)
    # plt.ylim(-3,3)
    # plt.savefig("/Users/adrian/Desktop/derp.png")
    # sys.exit(0)

    #return logsumexp(r_term + v_term + jac, axis=0)

    return r_term + v_term + jac
