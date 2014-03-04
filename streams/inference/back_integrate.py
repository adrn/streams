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

def back_integration_likelihood(t1, t2, dt, potential, p_hel, s_hel, logm0, logmdot,
                                tub, tail_bit):

    """ Compute the likelihood of 6D heliocentric star positions today given the
        potential and position of the satellite.

        Parameters
        ----------
        t1, t2, dt : float
            Integration parameters
        potential : streams.Potential
            The potential class.
        p_hel : array_like
            A (nparticles, 6) shaped array containing 6D heliocentric coordinates
            for all stars.
        s_hel : array_like
            A (1, 6) shaped array containing 6D heliocentric coordinates
            for the satellite.
        logm0 : float
            Log of the initial mass of the satellite.
        logmdot : float
            Log of the mass loss rate.
        tub : array_like
            Array of unbinding times for each star.
        tail_bit : array_like
            Array of tail bits (K) to specify leading/trailing tail for each star.
        fac_R : float
            Variance hyper-parameters for stars to handle spread in distance
            during shocks.
        fac_V : float
            Variance hyper-parameters for stars to handle spread in velocity
            during shocks.

    """

    p_gc = _hel_to_gc(p_hel)
    s_gc = _hel_to_gc(s_hel)
    tail_bit = np.sign(tail_bit)

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
    v_disp = s_V * r_tide / s_R / 2.2

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
    X = np.sum(rel_pos * x_hat, axis=-1)
    Y = np.sum(rel_pos * y_hat, axis=-1)
    Z = np.sum(rel_pos * z_hat, axis=-1)

    VX = np.sum(rel_vel * x_hat, axis=-1)
    VY = np.sum(rel_vel * y_hat, axis=-1)
    VZ = np.sum(rel_vel * z_hat, axis=-1)

    # position likelihood is gaussian at lagrange points
    var_x = (np.median(r_tide)/5.)**2
    r_term = -0.5*((np.log(var_x) + (X - tail_bit*r_tide)**2/var_x) + \
                   (np.log(3*var_x) + (Y)**2/(3*var_x)) + \
                   (np.log(var_x) + (Z)**2/var_x))

    var_vx = np.median(v_disp)**2/2.

    v_term = -0.5*((np.log(3*var_vx) + (VX)**2/(3*var_vx)) + \
                   (np.log(var_vx) + (VY)**2/var_vx) + \
                   (np.log(var_vx) + (VZ)**2/var_vx))

    # return logsumexp(r_term + v_term + jac1, axis=0)
    return (r_term + v_term + jac1)
