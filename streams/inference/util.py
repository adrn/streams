# coding: utf-8

""" Utilities.. """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys

# Third-party
import numpy as np
import astropy.units as u

# Project
from ..integrate import LeapfrogIntegrator

__all__ = ["particles_x1x2x3", "guess_tail_bit"]

def particles_x1x2x3(particles, satellite, potential, t1, t2, dt, at_tub=True):
    s = satellite
    p = particles

    X = np.vstack((s._X[...,:3], p._X[...,:3].copy()))
    V = np.vstack((s._X[...,3:], p._X[...,3:].copy()))
    integrator = LeapfrogIntegrator(potential._acceleration_at,
                                    np.array(X), np.array(V),
                                    args=(X.shape[0], np.zeros_like(X)))
    ts, rs, vs = integrator.run(t1=t1, t2=t2, dt=-1.)
    s_orbit = np.vstack((rs[:,0][:,np.newaxis].T, vs[:,0][:,np.newaxis].T)).T
    p_orbits = np.vstack((rs[:,1:].T, vs[:,1:].T)).T
    m_t = (-s.mdot*ts + s.m0)[:,np.newaxis]

    if at_tub:
        t_idx = np.array([np.argmin(np.fabs(ts - t)) for t in p.tub])
        s_orbit = np.array([s_orbit[jj,0] for jj in t_idx])
        p_orbits = np.array([p_orbits[jj,ii] for ii,jj in enumerate(t_idx)])
        m_t = np.squeeze(np.array([m_t[jj] for jj in t_idx]))

    r_tide = potential._tidal_radius(m_t, s_orbit[...,:3])
    s_R = np.sqrt(np.sum(s_orbit[...,:3]**2, axis=-1))
    s_V = np.sqrt(np.sum(s_orbit[...,3:]**2, axis=-1))
    v_disp = s_V * r_tide / s_R

    # instantaneous cartesian basis to project into
    x1_hat = s_orbit[...,:3] / np.sqrt(np.sum(s_orbit[...,:3]**2, axis=-1))[...,np.newaxis]
    _x2_hat = s_orbit[...,3:] / np.sqrt(np.sum(s_orbit[...,3:]**2, axis=-1))[...,np.newaxis]
    _x3_hat = np.cross(x1_hat, _x2_hat)
    _x2_hat = -np.cross(x1_hat, _x3_hat)
    x2_hat = _x2_hat / np.linalg.norm(_x2_hat, axis=-1)[...,np.newaxis]
    x3_hat = _x3_hat / np.linalg.norm(_x3_hat, axis=-1)[...,np.newaxis]

    # translate to satellite position
    rel_orbits = p_orbits - s_orbit
    rel_pos = rel_orbits[...,:3]
    rel_vel = rel_orbits[...,3:]

    # project onto X
    x1 = np.sum(rel_pos * x1_hat, axis=-1)
    x2 = np.sum(rel_pos * x2_hat, axis=-1)
    x3 = np.sum(rel_pos * x3_hat, axis=-1)

    vx1 = np.sum(rel_vel * x1_hat, axis=-1)
    vx2 = np.sum(rel_vel * x2_hat, axis=-1)
    vx3 = np.sum(rel_vel * x3_hat, axis=-1)

    return (x1,x2,x3,vx1,vx2,vx3), r_tide, v_disp

def guess_tail_bit(x1,x2):
    """ Guess the tail assigment for each particle. """

    Phi = np.arctan2(x2, x1)
    tail_bit = np.ones(len(x1))
    tail_bit[:] = np.nan
    tail_bit[np.cos(Phi) < -0.5] = -1.
    tail_bit[np.cos(Phi) > 0.5] = 1.

    return tail_bit