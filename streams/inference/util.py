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

    if at_tub:
        t_idx = np.array([np.argmin(np.fabs(ts - t)) for t in p.tub])
        s_orbit = np.array([s_orbit[jj,0] for jj in t_idx])
        p_orbits = np.array([p_orbits[jj,ii] for ii,jj in enumerate(t_idx)])

    # instantaneous cartesian basis to project into
    x_hat = s_orbit[...,:3] / np.sqrt(np.sum(s_orbit[...,:3]**2, axis=-1))[...,np.newaxis]
    y_hat = s_orbit[...,3:] / np.sqrt(np.sum(s_orbit[...,3:]**2, axis=-1))[...,np.newaxis]
    z_hat = np.cross(x_hat, y_hat)

    # translate to satellite position
    rel_orbits = p_orbits - s_orbit
    rel_pos = rel_orbits[...,:3]
    rel_vel = rel_orbits[...,3:]

    # project onto X
    X = np.sum(rel_pos * x_hat, axis=-1)
    Y = np.sum(rel_pos * y_hat, axis=-1)
    Z = np.sum(rel_pos * z_hat, axis=-1)

    VX = np.sum(rel_vel * x_hat, axis=-1)
    VY = np.sum(rel_vel * y_hat, axis=-1)
    VZ = np.sum(rel_vel * z_hat, axis=-1)

    return (X,Y,Z,VX,VY,VZ)

def guess_tail_bit(x1,x2):
    """ Guess the tail assigment for each particle. """

    Phi = np.arctan2(x2, x1)
    tail_bit = np.ones(len(x1))
    tail_bit[:] = np.nan
    tail_bit[np.cos(Phi) < -0.5] = -1.
    tail_bit[np.cos(Phi) > 0.5] = 1.

    return tail_bit