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

# Project
from ..coordinates import _hel_to_gc, _gc_to_hel
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

def back_integration_likelihood(t1, t2, dt, potential, p_hel, s_hel, tub):

    p_gc = _hel_to_gc(p_hel)
    s_gc = _hel_to_gc(s_hel)

    gc = np.vstack((s_gc,p_gc)).copy()
    acc = np.zeros_like(gc[:,:3])
    integrator = LeapfrogIntegrator(potential._acceleration_at,
                                    np.array(gc[:,:3]), np.array(gc[:,3:]),
                                    args=(gc.shape[0], acc))

    times, rs, vs = integrator.run(t1=t1, t2=t2, dt=dt)

    s_orbit = np.vstack((rs[:,0][:,np.newaxis].T, vs[:,0][:,np.newaxis].T)).T
    p_orbits = np.vstack((rs[:,1:].T, vs[:,1:].T)).T

    # These are the unbinding time indices for each particle
    t_idx = np.array([np.argmin(np.fabs(times - t)) for t in tub])

    # get back 6D positions for stars and satellite at tub
    # p_x = np.array([p_orbits[jj,ii] for ii,jj in enumerate(t_idx)])
    # s_x = np.array([s_orbit[jj,0] for jj in t_idx])
    # rel_x = p_x-s_x

    # p_x_hel = _gc_to_hel(p_x)
    # jac1 = xyz_sph_jac(p_x_hel)

    # r_tide = potential._tidal_radius(2.5e8, s_x)#*1.6
    #v_esc = potential._escape_velocity(2.5e8, r_tide=r_tide)
    v_disp = 0.017198632325

    r_tide = potential._tidal_radius(2.5e8, s_orbit)
    p_x_hel = _gc_to_hel(p_orbits)
    jac1 = xyz_sph_jac(p_x_hel).T
    rel_x = p_orbits - s_orbit

    R = np.sqrt(np.sum(rel_x[...,:3]**2, axis=-1))
    V = np.sqrt(np.sum(rel_x[...,3:]**2, axis=-1))
    lnR = np.log(R)
    lnV = np.log(V)

    sigma_r = 0.55
    mu_r = np.log(r_tide)
    r_term = -0.5*(2*np.log(sigma_r) + ((lnR-mu_r)/sigma_r)**2) - np.log(R**3)

    sigma_v = 0.8
    mu_v = np.log(v_disp)
    v_term = -0.5*(2*np.log(sigma_v) + ((lnV-mu_v)/sigma_v)**2) - np.log(V**3)

    ##
    ll = r_term + v_term + jac1
    return np.sum(ll, axis=0) / ll.shape[0]
    ##

    return r_term + v_term + jac1
