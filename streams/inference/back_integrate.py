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

def back_integration_likelihood(t1, t2, dt, potential, p_hel, s_hel, logm0, tub, tail_bit):

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

    s_orbit = np.vstack((rs[:,0][:,np.newaxis].T, vs[:,0][:,np.newaxis].T)).T
    p_orbits = np.vstack((rs[:,1:].T, vs[:,1:].T)).T

    t_idx = np.array([np.argmin(np.fabs(ts - t)) for t in tub])
    p_orbits = np.array([p_orbits[jj,ii] for ii,jj in enumerate(t_idx)])
    s_orbit = np.array([s_orbit[jj,0] for jj in t_idx])

    s_mass = np.exp(logm0)
    #r_tide = potential._tidal_radius(s_mass, s_orbit)*0.69336 #*1.4
    r_tide = potential._tidal_radius(s_mass, s_orbit)*1.4

    p_x_hel = _gc_to_hel(p_orbits)
    jac1 = _xyz_sph_jac(p_x_hel)

    s_R_orbit = np.sqrt(np.sum(s_orbit[...,:3]**2, axis=-1))
    a_pm = (s_R_orbit + r_tide*tail_bit) / s_R_orbit

    f = r_tide / s_R_orbit
    vdisp = np.sqrt(np.sum(s_orbit[...,3:]**2, axis=-1)) * f / 1.4
    #vdisp = np.sqrt(np.sum(s_orbit[...,3:]**2, axis=-1)) * (s_mass/(3*M_enc))**(0.33333333)

    sigma_r = r_tide
    R = p_orbits[:,:3] - (a_pm[:,np.newaxis]*s_orbit[:,:3])
    r_term = -0.5*(6*np.log(sigma_r) + np.sum((R/sigma_r[...,np.newaxis])**2,axis=-1))

    sigma_v = vdisp
    V = p_orbits[:,3:] - s_orbit[...,3:]
    v_term = -0.5*(6*np.log(sigma_v) + np.sum((V/sigma_v[...,np.newaxis])**2,axis=-1))

    return (r_term + v_term + jac1)

    #return logsumexp(r_term + v_term + jac1, axis=0)
