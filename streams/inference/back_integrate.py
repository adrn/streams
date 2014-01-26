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

    gc = _hel_to_gc(hel)
    x,y,z,vx,vy,vz = gc.T

    row0 = np.zeros_like(hel.T)
    row0[0] = -d*sinl*cosb
    row0[1] = -d*cosl*sinb
    row0[2] = cosl*cosb

    row1 = np.zeros_like(hel.T)
    row1[0] = d*cosl*cosb
    row1[1] = -d*sinl*sinb
    row1[2] = sinl*cosb

    row2 = np.zeros_like(hel.T)
    row2[0] = 0.
    row2[1] = -d*cosb
    row2[2] = sinb

    row3 = [-vr*cosb*sinl + mul*d*cosb*cosl - mub*d*sinb*sinl,
            -vr*sinb*cosl - mul*d*sinb*sinl + mub*d*cosb*cosl,
            cosb*sinl*mul + sinb*cosl*mub,
            d*cosb*sinl,
            d*sinb*cosl,
            cosb*cosl]

    row4 = [vr*cosb*cosl + mul*d*cosb*sinl + mub*d*sinb*cosl,
            -vr*sinb*sinl + mul*d*sinb*cosl + mub*d*cosb*sinl,
            -cosb*cosl*mul + sinb*sinl*mub,
            -d*cosb*cosl,
            d*sinb*sinl,
            cosb*sinl]

    row5 = np.zeros_like(hel.T)
    row5[0] = 0.
    row5[1] = cosb*vr + d*sinb*mub
    row5[2] = -cosb*mub
    row5[3] = 0.
    row5[4] = -d*cosb
    row5[5] = sinb

    return np.array([row0, row1, row2, row3, row4, row5]).T

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

    # Gaussian shell idea:
    p_x = np.array([p_orbits[jj,ii] for ii,jj in enumerate(t_idx)])
    s_x = np.array([s_orbit[jj,0] for jj in t_idx])
    rel_x = p_x-s_x

    p_x_hel = _gc_to_hel(p_x)
    J1 = xyz_sph_jac(p_x_hel)
    jac1 = np.array([np.linalg.slogdet(np.linalg.inv(j))[1] for j in J1])
    #J2 = lnRV_xyz_jac(rel_x)
    #jac2 = np.array([np.linalg.slogdet(np.linalg.inv(j))[1] for j in J2])

    r_tide = potential._tidal_radius(2.5e8, s_x)
    v_esc = potential._escape_velocity(2.5e8, r_tide=r_tide)
    v_disp = 0.017198632325

    R = np.sqrt(np.sum(rel_x[...,:3]**2, axis=-1))
    V = np.sqrt(np.sum(rel_x[...,3:]**2, axis=-1))
    lnR = np.log(R)
    lnV = np.log(V)

    v = 1.
    sigma_r = np.sqrt(np.log(1 + v/r_tide**2))
    mu_r = np.log(r_tide**2 / np.sqrt(v + r_tide**2))
    r_term = -0.5*(2*np.log(sigma_r) + ((lnR-mu_r)/sigma_r)**2) - np.log(R**3)

    v = v_disp
    sigma_v = np.sqrt(np.log(1 + v/v_esc**2))
    mu_v = np.log(v_esc**2 / np.sqrt(v + v_esc**2))
    v_term = -0.5*(2*np.log(sigma_v) + ((lnV-mu_v)/sigma_v)**2) - np.log(V**3)

    return np.sum(r_term + v_term) + np.sum(jac1)
