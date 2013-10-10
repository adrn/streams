# coding: utf-8

""" Direct N-body """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys

# Third-party
import numpy as np
import astropy.units as u
from astropy.constants import G

__all__ = ["nbody_integrate"]

def _nbody_acceleration(G, R, M, e=0.1):
    a = np.zeros_like(R)
    for ii in range(R.shape[0]):
        other_r = np.delete(R, ii, axis=0)
        other_m = np.delete(M, ii, axis=0)[...,np.newaxis]
        
        rr = R[ii] - other_r
        num = (-G * other_m * rr).sum(axis=0)
        denom = (np.sqrt(np.sum(rr**2,axis=0)) + e**2).sum(axis=0)**1.5
        a[ii] = num / denom
    
    return a

def nbody_integrate(particles, time_steps, 
                    external_acceleration=None, e=0.1):
    """ Direct N-body integration of the given ParticleCollection over the 
        specified time steps. Softening length is specified by 'e'. 
        
        Parameters
        ----------
    """
    
    dt = (time_steps[1]-time_steps[0])\
            .decompose(bases=particles.units).value
    Nsteps = len(time_steps)
    
    r_i = particles._r
    v_i = particles._v
    m = particles._m
    _G = G.decompose(bases=particles.units).value
    
    # Shape of final object should be (Ntimesteps, Nparticles, Ndim)
    rs = np.zeros((Nsteps,) + r_i.shape, dtype=np.float64)
    vs = np.zeros((Nsteps,) + v_i.shape, dtype=np.float64)
    
    if external_acceleration is not None:
        acc = lambda _G, r_i, m, e: _nbody_acceleration(_G, r_i, m, e) + \
                                        external_acceleration(r_i)
    else:
        acc = _nbody_acceleration
    
    a_ip1 = acc(_G, r_i, m, e)
    for ii in range(Nsteps):
        a_i = a_ip1
        
        r_ip1 = r_i + v_i*dt + 0.5*a_i*dt*dt
        # TODO: try with multiprocessing map here, if pool is supplied?
        a_ip1 = acc(_G, r_ip1, m, e)
        v_ip1 = v_i + 0.5*(a_i + a_ip1)*dt
        
        rs[ii,:,:] = r_i
        vs[ii,:,:] = v_i

        r_i = r_ip1
        v_i = v_ip1
    
    return rs, vs
