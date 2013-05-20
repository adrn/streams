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


def _nbody_acceleration(G, R, M):
    a = np.zeros_like(R)
    for ii in range(R.shape[0]):
        other_r = np.delete(R, ii, axis=0)
        other_m = np.delete(M, ii, axis=0)[...,np.newaxis]
        
        rr = R[ii] - other_r
        num = (-G * other_m * rr).sum(axis=0)
        denom = (np.sum(rr**2,axis=0)**1.5).sum(axis=0)
        a[ii] = num / denom
    
    return a

def nbody_integrate(particle_collection, time_steps, merge_length):
    """ TODO """
    
    dt = (time_steps[1]-time_steps[0])\
            .decompose(bases=particle_collection._units.values()).value
    Nsteps = len(time_steps)
    
    r_i = particle_collection._r
    v_i = particle_collection._v
    m = particle_collection._m
    _G = G.decompose(bases=particle_collection._units.values()).value
    
    # Shape of final object should be (Ntimesteps, Nparticles, Ndim)
    rs = np.zeros((Nsteps,) + r_i.shape, dtype=np.float64)
    vs = np.zeros((Nsteps,) + v_i.shape, dtype=np.float64)
    
    a_ip1 = _nbody_acceleration(_G, r_i, m)
    for ii in range(Nsteps):
        a_i = a_ip1
        
        r_ip1 = r_i + v_i*dt + 0.5*a_i*dt*dt
        a_ip1 = _nbody_acceleration(_G, r_ip1, m)
        v_ip1 = v_i + 0.5*(a_i + a_ip1)*dt
        
        rs[ii,:,:] = r_i
        vs[ii,:,:] = v_i

        r_i = r_ip1
        v_i = v_ip1
    
    return rs, vs
