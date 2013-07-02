# coding: utf-8

""" Contains priors and likelihood functions for inferring parameters of
    the Logarithmic potential using back integration.
"""

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys

# Third-party
import numpy as np
import astropy.units as u

from ..inference import generalized_variance
from ..potential.lm10 import LawMajewski2010, true_params, param_units
from ..nbody import OrbitCollection
from ..integrate.satellite_particles import SatelliteParticleIntegrator

__all__ = ["ln_posterior", "ln_likelihood"]

# Parameter ranges to initialize the walkers over
# v_halo range comes from 5E11 < M < 5E12, current range of MW mass @ 200 kpc
param_ranges = dict(v_halo=((100.*u.km/u.s).to(u.kpc/u.Myr).value,
                            (330.*u.km/u.s).to(u.kpc/u.Myr).value),
                    q1=(1.,2.),
                    q2=(0.5,1.5),
                    qz=(1.0,2.),
                    phi=(np.pi/4, 3*np.pi/4),
                    r_halo=(8,20)) # kpc

def ln_p_qz(qz):
    """ Flat prior on vertical (z) axis flattening parameter. """
    lo,hi = param_ranges["qz"]
    
    if qz <= lo or qz >= hi:
        return -np.inf
    else:
        return 0.

def ln_p_q1(q1):
    """ Flat prior on x-axis flattening parameter. """
    lo,hi = param_ranges["q1"]
    
    if q1 <= lo or q1 >= hi:
        return -np.inf
    else:
        return 0.

def ln_p_q2(q2):
    """ Flat prior on y-axis flattening parameter. """
    lo,hi = param_ranges["q2"]
    
    if q2 <= lo or q2 >= hi:
        return -np.inf
    else:
        return 0.

def ln_p_v_halo(v):
    """ Flat prior on mass of the halo (v_halo). The range imposed is 
        roughly a halo mass between 10^10 and 10^12 M_sun at 200 kpc.
    """
    lo,hi = param_ranges["v_halo"]
    
    if v <= lo or v >= hi:
        return -np.inf
    else:
        return 0.

def ln_p_phi(phi):
    """ Flat prior on orientation angle between DM halo and disk. """
    lo,hi = param_ranges["phi"]
    
    if phi < lo or phi > hi:
        return -np.inf
    else:
        return 0.

def ln_p_r_halo(r_halo):
    """ Flat prior on halo concentration parameter. """
    lo,hi = param_ranges["r_halo"]
    
    if r_halo < lo or r_halo > hi:
        return -np.inf
    else:
        return 0.

def ln_prior(p, param_names):
    """ Joint prior over all parameters. """
    
    sum = 0.
    for ii,param in enumerate(param_names):
        f = globals()["ln_p_{0}".format(param)]
        sum += f(p[ii])
    
    return sum

def timestep(r, v, potential, m_sat):
    R_tide = potential._tidal_radius(r=r[0], m=m_sat)
    #v_max = np.max(np.sqrt(np.sum((v[1:]-v[0])**2,axis=-1)))
    v_max = np.max(np.sqrt(np.sum(v**2,axis=-1)))
    dt = -(R_tide / v_max)
    
    if dt > -0.5:
        return -0.5
    else:
        return dt

def ln_likelihood(halo_p, halo_p_names, particles, satellite, t1, t2, resolution):
    """ Evaluate the likelihood function for a given set of halo 
        parameters.
    """
    halo_params = dict(zip(halo_p_names, halo_p))
    
    # LawMajewski2010 contains a disk, bulge, and logarithmic halo 
    lm10 = LawMajewski2010(**halo_params)
    
    integrator = SatelliteParticleIntegrator(lm10, satellite, particles)
    
    # not adaptive: s_orbit,p_orbits = integrator.run(t1=t1, t2=t2, dt=-1.)
    s_orbit,p_orbits = integrator.run(timestep_func=timestep,
                                      timestep_args=(lm10, satellite.m.value),
                                      resolution=resolution,
                                      t1=t1, t2=t2)
    
    r_tide = lm10._tidal_radius(m=s_orbit._m,
                                r=s_orbit._r)[:,:,np.newaxis]
    v_esc = lm10._escape_velocity(m=s_orbit._m,
                                  r_tide=r_tide)
    
    ys = np.zeros((len(particles),len(r_tide)))
    for jj in range(len(r_tide)):
        ys[:,jj] = model(p_orbits, s_orbit, r_tide, v_esc, jj)
        
    ls = []
    for ii in range(len(particles)):
        ys = np.zeros((len(particles),len(r_tide)))
        for jj in range(len(r_tide)):
            ys[:,jj] = model(p_orbits, s_orbit, r_tide, v_esc, 0)
        
        integral = integrate.simps(ys, x=s_orbit._t)
        print integral, ys.shape
        return
    
    l = np.sum(ls)
    return l

def ln_posterior(p, *args):
    param_names, particles, satellite, t1, t2, resolution = args
    return ln_prior(p, param_names) + ln_likelihood(p, *args)