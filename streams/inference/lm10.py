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
from ..potential.lm10 import LawMajewski2010, true_params, param_ranges, param_units
from ..nbody import Orbit, OrbitCollection
from ..integrate import leapfrog, adaptive_leapfrog

__all__ = ["ln_posterior", "ln_likelihood"]

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

# Note: if this doesn't work, I can always pass param_names in to each prior
#   and if it isn't in there, return 0...

def ln_likelihood(p, param_names, particles, satellite, t1, t2, resolution):
    """ Evaluate the likelihood function for a given set of halo 
        parameters.
    """
    halo_params = dict(zip(param_names, p))
    
    # LawMajewski2010 contains a disk, bulge, and logarithmic halo 
    lm10 = LawMajewski2010(**halo_params)
    
    t,r,v = adaptive_leapfrog(lm10._acceleration_at, 
                      satellite._r, satellite._v,
                      t1=t1, t2=t2, resolution=resolution)
    satellite_orbit = OrbitCollection(t=t*u.Myr, 
                                      r=r*satellite.r.unit,
                                      v=v*satellite.v.unit,
                                      m=[2.5E8]*u.M_sun,
                                      units=[u.kpc, u.Myr, u.M_sun])
    
    t,r,v = leapfrog(lm10._acceleration_at, 
                     particles._r, particles._v,
                     t=t)
    particle_orbits = OrbitCollection(t=t*u.Myr, 
                                      r=r*particles.r.unit, 
                                      v=v*particles.v.unit, 
                                      m=np.ones(len(r))*u.M_sun,
                                      units=[u.kpc, u.Myr, u.M_sun])
    
    return -generalized_variance(lm10, particle_orbits, satellite_orbit)

def ln_posterior(p, *args):
    param_names, particles, satellite, t1, t2, resolution = args
    return ln_prior(p, param_names) + ln_likelihood(p, *args)