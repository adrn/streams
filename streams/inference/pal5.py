# coding: utf-8

""" Contains priors and likelihood functions for inferring parameters of
    the Logarithmic potential using back integration.
"""

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys
import math

# Third-party
import numpy as np
import astropy.units as u

from ..inference import generalized_variance
from ..potential.pal5 import Palomar5, true_params
from ..dynamics import OrbitCollection
from ..integrate.satellite_particles import SatelliteParticleIntegrator
from .core import objective, objective2

__all__ = ["ln_posterior", "ln_likelihood"]

# Parameter ranges to initialize the walkers over
param_ranges = dict(log_m=(26.93787, 29.24046),
                    qz=(0.707,1.2),
                    Rs=(10.,45.))

def ln_p_qz(qz):
    """ Flat prior on vertical (z) axis flattening parameter. """
    lo,hi = param_ranges["qz"]
    
    if qz <= lo or qz >= hi:
        return -np.inf
    else:
        return 0.

def ln_p_log_m(log_m):
    """ Flat prior on mass of the halo
    """
    lo,hi = param_ranges["log_m"]
    
    if log_m <= lo or log_m >= hi:
        return -np.inf
    else:
        return 0.
        
def ln_p_Rs(Rs):
    """ Flat prior on the scale length of the halo
    """
    lo,hi = param_ranges["Rs"]
    
    if Rs <= lo or Rs >= hi:
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

def ln_likelihood(p, param_names, particles, satellite, t1, t2, resolution):
    """ Evaluate the likelihood function for a given set of halo 
        parameters.
    """
    halo_params = dict(zip(param_names, p))
    
    # LawMajewski2010 contains a disk, bulge, and logarithmic halo 
    potential = Palomar5(**halo_params)
    
    integrator = SatelliteParticleIntegrator(potential, satellite, particles)
    
    # not adaptive: s_orbit,p_orbits = integrator.run(t1=t1, t2=t2, dt=-1.)
    s_orbit,p_orbits = integrator.run(t1=t1, t2=t2, dt=-1.)
    
    # v_disp from Andreas
    return objective2(potential, s_orbit, p_orbits, v_disp=0.0010)

def ln_posterior(p, *args):
    param_names, particles, satellite, t1, t2, resolution = args
    return ln_prior(p, param_names) + ln_likelihood(p, *args)