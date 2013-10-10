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
from ..potential.lm10 import LawMajewski2010, true_params, param_units
from ..dynamics import OrbitCollection
from ..integrate.satellite_particles import SatelliteParticleIntegrator
from .core import objective, objective2

__all__ = ["ln_posterior", "ln_likelihood", "objective"]

# Parameter ranges to initialize the walkers over
# v_halo range comes from 5E11 < M < 5E12, current range of MW mass @ 200 kpc
param_ranges = dict(v_halo=((100.*u.km/u.s).to(u.kpc/u.Myr).value,
                            (330.*u.km/u.s).to(u.kpc/u.Myr).value),
                    q1=(1.,2.),
                    q2=(0.5,2.),
                    qz=(1.0,2.),
                    phi=(np.pi/4, 3*np.pi/4),
                    r_halo=(8,20)) # kpc

def ln_likelihood(p, param_names, particles, satellite, t1, t2, resolution):
    """ Evaluate the likelihood function for a given set of halo 
        parameters.
    """
    halo_params = dict(zip(param_names, p))
    
    # LawMajewski2010 contains a disk, bulge, and logarithmic halo 
    lm10 = LawMajewski2010(**halo_params)
    integrator = SatelliteParticleIntegrator(lm10, satellite, particles)
    
    # not adaptive: s_orbit,p_orbits = integrator.run(t1=t1, t2=t2, dt=-1.)
    s_orbit,p_orbits = integrator.run(t1=t1, t2=t2, dt=-1.)
    
    # v_disp measured from still-bound LM10 particles
    return objective2(lm10, s_orbit, p_orbits, v_disp=0.0133)

StatisticalModel()

# TODO:
# - merge infer_potential stuff in to StatisticalModel.run()
# - update all tests
# - Make likelihood a general "back integration likelihood"
# - likelihood should accept a potential, so it's generic. maybe specify in config file?
# - SatelliteParticleIntegrator is dumb and should be something else. maybe function?
# - adapative timestep stuff?