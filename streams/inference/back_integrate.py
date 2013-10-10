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

from ..inference import generalized_variance
from ..integrate.satellite_particles import SatelliteParticleIntegrator

__all__ = ["back_integrate_likelihood", "variance_likelihood"]


def back_integrate_likelihood(p, param_names, particles, satellite, 
                              Potential, t1, t2):
    """ Evaluate the TODO """

    model_params = dict(zip(param_names, p))
    potential = Potential(**model_params)

    integrator = SatelliteParticleIntegrator(potential, satellite, particles)
    
    # not adaptive:
    s_orbit,p_orbits = integrator.run(t1=t1, t2=t2, dt=-1.)
    
    # TODO: old one used generalized_variance, new one uses??
    return #objective2(lm10, s_orbit, p_orbits, v_disp=0.0133)

def variance_likelihood(p, param_names, particles, satellite, 
                        Potential, t1, t2):
    """ Evaluate the TODO """

    model_params = dict(zip(param_names, p))
    potential = Potential(**model_params)

    integrator = SatelliteParticleIntegrator(potential, satellite, particles)
    
    # not adaptive:
    s_orbit,p_orbits = integrator.run(t1=t1, t2=t2, dt=-1.)
    
    return -generalized_variance(lm10, s_orbit, p_orbits)