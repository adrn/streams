# coding: utf-8

""" A script for making figures for our Streams Paper 1 """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys

# Third-party
import astropy.units as u
import numpy as np
import matplotlib.pyplot as plt

from streams.data.gaia import parallax_error, proper_motion_error, \
                              rr_lyrae_V_minus_I, rr_lyrae_M_V, \
                              apparent_magnitude
from streams.potential import LawMajewski2010
from streams.integrate.satellite_particles import SatelliteParticleIntegrator
from streams.data import lm10_particles, lm10_satellite, lm10_time

def normed_objective_plot():
    """ Plot our objective function in each of the 4 parameters we vary """
    
    # Can change this to the true adaptive functions so I can compare
    timestep2 = lambda *args,**kwargs: -1.
    resolution = 1.
    
    # Read in the LM10 data
    particles = lm10_particles(N=100, expr="(Pcol>0) & (abs(Lmflag)==1)")
    satellite = lm10_satellite()
    t1,t2 = lm10_time()
    resolution = 3.
    
    variances = dict()
    for param in ['q1','qz','v_halo','phi']:
        if not variances.has_key(param):
            variances[param] = []
            
        stats = np.linspace(true_params[param]*0.9,
                            true_params[param]*1.1, 
                            10)
    
        for stat in stats:
            params = true_params.copy()
            params[param] = stat
            lm10 = LawMajewski2010(**params)
            integrator = SatelliteParticleIntegrator(lm10, satellite, particles)
            s_orbit,p_orbits = integrator.run(timestep_func=timestep2,
                                      timestep_args=(lm10, satellite.m.value),
                                      resolution=resolution,
                                      t1=t1, t2=t2)
            variances[param].append(generalized_variance_prod(lm10, p_orbits, s_orbit))