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

from streams.observation import apparent_magnitude
from streams.observation.gaia import parallax_error, proper_motion_error
from streams.observation.rrlyrae import rrl_M_V, rrl_V_minus_I

from streams.potential import LawMajewski2010
from streams.integrate.satellite_particles import SatelliteParticleIntegrator
from streams.data import lm10_particles, lm10_satellite, lm10_time

def gaia_spitzer_errors():
    """ Visualize the observational errors from Gaia and Spitzer, along with
        dispersion and distance scale of Sgr and Orphan. 
    """
    
    # Distance from 1kpc to ~100kpc
    D = np.logspace(0., 2., 50)*u.kpc
    
    # Compute the apparent magnitude as a function of distance
    m_V = apparent_magnitude(rr_lyrae_M_V, D)
    
    fig,axes = plt.subplots(2, 1, figsize=(12, 12))
    
    # Plot Gaia distance errors
    dp = parallax_error(m_V, rr_lyrae_V_minus_I).arcsecond
    dD = D.to(u.pc).value**2 * dp * u.pc
    axes[0].loglog(D, (dD/D).decompose(), color="k", linewidth=1, alpha=0.5)
        
    # Plot tangential velocity errors
    dpm = proper_motion_error(m_V, rr_lyrae_V_minus_I)
    dVtan = (dpm*D).to(u.km*u.radian/u.s).value
    axes[1].loglog(D, dVtan, color="k", linewidth=1, alpha=0.5)
    
    plt.show()

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

if __name__ == '__main__':
    gaia_spitzer_errors()