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
from matplotlib import rc_context

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
    
    rcparams = {'lines.linestyle' : '-', 
                'lines.linewidth' : 1.,
                'lines.color' : 'k',
                'lines.marker' : None,
                'text.usetex' : True,
                'axes.edgecolor' : '#444444'}
    
    # Sample metallicities from: http://arxiv.org/pdf/1211.7073v1.pdf
    fe_h = np.random.normal(-1.67, 0.3, size=10000)
    fe_h = np.append(fe_h, np.random.normal(-2.33, 0.3, size=2000))
    
    # 
    
    # Distance from 1kpc to ~100kpc
    D = np.logspace(0., 2., 50)*u.kpc
    
    # Compute the apparent magnitude as a function of distance
    M_V = rrl_M_V(fe_h=-1.5)[0]
    m_V = apparent_magnitude(M_V, D)
    
    # Distance error
    dp = parallax_error(m_V, rrl_V_minus_I).arcsecond
    dD = D.to(u.pc).value**2 * dp * u.pc
    
    # Velocity error
    dpm = proper_motion_error(m_V, rrl_V_minus_I)
    dVtan = (dpm*D).to(u.km*u.radian/u.s).value
    
    with rc_context(rc=rcparams):
        fig,axes = plt.subplots(2, 1, figsize=(12, 12), sharex=True)
        
        # Plot Gaia distance errors
        axes[0].loglog(D.kiloparsec, (dD/D).decompose())
            
        # Plot tangential velocity errors
        axes[1].loglog(D.kiloparsec, dVtan)
    
    fig.subplots_adjust(hspace=0.1)
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