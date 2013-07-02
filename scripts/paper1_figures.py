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
import matplotlib
from matplotlib import rc_context
from matplotlib.patches import Rectangle

from streams.observation import apparent_magnitude
from streams.observation.gaia import parallax_error, proper_motion_error, \
                                     add_uncertainties_to_particles
from streams.observation.rrlyrae import rrl_M_V, rrl_V_minus_I

from streams.potential import LawMajewski2010
from streams.integrate.satellite_particles import SatelliteParticleIntegrator
from streams.data import lm10_particles, lm10_satellite, lm10_time

matplotlib.rc('xtick', labelsize=14)
matplotlib.rc('ytick', labelsize=14)
matplotlib.rc('text', usetex=True)
matplotlib.rc('axes', edgecolor='#444444', labelsize=20)
matplotlib.rc('lines', markeredgewidth=0)

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
    
    with rc_context(rc=rcparams):
        fig,axes = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
        
        # Distance from 1kpc to ~100kpc
        D = np.logspace(0., 2., 50)*u.kpc
        
        # Sample metallicities from: http://arxiv.org/pdf/1211.7073v1.pdf
        fe_hs = np.random.normal(-1.67, 0.3, size=50)
        fe_hs = np.append(fe_hs, np.random.normal(-2.33, 0.3, size=len(fe_hs)//5))
        
        for fe_h in fe_hs:
            # Johnson/Cousins (V - I_C) color for RR Lyrae at *minimum*
            # Guldenschuh et al. (2005 PASP 117, 721), pg. 725
            rrl_V_minus_I = np.random.normal(0.579, 0.006)
            
            # Compute the apparent magnitude as a function of distance
            M_V = rrl_M_V(fe_h=fe_h)[0]
            m_V = apparent_magnitude(M_V, D)
            
            # Distance error
            dp = parallax_error(m_V, rrl_V_minus_I).arcsecond
            dD = D.to(u.pc).value**2 * dp * u.pc
            
            # Velocity error
            dpm = proper_motion_error(m_V, rrl_V_minus_I)
            dVtan = (dpm*D).to(u.km*u.radian/u.s).value
        
            # Plot Gaia distance errors
            axes[0].loglog(D.kiloparsec, (dD/D).decompose(), color='k', alpha=0.1)
                
            # Plot tangential velocity errors
            axes[1].loglog(D.kiloparsec, dVtan, color='k', alpha=0.1)
    
        # Add spitzer 2% line to distance plot
        axes[0].axhline(0.02, linestyle='--', linewidth=3, color='#998EC3')
    
    # Now add rectangles for Sgr, Orphan
    sgr_d = Rectangle((10., 5./10.), 60., (15/60 - 5/10), 
                      color='#67A9CF', alpha=0.75, label='Sgr')
    axes[0].add_patch(sgr_d)
    
    # From fig. 3 in http://mnras.oxfordjournals.org/content/389/3/1391.full.pdf+html
    orp_d = Rectangle((10., 2./10.), 35., (10/35. - 2./10),
                      color='#EF8A62', alpha=0.75, label='Orp')
    axes[0].add_patch(orp_d)
    
    # ??
    sgr_v = Rectangle((10., 12), 60., 1., color='#67A9CF', alpha=0.75)
    axes[1].add_patch(sgr_v)
    
    orp_v = Rectangle((10., 10.), 35., 1., color='#EF8A62', alpha=0.75)
    axes[1].add_patch(orp_v)
    
    axes[0].set_ylabel("Frac. distance error $\sigma_D/D$")
    axes[1].set_ylabel("$v_{tan}$ error [km/s]")
    axes[1].set_xlabel("Distance [kpc]")
    axes[1].set_xticklabels(["1", "10", "100"])
    
    axes[0].legend(loc='upper left')
    fig.subplots_adjust(hspace=0.1)
    plt.tight_layout()
    plt.show()

def lm10_particles_selection():
    """ Top-down plot of Sgr particles, with selected stars and then 
        re-observed
    """
    
    np.random.seed(42)
    all_particles = lm10_particles(N=0, expr="(Pcol<7) & (abs(Lmflag)==1)")
    particles = lm10_particles(N=100, expr="(Pcol<7) & (Pcol>0) & (abs(Lmflag)==1)")
    err_particles = add_uncertainties_to_particles(particles, 
                                                   radial_velocity_error=15.*u.km/u.s,
                                                   distance_error_percent=2.)
    
    rcparams = {'lines.linestyle' : 'none', 
                'lines.color' : 'k',
                'lines.marker' : '.'}
    
    with rc_context(rc=rcparams): 
        fig,ax = plt.subplots(1, 1, figsize=(10,10))
        ax.plot(all_particles._r[:,0], all_particles._r[:,2],
                color='#666666', markersize=3, alpha=0.15, marker='o')
        ax.plot(particles._r[:,0], particles._r[:,2],
                color='k', markersize=10, alpha=1., marker='.')
        ax.plot(err_particles._r[:,0], err_particles._r[:,2],
                color='#CA0020', markersize=10, alpha=1., marker='.')

        ax.set_xlabel("$X_{GC}$ [kpc]")
        ax.set_ylabel("$Z_{GC}$ [kpc]")
    
    #ax.scatter()
    plt.tight_layout()
    plt.show()
    
if __name__ == '__main__':
    #gaia_spitzer_errors()
    lm10_particles_selection()