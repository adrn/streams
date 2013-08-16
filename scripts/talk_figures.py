# coding: utf-8

""" A script for making figures for our Streams Paper 1 """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys
import cPickle as pickle

# Third-party
import astropy.units as u
from astropy.table import Table, Column
from astropy.io import ascii
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import rc_context, rcParams, cm
from matplotlib.patches import Rectangle, Ellipse

from streams.util import project_root
from streams.observation import apparent_magnitude
from streams.observation.gaia import parallax_error, proper_motion_error, \
                                     add_uncertainties_to_particles
from streams.observation.rrlyrae import rrl_M_V, rrl_V_minus_I

from streams.inference import relative_normalized_coordinates, generalized_variance, minimum_distance_matrix
from streams.inference.lm10 import timestep
from streams.potential import LawMajewski2010
from streams.potential.lm10 import true_params, _true_params, param_to_latex
from streams.integrate.satellite_particles import SatelliteParticleIntegrator
from streams.io.lm10 import particle_table, particles_today, satellite_today, time

matplotlib.rc('xtick', labelsize=24, direction='in', )
matplotlib.rc('ytick', labelsize=24, direction='in')
#matplotlib.rc('text', usetex=True)
matplotlib.rc('axes', edgecolor='#444444', labelsize=24, labelweight=400, linewidth=2.0)
matplotlib.rc('lines', markeredgewidth=0)
matplotlib.rc('font', family='Source Sans Pro', weight=400)
#matplotlib.rc('savefig', bbox='standard')

plot_path = "plots/talks/"
if not os.path.exists(plot_path):
    os.mkdir(plot_path)
    
# Read in the LM10 data
np.random.seed(142)
particles = particles_today(N=100, expr="(Pcol>-1) & (abs(Lmflag) == 1) & (Pcol < 8)")
satellite = satellite_today()
t1,t2 = time()    

def gaia_spitzer_errors():
    """ Visualize the observational errors from Gaia and Spitzer, along with
        dispersion and distance scale of Sgr and Orphan. 
    """
    
    rcparams = {'lines.linestyle' : '-', 
                'lines.linewidth' : 1.,
                'lines.color' : 'k',
                'lines.marker' : None,
                'axes.facecolor' : '#ffffff',
                'xtick.major.size' : 10, 'xtick.minor.size' : 6, 'xtick.major.pad' : 8,
                'ytick.major.size' : 10, 'ytick.minor.size' : 6, 'ytick.major.pad' : 8}
    
    with rc_context(rc=rcparams):
        fig,ax = plt.subplots(1, 1, figsize=(10, 6))
        
        # Distance from 1kpc to ~100kpc
        D = np.logspace(0., 2.1, 50)*u.kpc
        
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
            
            # Velocity error
            dpm = proper_motion_error(m_V, rrl_V_minus_I)
            dVtan = (dpm*D).to(u.km*u.radian/u.s).value
                
            # Plot tangential velocity errors
            ax.loglog(D.kiloparsec, dVtan, color='#666666', alpha=0.1)
    
    ax.set_xlim(9, 125)
    ax.set_ylim(0.5, 125)
    
    ax.set_xticks([10, 20, 60, 100])
    ax.set_xticklabels(["{:g} kpc".format(xt) for xt in ax.get_xticks()])
    ax.set_yticklabels(["{:g} km/s".format(yt) for yt in ax.get_yticks()])
    
    #plt.tight_layout()
    plt.savefig(os.path.join(plot_path, "gaia.pdf"))

def dump_gaia_csv():
    # Distance from 1kpc to ~100kpc
    D = np.logspace(0., 2.1, 50)*u.kpc
    
    # Sample metallicities from: http://arxiv.org/pdf/1211.7073v1.pdf
    fe_hs = np.random.normal(-1.67, 0.3, size=50)
    fe_hs = np.append(fe_hs, np.random.normal(-2.33, 0.3, size=len(fe_hs)//5))
    
    # 50 kpc
    rrl_V_minus_I = 0.579
    M_V = rrl_M_V(fe_h=-1.67)[0]
    DD = 35.*u.kpc
    m_V = apparent_magnitude(M_V, DD)
    dpm = proper_motion_error(m_V, rrl_V_minus_I)
    dVtan = (dpm*DD).to(u.km*u.radian/u.s).value
    print("{1} kpc: {0}".format(dVtan, DD))
    
    t = Table()
    
    avg_dVtan = np.zeros_like(D.value)
    for fe_h in fe_hs:
        # Johnson/Cousins (V - I_C) color for RR Lyrae at *minimum*
        # Guldenschuh et al. (2005 PASP 117, 721), pg. 725
        rrl_V_minus_I = np.random.normal(0.579, 0.006)
        
        # Compute the apparent magnitude as a function of distance
        M_V = rrl_M_V(fe_h=fe_h)[0]
        m_V = apparent_magnitude(M_V, D)
        
        # Velocity error
        dpm = proper_motion_error(m_V, rrl_V_minus_I)
        dVtan = (dpm*D).to(u.km*u.radian/u.s).value
        
        avg_dVtan += dVtan
    
    avg_dVtan /= len(fe_hs)
    c1 = Column(data=D.value, dtype=float, name='d{0}'.format(fe_h))
    c2 = Column(data=avg_dVtan, dtype=float, name='{0}'.format(fe_h))
    t.add_column(c1)
    t.add_column(c2)
    
    ascii.write(t, os.path.join(plot_path, "gaia.csv"), Writer=ascii.Basic, delimiter=',')

if __name__ == '__main__':
    #gaia_spitzer_errors()
    dump_gaia_csv()