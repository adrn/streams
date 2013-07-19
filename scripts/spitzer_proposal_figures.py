# coding: utf-8

""" A script for making figures for the Spitzer proposal """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys

# Third-party
from astropy.io import ascii
import astropy.units as u
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import rc_context, rcParams
from matplotlib.patches import Rectangle
from matplotlib import cm

from streams.util import project_root
from streams.coordinates import ra_dec_dist_to_xyz
from streams.observation import apparent_magnitude
from streams.observation.gaia import parallax_error, proper_motion_error, \
                                     add_uncertainties_to_particles
from streams.observation.rrlyrae import rrl_M_V, rrl_V_minus_I

from streams.inference import relative_normalized_coordinates
from streams.inference.lm10 import timestep
from streams.potential import LawMajewski2010
from streams.potential.lm10 import true_params
from streams.integrate.satellite_particles import SatelliteParticleIntegrator
from streams.io.lm10 import particle_table, particles_today, satellite_today, time
from streams.io.catalogs import read_stripe82
from streams.plot import sgr_kde

matplotlib.rc('xtick', labelsize=14)
matplotlib.rc('ytick', labelsize=14)
matplotlib.rc('text', usetex=True)
matplotlib.rc('axes', edgecolor='#444444', labelsize=20)
matplotlib.rc('lines', markeredgewidth=0)
matplotlib.rc('font', family='sans-serif')
rcParams['font.sans-serif'] = 'helvetica'

plot_path = "plots/spitzer_proposal/"
if not os.path.exists(plot_path):
    os.mkdir(plot_path)

stripe82_catalog = """ra dec dist
14.616653 -0.75063 21.78
23.470286 -0.75074 20.02
25.920219 0.507809 18.37
28.105067 -0.28684 10.56
27.339723 -0.222394 19.37
28.709058 0.250204 7.59
29.908223 1.222851 18.35
27.37402 0.155047 19.87
25.46789 -0.89094 8.5
33.55261 0.660465 12.87
30.691522 -0.000466 20.11
30.13114 0.994329 20.66
32.419304 -0.036812 18.26
32.717291 0.631938 16.35
37.29115 1.07193 17.18
35.274198 -0.647943 12.27
40.612944 1.225526 7.29
40.508969 0.665032 20.11
44.338191 -0.220803 17.3
46.057459 -1.220635 18.74
47.609341 0.461527 9.4
52.601027 1.209482 9.24
51.883696 0.064283 17.62
53.635031 0.026812 7.44
53.80621 1.068064 10.51
56.963075 -0.291889 11.96
59.037319 -0.08628 7.19
58.036886 0.999678 8.18"""
    
def gaia_spitzer_errors():
    """ Visualize the observational errors from Gaia and Spitzer, along with
        dispersion and distance scale of Sgr and Orphan. 
    """
    
    rcparams = {'lines.linestyle' : '-', 
                'lines.linewidth' : 1.,
                'lines.color' : 'k',
                'lines.marker' : None,
                'text.usetex' : True,
                'axes.edgecolor' : '#444444',
                'axes.facecolor' : '#ffffff'}
    
    sgr_color = '#67A9CF'
    orp_color = '#EF8A62'
    
    with rc_context(rc=rcparams):
        fig,axes = plt.subplots(1, 2, figsize=(12, 6))
        
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
        axes[0].axhline(0.02, linestyle='--', linewidth=4, color='#7B3294', alpha=0.8)
    
    # Now add rectangles for Sgr, Orphan
    sgr_d = Rectangle((10., 0.15), 60., 0.15, 
                      color=sgr_color, alpha=1., label='Sgr thickness')
    axes[0].add_patch(sgr_d)
    
    # From fig. 3 in http://mnras.oxfordjournals.org/content/389/3/1391.full.pdf+html
    orp_d = Rectangle((10., 0.03), 35., 0.03,
                      color=orp_color, alpha=1., label='Orp thickness')
    axes[0].add_patch(orp_d)
    
    # Dispersion from Majewski 2004: 10 km/s
    sgr_v = Rectangle((10., 10), 60., 1., color=sgr_color, alpha=0.75,
                      label='Sgr dispersion')
    axes[1].add_patch(sgr_v)
    
    orp_v = Rectangle((10., 8.), 35., 1., color=orp_color, alpha=0.75,
                      label='Orp dispersion')
    axes[1].add_patch(orp_v)
    
    axes[0].set_ylim(top=10.)
    axes[0].set_xlim(1, 100)
    
    axes[1].set_ylim(0.1, 100)
    axes[1].set_xlim(10, 100)
    
    axes[0].set_xlabel("Distance [kpc]")
    axes[0].set_ylabel("Frac. distance error $\sigma_D/D$")
    axes[1].set_ylabel("$v_{tan}$ error [km/s]")
    axes[1].set_xlabel("Distance [kpc]")
    
    axes[0].set_xticklabels(["1", "10", "100"])
    axes[1].set_xticklabels(["10", "100"])
    
    axes[0].set_yticklabels(["{:g}".format(yt) for yt in axes[0].get_yticks()])
    axes[1].set_yticklabels(["{:g}".format(yt) for yt in axes[1].get_yticks()])
    
    # add Gaia and Spitzer text to first plot
    axes[0].text(4., 0.12, 'Gaia', fontsize=16, rotation=45)
    axes[0].text(4., 0.011, 'Spitzer', fontsize=16, color="#7B3294", alpha=0.8)
    
    # add legends
    axes[0].legend(loc='upper left', fancybox=True)
    axes[1].legend(loc='upper left', fancybox=True)
    
    fig.subplots_adjust(hspace=0.1)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_path, "gaia.pdf"))

def top_down_sgr():
    """ Top-down plot of Sgr particles, with selected stars and then 
        re-observed
    """
    
    fig,ax = plt.subplots(1, 1)
    
    # read in all particles as a table
    pdata = particle_table(N=0, expr="(Pcol<7) & (abs(Lmflag)==1)")
    
    extent = {'x':(-90,55), 
              'y':(-50,60)}
    Z = sgr_kde(pdata, extent=extent)
    ax.imshow(Z**0.5, interpolation="nearest", 
              extent=extent['x']+extent['y'],
              cmap=cm.Blues, aspect=1)
    
    np.random.seed(42)
    particles = particles_today(N=100, expr="(Pcol<7) & (Pcol>0) & (abs(Lmflag)==1)")   
    
    # read in Stripe 82 RR Lyrae
    s82 = ascii.read(stripe82_catalog)
    stripe82_xyz = ra_dec_dist_to_xyz(s82['ra']*u.deg, s82['dec']*u.deg, 
                                      s82['dist']*u.kpc)
    
    catalina = ascii.read(os.path.join(project_root, "data/spitzer_sgr_sample.txt"))
    catalina_xyz = ra_dec_dist_to_xyz(catalina['ra']*u.deg, 
                                      catalina['dec']*u.deg, 
                                      catalina['dist']*u.kpc)
    
    rcparams = {'lines.linestyle' : 'none', 
                'lines.color' : 'k',
                'lines.marker' : 'o'}
    
    with rc_context(rc=rcparams): 
        #ax.plot(particles._r[:,0], particles._r[:,2],
        #        marker='.', color='k', alpha=0.85, ms=12)
        
        ax.plot(stripe82_xyz[:,0], stripe82_xyz[:,2], marker='.', 
                color='#111111', alpha=0.85, markersize=10, label="Stripe 82",
                markeredgewidth=0)
        
        ax.plot(catalina_xyz[:,0], catalina_xyz[:,2], marker='^', 
                color='#111111', alpha=0.85, markersize=10, label="Catalina",
                markeredgewidth=0)
        
        ax.plot(-8., 0., marker='.', color='y', alpha=1., 
                markersize=12)

        ax.set_xlabel("$X_{GC}$ [kpc]")
        ax.set_ylabel("$Z_{GC}$ [kpc]")
        ax.set_xlim(extent['x'])
        ax.set_ylim(extent['y'])
    
    ax.legend(loc='upper left')
    plt.tight_layout()
    fig.savefig(os.path.join(plot_path, "lm10.pdf"))

if __name__ == '__main__':
    #gaia_spitzer_errors()
    top_down_sgr()