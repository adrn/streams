# coding: utf-8

""" A script for making figures for the Spitzer proposal """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys

# Third-party
import astropy.units as u
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import rc_context, rcParams
from matplotlib.patches import Rectangle

from streams.observation import apparent_magnitude
from streams.observation.gaia import parallax_error, proper_motion_error, \
                                     add_uncertainties_to_particles
from streams.observation.rrlyrae import rrl_M_V, rrl_V_minus_I

from streams.inference import relative_normalized_coordinates
from streams.inference.lm10 import timestep
from streams.potential import LawMajewski2010
from streams.potential.lm10 import true_params
from streams.integrate.satellite_particles import SatelliteParticleIntegrator
from streams.io.lm10 import particles_today, satellite_today, time

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

def lm10_particles_selection(selected_star_idx):
    """ Top-down plot of Sgr particles, with selected stars and then 
        re-observed
    """
    
    np.random.seed(42)
    particles = particles(N=100, expr="(Pcol<7) & (Pcol>0) & (abs(Lmflag)==1)")   
    all_particles = particles(N=0, expr="(Pcol<7) & (abs(Lmflag)==1)")
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
        ax.plot(particles._r[np.logical_not(selected_star_idx),0], 
                particles._r[np.logical_not(selected_star_idx),2],
                color='k', markersize=10, alpha=1., marker='.')
        ax.plot(err_particles._r[np.logical_not(selected_star_idx),0], 
                err_particles._r[np.logical_not(selected_star_idx),2],
                color='#CA0020', markersize=10, alpha=1., marker='.')
    
        ax.plot(particles._r[selected_star_idx,0], 
                particles._r[selected_star_idx,2],
                color='k', alpha=1., marker='*', markersize=15)
        ax.plot(err_particles._r[selected_star_idx,0], 
                err_particles._r[selected_star_idx,2],
                color='#CA0020', alpha=1., marker='*', markersize=15)
                

        ax.set_xlabel("$X_{GC}$ [kpc]")
        ax.set_ylabel("$Z_{GC}$ [kpc]")
    
    plt.tight_layout()
    fig.savefig(os.path.join(plot_path, "lm10.pdf"))

def phase_space_d_vs_time(N=10):
    """ Plot the PSD for 10 stars vs. back-integration time. """
    
    np.random.seed(142)
    randidx = np.random.randint(100, size=N)
    selected_star_idx = np.zeros(100).astype(bool)
    selected_star_idx[randidx] = True
    
    wrong_params = true_params.copy()
    wrong_params['qz'] = 1.2*wrong_params['qz']
    #for k,v in wrong_params.items():
    #    wrong_params[k] = 0.9*v
    
    # define correct potential, and 5% wrong potential
    true_potential = LawMajewski2010(**true_params)
    wrong_potential = LawMajewski2010(**wrong_params)
    
    particles = particles_today(N=100, expr="(Pcol<7) & (Pcol>0) & (abs(Lmflag)==1)")    
    satellite = satellite_today()
    t1,t2 = time()
    resolution = 3.
    
    sat_R = list()
    D_pses = list()
    ts = list()
    for potential in [true_potential, wrong_potential]:
        integrator = SatelliteParticleIntegrator(potential, satellite, particles)
        s_orbit,p_orbits = integrator.run(timestep_func=timestep,
                                      timestep_args=(potential, satellite.m.value),
                                      resolution=resolution,
                                      t1=t1, t2=t2)
        
        R,V = relative_normalized_coordinates(potential, p_orbits, s_orbit) 
        D_ps = np.sqrt(np.sum(R**2, axis=-1) + np.sum(V**2, axis=-1))
        D_pses.append(D_ps)
        sat_R.append(np.sqrt(np.sum(s_orbit._r**2, axis=-1)))
        ts.append(s_orbit._t)
    
    rcparams = {'lines.linestyle' : '-', 
                'lines.linewidth' : 1.,
                'lines.color' : 'k',
                'lines.marker' : None,
                'figure.facecolor' : '#ffffff',
                'text.color' : '#000000'}
    
    dark_rcparams = {'lines.linestyle' : '-', 
                'lines.linewidth' : 1.,
                'lines.color' : '#92C5DE',
                'lines.marker' : None,
                'axes.facecolor' : '#777777',
                'axes.edgecolor' : '#aaaaaa',
                'figure.facecolor' : '#555555',
                'text.color' : '#dddddd',
                'xtick.color' : '#dddddd',
                'ytick.color' : '#dddddd',
                'axes.labelcolor' : '#dddddd',
                'axes.labelweight' : 100}
    
    with rc_context(rc=rcparams):  
        fig,axes = plt.subplots(2,1, sharex=True, sharey=True, figsize=(10,10))
        axes[0].axhline(2, linestyle='--', color='#444444')
        axes[1].axhline(2, linestyle='--', color='#444444')
        
        for ii in randidx:
            for jj in range(2):
                d = D_pses[jj][:,ii]
                sR = sat_R[jj]
                axes[jj].semilogy(ts[jj]/1000, d, alpha=0.5, color=rcparams['lines.color'])
                axes[jj].semilogy(ts[jj]/1000, 1.05+0.9*(sR-sR.min())/(sR.max()-sR.min()), 
                                  alpha=0.75, color='#CA0020')
                axes[jj].semilogy(ts[jj][np.argmin(d)]/1000, np.min(d), marker='o',
                                  alpha=0.75, color=rcparams['lines.color'])
        
        axes[1].set_ylim(1,100)
        axes[1].set_xlim(-6, 0)
    
    axes[1].set_xlabel("Backwards integration time [Gyr]")
    axes[0].set_ylabel(r"$D_{ps}$", rotation='horizontal')
    axes[1].set_ylabel(r"$D_{ps}$", rotation='horizontal')
    
    fig.subplots_adjust(hspace=0.1)
    fig.savefig(os.path.join(plot_path, "ps_distance.pdf"), 
                facecolor=rcparams['figure.facecolor'])
    
    return selected_star_idx
if __name__ == '__main__':
    gaia_spitzer_errors()
    #selected_star_idx = phase_space_d_vs_time()
    #lm10_particles_selection(selected_star_idx)