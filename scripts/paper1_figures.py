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
from streams.data import lm10_particles, lm10_satellite, lm10_time

matplotlib.rc('xtick', labelsize=14)
matplotlib.rc('ytick', labelsize=14)
matplotlib.rc('text', usetex=True)
matplotlib.rc('axes', edgecolor='#444444', labelsize=20)
matplotlib.rc('lines', markeredgewidth=0)
matplotlib.rc('font', family='sans-serif')
rcParams['font.sans-serif'] = 'helvetica'

plot_path = "plots/paper1/"
if not os.path.exists(plot_path):
    os.mkdir(plot_path)
    
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
        fig,axes = plt.subplots(2, 1, figsize=(8, 10), sharex=True)
        
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
                      color='#67A9CF', alpha=0.75, label='Sgr thickness')
    axes[0].add_patch(sgr_d)
    
    # From fig. 3 in http://mnras.oxfordjournals.org/content/389/3/1391.full.pdf+html
    orp_d = Rectangle((10., 0.03), 35., 0.03,
                      color='#EF8A62', alpha=0.75, label='Orp thickness')
    axes[0].add_patch(orp_d)
    
    # ??
    sgr_v = Rectangle((10., 12), 60., 1., color='#67A9CF', alpha=0.75,
                      label='Sgr dispersion')
    axes[1].add_patch(sgr_v)
    
    orp_v = Rectangle((10., 10.), 35., 1., color='#EF8A62', alpha=0.75,
                      label='Orp dispersion')
    axes[1].add_patch(orp_v)
    
    axes[0].set_ylim(top=10.)
    axes[1].set_xlim(1, 100)
    
    axes[0].set_ylabel("Frac. distance error $\sigma_D/D$")
    axes[1].set_ylabel("$v_{tan}$ error [km/s]")
    axes[1].set_xlabel("Distance [kpc]")
    axes[1].set_xticklabels(["1", "10", "100"])
    
    axes[0].set_yticklabels(["{:g}".format(yt) for yt in axes[0].get_yticks()])
    axes[1].set_yticklabels(["{:g}".format(yt) for yt in axes[1].get_yticks()])
    
    # add Gaia and Spitzer to first plot
    axes[0].text(4., 0.1, 'Gaia', fontsize=16, rotation=34)
    axes[0].text(4., 0.011, 'Spitzer', fontsize=16, color="#7B3294", alpha=0.8)
    
    axes[0].legend(loc='upper left')
    axes[1].legend(loc='upper left')
    fig.subplots_adjust(hspace=0.1)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_path, "gaia.pdf"))

def lm10_particles_selection(selected_star_idx):
    """ Top-down plot of Sgr particles, with selected stars and then 
        re-observed
    """
    
    np.random.seed(42)
    particles = lm10_particles(N=100, expr="(Pcol<7) & (Pcol>0) & (abs(Lmflag)==1)")
    psd_particles = particles[selected_star_idx]
    particles = particles[np.logical_not(selected_star_idx)]
    
    all_particles = lm10_particles(N=0, expr="(Pcol<7) & (abs(Lmflag)==1)")
    err_particles = add_uncertainties_to_particles(particles, 
                                                   radial_velocity_error=15.*u.km/u.s,
                                                   distance_error_percent=2.)
    err_psd_particles = add_uncertainties_to_particles(psd_particles, 
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
                
        ax.plot(psd_particles._r[:,0], psd_particles._r[:,2],
                color='k', markersize=10, alpha=1., marker='+')
        ax.plot(err_psd_particles._r[:,0], err_psd_particles._r[:,2],
                color='#CA0020', markersize=10, alpha=1., marker='+')

        ax.set_xlabel("$X_{GC}$ [kpc]")
        ax.set_ylabel("$Z_{GC}$ [kpc]")
    
    plt.tight_layout()
    fig.savefig(os.path.join(plot_path, "lm10.pdf"))

def phase_space_d_vs_time(N=10):
    """ Plot the PSD for 10 stars vs. back-integration time. """
    
    np.random.seed(142)
    selected_star_idx = np.random.randint(100, size=N)
    
    wrong_params = true_params.copy()
    wrong_params['qz'] = 1.2*wrong_params['qz']
    #for k,v in wrong_params.items():
    #    wrong_params[k] = 0.9*v
    
    # define correct potential, and 5% wrong potential
    true_potential = LawMajewski2010(**true_params)
    wrong_potential = LawMajewski2010(**wrong_params)
    
    particles = lm10_particles(N=100, expr="(Pcol<7) & (Pcol>0) & (abs(Lmflag)==1)")    
    satellite = lm10_satellite()
    t1,t2 = lm10_time()
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
                'lines.marker' : None}
    
    with rc_context(rc=rcparams):  
        fig,axes = plt.subplots(2,1, sharex=True, sharey=True, figsize=(10,10))
        axes[0].axhline(2, linestyle='--')
        axes[1].axhline(2, linestyle='--')
        
        for ii in selected_star_idx:
            for jj in range(2):
                d = D_pses[jj][:,ii]
                sR = sat_R[jj]
                axes[jj].semilogy(ts[jj], d, alpha=0.5, color='k')
                axes[jj].semilogy(ts[jj], 1.+(sR-sR.min())/(sR.max()-sR.min()), alpha=0.5, color='r')
                axes[jj].semilogy(ts[jj][np.argmin(d)], np.min(d), marker='o',
                                  alpha=0.75, color='k')
        
        axes[1].set_ylim(1,100)
        axes[1].set_xlim(-4000, 0)
    
    axes[1].set_xlabel("Backwards integration time [Myr]")
    axes[0].set_ylabel(r"$D_{ps}$")
    axes[1].set_ylabel(r"$D_{ps}$")
    
    fig.subplots_adjust(hspace=0.1)
    fig.savefig(os.path.join(plot_path, "ps_distance.pdf"))
    
    return selected_star_idx
if __name__ == '__main__':
    gaia_spitzer_errors()
    selected_star_idx = phase_space_d_vs_time()
    lm10_particles_selection(selected_star_idx)