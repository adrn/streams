# coding: utf-8

""" A script for making figures for our Streams Paper 1 """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys
import cPickle as pickle

# Third-party
import astropy.units as u
from astropy.io.misc import fnpickle, fnunpickle
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import rc_context, rcParams, cm
from matplotlib.patches import Rectangle, Ellipse
import scipy.optimize as so

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
from streams.io import read_table
from streams.plot import *

matplotlib.rc('xtick', labelsize=18)
matplotlib.rc('ytick', labelsize=18)
#matplotlib.rc('text', usetex=True)
matplotlib.rc('axes', edgecolor='#444444', labelsize=24, labelweight=400, linewidth=2.0)
matplotlib.rc('lines', markeredgewidth=0)
matplotlib.rc('font', family='Source Sans Pro')
#matplotlib.rc('savefig', bbox='standard')

plot_path = "plots/paper1/"
if not os.path.exists(plot_path):
    os.mkdir(plot_path)

def normed_objective_plot(**kwargs):
    """ Plot our objective function in each of the 4 parameters we vary """
    
    # Read in the LM10 data
    np.random.seed(142)
    particles = particles_today(N=100, expr="(Pcol>-1) & (abs(Lmflag) == 1) & (Pcol < 7)")
    satellite = satellite_today()
    t1,t2 = time()

    # optional parameters:
    Nbins = int(kwargs.get("Nbins", 15))
    percent_bounds = float(kwargs.get("percent_bounds", 10))

    frac_p_range = (1-percent_bounds/100, 1+percent_bounds/100)

    # data file to cache data in, makes plotting faster...
    data_file = os.path.join(plot_path, 'normed_objective.pickle')

    if os.path.exists(data_file) and kwargs.get("overwrite", False):
        os.remove(data_file)
    
    if not os.path.exists(data_file):
        variances = dict()
        for param in ['q1','qz','v_halo','phi']:
            if not variances.has_key(param):
                variances[param] = []
                
            stats = np.linspace(true_params[param]*frac_p_range[0],
                                true_params[param]*frac_p_range[1], 
                                Nbins)
        
            for stat in stats:
                params = true_params.copy()
                params[param] = stat
                lm10 = LawMajewski2010(**params)
                integrator = SatelliteParticleIntegrator(lm10, satellite, particles)
                s_orbit,p_orbits = integrator.run(t1=t1, t2=t2, dt=-1.)
                variances[param].append(generalized_variance(lm10, s_orbit, p_orbits))
        
        # pickle the data to a file
        fnpickle(variances, data_file)

    # unpickle data
    variances = fnunpickle(data_file)
    
    fig,ax = plt.subplots(1,1,figsize=(8,8))
    
    nstats = np.linspace(frac_p_range[0], frac_p_range[1], Nbins)
    linestyles = [(2,'-'), (3,'-.'), (3,'--'), (3,':')]
    for ii, name in enumerate(['q1','qz','phi','v_halo']):
        vals = variances[name]
        ls = linestyles[ii] 
        ax.plot(nstats, vals, label=param_to_latex[name], linewidth=ls[0], linestyle=ls[1])
    
    ax.xaxis.tick_bottom()
    ax.yaxis.tick_left()
    ax.legend(loc='upper right', fancybox=True)
    
    ax.set_yticks([])
    ax.set_xlabel('Normalized parameter value', labelpad=15)
    ax.set_ylabel('Generalized variance', labelpad=15)
    
    fig.savefig(os.path.join(plot_path, "objective_function.pdf"))

def gaia_spitzer_errors(**kwargs):
    """ Visualize the observational errors from Gaia and Spitzer, along with
        dispersion and distance scale of Sgr and Orphan. 
    """
    
    rcparams = {'lines.linestyle' : '-', 
                'lines.linewidth' : 1.,
                'lines.color' : 'k',
                'lines.marker' : None,
                'axes.facecolor' : '#ffffff'}
    
    sgr_color = '#2B83BA'
    orp_color = '#ABDDA4'
    
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
            axes[0].loglog(D.kiloparsec, (dD/D).decompose(), color='#666666', 
                           alpha=0.1)
                
            # Plot tangential velocity errors
            axes[1].loglog(D.kiloparsec, dVtan, color='#666666', alpha=0.1)
    
        # Add spitzer 2% line to distance plot
        axes[0].axhline(0.02, linestyle='--', linewidth=4, color='k', alpha=0.75)
        
        # Add photometric 20% line to distance plot
        axes[0].axhline(0.1, linestyle=':', linewidth=4, color='k', alpha=0.75)
    
    # Now add rectangles for Sgr, Orphan
    sgr_d = Rectangle((10., 0.15), 60., 0.15, 
                      color=sgr_color, alpha=1., label='Sgr stream')
    axes[0].add_patch(sgr_d)
    
    # From fig. 3 in http://mnras.oxfordjournals.org/content/389/3/1391.full.pdf+html
    orp_d = Rectangle((10., 0.03), 35., 0.03,
                      color=orp_color, alpha=1., label='Orp stream')
    axes[0].add_patch(orp_d)
    
    # Dispersion from Majewski 2004: 10 km/s
    sgr_v = Rectangle((10., 10), 60., 1., color=sgr_color)
    axes[1].add_patch(sgr_v)
    
    orp_v = Rectangle((10., 8.), 35., 1., color=orp_color)
    axes[1].add_patch(orp_v)
    
    axes[0].set_ylim(top=10.)
    axes[0].set_xlim(1, 100)
    axes[1].set_ylim(0.01, 100)
    
    axes[0].set_ylabel(r"$\sigma_D/D$ (frac. dist. error)")
    axes[1].set_ylabel(r"$\sigma_\mu \times$ $D$ ($v_{tan}$ error) [km/s]")
    axes[1].set_xlabel("Distance [kpc]")
    
    axes[0].set_xticklabels(["1", "10", "100"])
    
    axes[0].set_yticklabels(["{:g}".format(yt) for yt in axes[0].get_yticks()])
    axes[1].set_yticklabels(["{:g}".format(yt) for yt in axes[1].get_yticks()])
    
    # add Gaia and Spitzer text to first plot
    axes[0].text(15., 1.5, 'Gaia', fontsize=16, rotation=32, fontweight=500)
    axes[0].text(4., 0.011, 'Spitzer', fontsize=16, alpha=0.75, fontweight=500)
    
    # add legends
    axes[0].legend(loc='upper left', fancybox=True)
    #axes[1].legend(loc='upper left', fancybox=True)
    
    fig.subplots_adjust(hspace=0.0)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_path, "gaia.pdf"))

def phase_space_d_vs_time(**kwargs):
    """ Plot the PSD for 10 stars vs. back-integration time. """
    
    N = int(kwargs.get("N", 10))
    seed = int(kwargs.get("seed", 112358))

    # Read in the LM10 data
    np.random.seed(seed)
    particles = particles_today(N=N, expr="(Pcol>-1) & (abs(Lmflag) == 1) & (Pcol < 7)")
    satellite = satellite_today()
    t1,t2 = time()
    t2 = -6000.

    wrong_params = true_params.copy()
    wrong_params['qz'] = 1.25*wrong_params['qz']
    #wrong_params['v_halo'] = wrong_params['v_halo']
    
    # define correct potential, and wrong potential
    true_potential = LawMajewski2010(**true_params)
    wrong_potential = LawMajewski2010(**wrong_params)
    
    resolution = 3.

    sat_R = list()
    D_pses = list()
    ts = list()
    for potential in [true_potential, wrong_potential]:
        integrator = SatelliteParticleIntegrator(potential, satellite, particles)
        s_orbit,p_orbits = integrator.run(t1=t1, t2=t2, dt=-1.)
        
        R,V = relative_normalized_coordinates(potential, s_orbit, p_orbits) 
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
        fig,axes = plt.subplots(2,1, sharex=True, sharey=True, figsize=(8,12))
        axes[0].axhline(2, linestyle='--', color='#444444')
        axes[1].axhline(2, linestyle='--', color='#444444')
        
        for ii in range(N):
            for jj in range(2):
                d = D_pses[jj][:,ii]
                sR = sat_R[jj]
                axes[jj].semilogy(ts[jj]/1000, d, alpha=0.25, color=rcparams['lines.color'])
                #axes[jj].semilogy(ts[jj]/1000, 0.3*0.9*(sR-sR.min())/(sR.max()-sR.min()) + 0.45, 
                #                  alpha=0.75, color='#CA0020')
                #axes[jj].semilogy(ts[jj][np.argmin(d)]/1000, np.min(d), marker='o',
                #                  alpha=0.9, color=rcparams['lines.color'], 
                #                  markersize=8)
                axes[jj].semilogy(ts[jj][np.argmin(d)]/1000, np.min(d), marker='+',
                                  markeredgewidth=2, markeredgecolor='k',
                                  alpha=0.9, color=rcparams['lines.color'], 
                                  markersize=10)
        
        axes[1].set_ylim(0.6,20)
        axes[1].set_xlim(-6.1, 0.1)
        
        axes[0].xaxis.tick_bottom()
        axes[1].xaxis.tick_bottom()
    
    axes[1].set_xlabel("Backwards integration time [Gyr]")
    axes[0].set_ylabel(r"$D_{ps}$", rotation='horizontal')
    axes[1].set_ylabel(r"$D_{ps}$", rotation='horizontal')
    
    fig.subplots_adjust(hspace=0.1)
    fig.savefig(os.path.join(plot_path, "ps_distance.pdf"), 
                facecolor=rcparams['figure.facecolor'])
    
    return 

def sgr(**kwargs):
    """ Top-down plot of Sgr particles, with selected stars and then 
        re-observed
    """
    
    data_file = os.path.join(plot_path, 'sgr_kde.pickle')
    
    fig,ax = plt.subplots(1, 1, figsize=(8,8))
    
    # read in all particles as a table
    pdata = particle_table(N=0, expr="(Pcol<7) & (abs(Lmflag)==1)")
    extent = {'x':(-90,55), 
              'y':(-58,68)}
    
    if not os.path.exists(data_file):    
        Z = sgr_kde(pdata, extent=extent)
        fnpickle(Z, data_file)
    
    Z = fnunpickle(data_file)
    
    ax.imshow(Z**0.5, interpolation="nearest", 
              extent=extent['x']+extent['y'],
              cmap=cm.Blues, aspect=1)
    
    np.random.seed(42)
    pdata = particle_table(N=100, expr="(Pcol>-1) & (Pcol<7) & (abs(Lmflag)==1)")
    
    rcparams = {'lines.linestyle' : 'none', 
                'lines.color' : 'k',
                'lines.marker' : 'o'}
    
    with rc_context(rc=rcparams): 
        ax.plot(pdata['x'], pdata['z'], marker='.', alpha=0.85, ms=9)
        
        # add solar symbol
        ax.text(-8.-5.75, -3., s=r"$\odot$", fontsize=32)

        ax.set_xlabel("$X_{GC}$ [kpc]")
        ax.set_xlabel("$X_{GC}$ [kpc]")
    
    ax.set_xlim(extent['x'])
    ax.set_ylim(extent['y'])
    
    # turn off right, left ticks respectively
    ax.yaxis.tick_left()
    
    # turn off top ticks
    ax.xaxis.tick_bottom()
    
    ax.set_aspect('equal')
    
    fig.savefig(os.path.join(plot_path, "lm10.pdf"))

def bootstrapped_parameters(**kwargs):
    data_file = os.path.join(project_root, "plots", "hotfoot", 
                             "SMASH_aspen", "all_best_parameters.pickle")
    data = fnunpickle(data_file)
    
    rcparams = {'axes.linewidth' : 3.,
                'xtick.major.size' : 8.}
    with rc_context(rc=rcparams): 
        fig,axes = plt.subplots(3,1,figsize=(5,12))

    y_param = 'v_halo'
    x_params = ['q1', 'qz', 'phi']

    for ii,x_param in enumerate(x_params):
        ydata = (np.array(data[y_param])-_true_params[y_param]) / _true_params[y_param]
        xdata = (np.array(data[x_param])-_true_params[x_param]) / _true_params[x_param]
        
        axes[ii].axhline(0., linewidth=2, color='#ABDDA4', alpha=0.75, zorder=-1)
        axes[ii].axvline(0., linewidth=2, color='#ABDDA4', alpha=0.75, zorder=-1)
        
        points = np.vstack([ydata, xdata]).T
        plot_point_cov(points, nstd=2, ax=axes[ii], alpha=0.25, color='#777777')
        plot_point_cov(points, nstd=1, ax=axes[ii], alpha=0.5, color='#777777')
        plot_point_cov(points, nstd=2, ax=axes[ii], color='#000000', fill=False)
        plot_point_cov(points, nstd=1, ax=axes[ii], color='#000000', fill=False)
        
        axes[ii].plot(ydata, xdata, marker='.', markersize=7, alpha=0.75, 
                      color='#2B83BA', linestyle='none')
        axes[ii].set_xlim((-0.12, 0.12))
        axes[ii].set_ylim((-0.12, 0.12))
        
        axes[ii].yaxis.tick_left()        
        axes[ii].set_yticks([-0.1, -0.05, 0., 0.05, 0.1])
    
    axes[2].set_xlabel(r"$\delta v_{\rm halo}$", 
                       fontsize=26, rotation='horizontal')
    
    axes[0].xaxis.tick_bottom()
    axes[0].set_xticklabels([])
    axes[1].set_xticklabels([])
    axes[2].set_xticks([-0.1, -0.05, 0., 0.05, 0.1])
    
    axes[0].set_ylabel(r"$\delta q_1$", fontsize=26, rotation='horizontal')
    axes[1].set_ylabel(r"$\delta q_z$", fontsize=26, rotation='horizontal')
    axes[2].set_ylabel(r"$\delta \phi$", fontsize=26, rotation='horizontal')
    
    plt.tight_layout()
    fig.subplots_adjust(hspace=0., wspace=0.)
    fig.savefig(os.path.join(plot_path, "bootstrap.pdf"))

def bootstrapped_parameters_transpose(**kwargs):
    data_file = os.path.join(project_root, "plots", "hotfoot", 
                             "SMASH_aspen", "all_best_parameters.pickle")
    
    with open(data_file) as f:
        data = pickle.load(f)
    
    rcparams = {'axes.linewidth' : 3.,
                'xtick.major.size' : 8.}
    with rc_context(rc=rcparams): 
        fig,axes = plt.subplots(1,3,figsize=(12,5))

    y_param = 'v_halo'
    x_params = ['q1', 'qz', 'phi']
    
    data['phi'] = (data['phi']*u.radian).to(u.degree).value
    _true_params['phi'] = (_true_params['phi']*u.radian).to(u.degree).value
    data['v_halo'] = (data['v_halo']*u.kpc/u.Myr).to(u.km/u.s).value
    _true_params['v_halo'] = (_true_params['v_halo']*u.kpc/u.Myr).to(u.km/u.s).value
    
    style = dict()
    style['q1'] = dict(ticks=[1.3, 1.38, 1.46],
                       lims=(1.27, 1.49))
    style['qz'] = dict(ticks=[1.28, 1.36, 1.44],
                       lims=(1.25, 1.47))
    style['phi'] = dict(ticks=[92, 97, 102],
                        lims=(90, 104))
    style['v_halo'] = dict(ticks=[115, 122, 129],
                           lims=(110, 134))
    
    for ii,x_param in enumerate(x_params):
        
        true_x = _true_params[x_param]
        true_y = _true_params[y_param]
        xdata = np.array(data[x_param])
        ydata = np.array(data[y_param])
        
        axes[ii].axhline(true_y, linewidth=2, color='#ABDDA4', alpha=0.75, zorder=-1)
        axes[ii].axvline(true_x, linewidth=2, color='#ABDDA4', alpha=0.75, zorder=-1)
        
        points = np.vstack([xdata, ydata]).T
        plot_point_cov(points, nstd=2, ax=axes[ii], alpha=0.25, color='#777777')
        plot_point_cov(points, nstd=1, ax=axes[ii], alpha=0.5, color='#777777')
        plot_point_cov(points, nstd=2, ax=axes[ii], color='#000000', fill=False)
        plot_point_cov(points, nstd=1, ax=axes[ii], color='#000000', fill=False)
        
        axes[ii].plot(xdata, ydata, marker='.', markersize=7, alpha=0.75, 
                      color='#2B83BA', linestyle='none')
        axes[ii].set_xlim(style[x_param]['lims'])
        axes[ii].set_ylim(style[y_param]['lims'])
        
        axes[ii].yaxis.tick_left()        
        axes[ii].set_xticks(style[x_param]['ticks'])
        axes[ii].xaxis.tick_bottom()
    
    axes[0].set_ylabel(r"$v_{\rm halo}$", 
                       fontsize=26, rotation='horizontal')
    
    axes[0].set_yticks(style[y_param]['ticks'])
    axes[1].set_yticklabels([])
    axes[2].set_yticklabels([])
    
    axes[0].set_xlabel(r"$q_1$", fontsize=26, rotation='horizontal')
    axes[1].set_xlabel(r"$q_z$", fontsize=26, rotation='horizontal')
    axes[2].set_xlabel(r"$\phi$", fontsize=26, rotation='horizontal')
    
    fig.text(0.855, 0.07, "[deg]", fontsize=16)
    fig.text(0.025, 0.49, "[km/s]", fontsize=16)
    
    plt.tight_layout()
    fig.subplots_adjust(hspace=0., wspace=0.)
    fig.savefig(os.path.join(plot_path, "bootstrap.pdf"))

def parameter_errors(**kwargs):
    data_file = os.path.join(project_root, "plots", "hotfoot", 
                             "SMASH_aspen", "all_best_parameters.pickle")
    data = fnunpickle(data_file)
    
    params = ['q1', 'qz', 'phi', 'v_halo']
    d = np.vstack(tuple([data[p] for p in params]))
    d[2] = np.degrees(d[2])
    d[3] = (d[3]*u.kpc/u.Myr).to(u.km/u.s).value
    cov = np.cov(d)

    errors = np.sqrt(np.diag(cov))
    
    for ii,p in enumerate(params):
        print("{0} = {1:.2f} + {2:.2f}".format(p, np.mean(d[ii]), errors[ii]))

# ============================================================================
# Below here, responding to referee
#

def num_particles_recombine():
    pass
    
def when_particles_recombine(**kwargs):
    """ I should plot number of bound particles vs. time over the orbit of 
        the progenitor vs. time. 
    """

    N = int(kwargs.get("N",10000))
    D_ps_limit = float(kwargs.get("D_ps_limit", 2.1))

    from streams.io.sgr import mass_selector, usys_from_file
    from streams.integrate import LeapfrogIntegrator
    particles_today, satellite_today, time = mass_selector("2.5e8")
    
    potential = LawMajewski2010()
    satellite = satellite_today()
    t1,t2 = time()
    plchldr = np.zeros((len(satellite._r), 3))
    integrator = LeapfrogIntegrator(potential._acceleration_at, 
                                    np.array(satellite._r), np.array(satellite._v),
                                    args=(len(satellite._r), plchldr))
    ts, xs, vs = integrator.run(t1=t1, t2=t2, dt=-1.)
    
    _full_path = os.path.join(project_root, "data", "simulation", "Sgr", "2.5e8")
    usys = usys_from_file(os.path.join(_full_path, "SCFPAR"))
    data = read_table("SNAP", path=_full_path, N=0)

    tub = (data['tub']*usys['time']).to(u.Myr).value
    R_orbit = np.sqrt(np.sum(xs**2, axis=-1))
    
    tub_file = os.path.join(_full_path, "tub_back{0}_{1}.pickle".format(N,D_ps_limit))
    if overwrite and os.path.exists(tub_file):
        os.remove(tub_file)

    if not os.path.exists(tub_file):
        # read full orbits of particles from running on hotfoot
        p_x = np.load(os.path.join(project_root, "data", "{0}particles_p.npy".format(N)))
        s_x = np.load(os.path.join(project_root, "data", "{0}particles_s.npy".format(N)))

        r_tide = potential._tidal_radius(m=satellite._m,
                                         r=s_x[...,:3])[:,:,np.newaxis]

        v_esc = potential._escape_velocity(m=satellite._m,
                                           r_tide=r_tide)
    
        R,V = (p_x[...,:3] - s_x[...,:3]) / r_tide, (p_x[...,3:] - s_x[...,3:]) / v_esc
        D_ps = np.sqrt(np.sum(R**2, axis=-1) + np.sum(V**2, axis=-1))
        tub_back = []
        for ii in range(D_ps.shape[1]):
            idx = D_ps[:,ii] < D_ps_limit
            
            if np.any(idx):
                tub_back.append(np.median(ts[idx]))
                #tub_back.append(ts[idx][-1])
            else:
                #print("min Dps:",np.min(D_ps[:,ii]))
                #print(ts[D_ps[:,ii].argmin()])
                tub_back.append(np.nan)
        tub_back = np.array(tub_back)
        fnpickle(tub_back, tub_file)

    tub_back = fnunpickle(tub_file)
    bound_stars_back = [np.sum(tub_back > t) for t in ts]

    from scipy.signal import argrelextrema
    apos,xxx = argrelextrema(R_orbit, np.greater)
    peris,xxx = argrelextrema(R_orbit, np.less)    

    fig,axes = plt.subplots(3,1,sharex=True,figsize=(18,12))
    fig.suptitle("Dps boundary: {0}".format(D_ps_limit))
    
    bins = np.linspace(0., max(tub), 100)
    N_bootstrap = 10
    for ii in range(N_bootstrap):
        tub_b = tub[np.random.randint(len(tub), size=N)]
        bound_stars = [np.sum(tub_b > t) for t in ts]
        axes[0].semilogy(ts, bound_stars/max(bound_stars), color='k', lw=1., alpha=0.25)
        n,bins,patches = axes[1].hist(tub_b[tub_b!=0], normed=True, bins=bins, color='k', histtype="step", alpha=0.25)

    axes[0].set_ylabel("Frac. of bound particles")
    axes[0].semilogy(ts, bound_stars_back/max(bound_stars_back), color='#1A9641', lw=3.)
    axes[0].set_ylim(1E-2, 1.1)

    axes[1].hist(tub_back, normed=True, bins=bins, color='#1A9641', histtype="step", lw=3.)
    axes[1].yaxis.set_ticks([])
    
    axes[-1].plot(ts, R_orbit)
    axes[-1].plot(t1, np.sqrt(np.sum(satellite._r**2, axis=-1)), marker='o', color='r')
    axes[-1].set_xlabel("Time [Myr]")
    axes[-1].set_ylabel("$R_{GC}$ of sat.")
    axes[-1].set_xlim(min(ts), max(ts))
    
    for ii,peri in enumerate(peris):
        if ii == 0:
            axes[0].axvline(ts[peri], color='#F4A582', label='peri')
        else:
            axes[0].axvline(ts[peri], color='#F4A582')
        for ax in axes[1:]:
            ax.axvline(ts[peri], color='#F4A582')
    
    for ii,apo in enumerate(apos):
        if ii == 0:
            axes[0].axvline(ts[apo], color='#92C5DE', label='apo')
        else:
            axes[0].axvline(ts[apo], color='#92C5DE')
        for ax in axes[1:]:
            ax.axvline(ts[apo], color='#92C5DE')
    
    
    axes[0].legend(loc='lower left')
    fig.subplots_adjust(hspace=0.05)
    fig.savefig(os.path.join(plot_path, "when_recombine_N{0}_Dps{1}.png".format(N,D_ps_limit)))

if __name__ == '__main__':
    from argparse import ArgumentParser
    import logging
    
    # Create logger
    logger = logging.getLogger(__name__)
    ch = logging.StreamHandler()
    formatter = logging.Formatter("%(name)s / %(levelname)s / %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    # Define parser object
    parser = ArgumentParser(description="")
    parser.add_argument("-v", "--verbose", action="store_true", dest="verbose",
                        default=False, help="Be chatty! (default = False)")
    parser.add_argument("-q", "--quiet", action="store_true", dest="quiet", 
                        default=False, help="Be quiet! (default = False)")
    parser.add_argument("-o", "--overwrite", action="store_true", dest="overwrite", 
                        default=False, help="Overwrite existing files.")
    parser.add_argument("-f", "--function", dest="function", type=str,
                        required=True, help="The name of the function to execute.")
    parser.add_argument("--kwargs", dest="kwargs", nargs="+", type=str,
                        help="kwargs passed in to whatever function you call.")

    args = parser.parse_args()
    try:
        kwargs = dict([tuple(k.split("=")) for k in args.kwargs])
    except TypeError:
        kwargs = dict()

    # Set logger level based on verbose flags
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    elif args.quiet:
        logger.setLevel(logging.ERROR)
    else:
        logger.setLevel(logging.INFO)
    
    func = getattr(sys.modules[__name__], args.__dict__.get("function"))
    func(**kwargs)

