# coding: utf-8

""" A script for making figures for our Streams Paper 1 """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys
import cPickle as pickle

# Third-party
import astropy.units as u
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
    
# Read in the LM10 data
np.random.seed(142)
particles = particles_today(N=100, expr="(Pcol>-1) & (abs(Lmflag) == 1) & (Pcol < 7)")
satellite = satellite_today()
t1,t2 = time()    

def normed_objective_plot():
    """ Plot our objective function in each of the 4 parameters we vary """
    
    # data file to cache data in, makes plotting faster...
    cached_data_file = os.path.join(plot_path, 'normed_objective.pickle')
    
    Nbins = 15
    p_range = (0.9, 1.1)
    
    if not os.path.exists(cached_data_file):
        variances = dict()
        for param in ['q1','qz','v_halo','phi']:
            if not variances.has_key(param):
                variances[param] = []
                
            stats = np.linspace(true_params[param]*p_range[0],
                                true_params[param]*p_range[1], 
                                Nbins)
        
            for stat in stats:
                params = true_params.copy()
                params[param] = stat
                lm10 = LawMajewski2010(**params)
                integrator = SatelliteParticleIntegrator(lm10, satellite, particles)
                s_orbit,p_orbits = integrator.run(t1=t1, t2=t2, dt=-1.)
                variances[param].append(generalized_variance(lm10, p_orbits, s_orbit))
        
        with open(cached_data_file, 'w') as f:
            pickle.dump(variances, f)
    
    with open(cached_data_file, 'r') as f:
        variances = pickle.load(f)
    
    fig,ax = plt.subplots(1,1,figsize=(8,8))
    
    nstats = np.linspace(p_range[0], p_range[1], Nbins)
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

def variance_projections():
    """ Figure showing 2D projections of the 6D variance """
    
    #particles = particles_today(N=100, expr="(Pcol>-1) & (abs(Lmflag) == 1) & (Pcol < 7)")
    
    params = true_params.copy()
    params['qz'] = true_params['qz']*1.2
    #params['v_halo'] = true_params['v_halo']*1.2
    
    # define both potentials
    correct_lm10 = LawMajewski2010(**true_params)
    wrong_lm10 = LawMajewski2010(**params)
    
    rcparams = {'lines.linestyle' : '-', 
                'lines.linewidth' : 1.,
                'lines.color' : 'k',
                'lines.marker' : None,
                'axes.facecolor' : '#ffffff'}
    
    with rc_context(rc=rcparams):
        fig,axes = plt.subplots(2, 2, figsize=(10,10))
        colors = ['#0571B0', '#D7191C']
        markers = ['o', '^']
        labels = ['true', '20% wrong $q_z$']
        for ii,potential in enumerate([correct_lm10, wrong_lm10]):
            integrator = SatelliteParticleIntegrator(potential, satellite, particles)
            s_orbit,p_orbits = integrator.run(t1=t1, t2=t2, dt=-1.)
            min_ps = minimum_distance_matrix(potential, p_orbits, s_orbit)
            
            axes[0,0].plot(min_ps[:,0], min_ps[:,3], markers[ii], alpha=0.75, 
                           linestyle='none', color=colors[ii], label=labels[ii])
            axes[1,0].plot(min_ps[:,0], min_ps[:,4], markers[ii], alpha=0.75, 
                           linestyle='none', color=colors[ii])
            axes[0,1].plot(min_ps[:,1], min_ps[:,3], markers[ii], alpha=0.75, 
                           linestyle='none', color=colors[ii])
            axes[1,1].plot(min_ps[:,1], min_ps[:,4], markers[ii], alpha=0.75, 
                           linestyle='none', color=colors[ii])
        
        # limits
        for ax in np.ravel(axes):
            ax.set_xlim(-4,4)
            ax.set_ylim(-4,4)
            ax.xaxis.tick_bottom()
            ax.yaxis.tick_left()
            ax.xaxis.set_ticks([-2,0,2])
            ax.yaxis.set_ticks([-2,0,2])
        
        # tick hacking
        axes[0,0].set_xticks([])
        axes[0,1].set_xticks([])
        axes[0,1].set_yticks([])
        axes[1,1].set_yticks([])
        
        # labels
        axes[0,0].set_ylabel(r'$p_x$')
        axes[1,0].set_xlabel(r'$q_x$')
        axes[1,0].set_ylabel(r'$p_y$')
        axes[1,1].set_xlabel(r'$q_y$')
        
        axes[0,0].legend(loc='upper left', fancybox=True)
    
    plt.tight_layout()
    fig.subplots_adjust(wspace=0., hspace=0.)
    #plt.show()
    fig.savefig(os.path.join(plot_path, "variance_projections.pdf"))

def gaia_spitzer_errors():
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

def phase_space_d_vs_time(N=10):
    """ Plot the PSD for 10 stars vs. back-integration time. """
    
    np.random.seed(112)
    randidx = np.random.randint(100, size=N)
    selected_star_idx = np.zeros(100).astype(bool)
    selected_star_idx[randidx] = True
    
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
        s_orbit,p_orbits = integrator.run(timestep_func=timestep,
                                      timestep_args=(),
                                      resolution=resolution,
                                      t1=t1, t2=t2)
        
        R,V = relative_normalized_coordinates(potential, p_orbits, s_orbit) 
        D_ps = np.sqrt(np.sum(R**2, axis=-1)/5. + np.sum(V**2, axis=-1))
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
        axes[0].axhline(np.sqrt(2), linestyle='--', color='#444444')
        axes[1].axhline(np.sqrt(2), linestyle='--', color='#444444')
        
        for ii in randidx:
            for jj in range(2):
                d = D_pses[jj][:,ii]
                sR = sat_R[jj]
                axes[jj].semilogy(ts[jj]/1000, d, alpha=0.25, color=rcparams['lines.color'])
                axes[jj].semilogy(ts[jj]/1000, 0.3*0.9*(sR-sR.min())/(sR.max()-sR.min()) + 0.45, 
                                  alpha=0.75, color='#CA0020')
                #axes[jj].semilogy(ts[jj][np.argmin(d)]/1000, np.min(d), marker='o',
                #                  alpha=0.9, color=rcparams['lines.color'], 
                #                  markersize=8)
                axes[jj].semilogy(ts[jj][np.argmin(d)]/1000, np.min(d), marker='+',
                                  markeredgewidth=2, markeredgecolor='k',
                                  alpha=0.9, color=rcparams['lines.color'], 
                                  markersize=10)
        
        axes[1].set_ylim(0.4,20)
        axes[1].set_xlim(-6, 0)
        
        axes[0].xaxis.tick_bottom()
        axes[1].xaxis.tick_bottom()
    
    axes[1].set_xlabel("Backwards integration time [Gyr]")
    axes[0].set_ylabel(r"$D_{ps}$", rotation='horizontal')
    axes[1].set_ylabel(r"$D_{ps}$", rotation='horizontal')
    
    fig.subplots_adjust(hspace=0.1)
    fig.savefig(os.path.join(plot_path, "ps_distance.pdf"), 
                facecolor=rcparams['figure.facecolor'])
    
    return 

def sgr():
    """ Top-down plot of Sgr particles, with selected stars and then 
        re-observed
    """
    
    cached_data_file = os.path.join(plot_path, 'sgr_kde.pickle')
    
    fig,ax = plt.subplots(1, 1, figsize=(8,8))
    
    # read in all particles as a table
    pdata = particle_table(N=0, expr="(Pcol<7) & (abs(Lmflag)==1)")
    extent = {'x':(-90,55), 
              'y':(-58,68)}
    
    if not os.path.exists(cached_data_file):    
        Z = sgr_kde(pdata, extent=extent)
        
        with open(cached_data_file, 'w') as f:
            pickle.dump(Z, f)
    
    with open(cached_data_file, 'r') as f:
        Z = pickle.load(f)
    
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
        ax.text(-8., 0., s=r"$\odot$", fontsize=32)

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

def plot_point_cov(points, nstd=2, ax=None, **kwargs):
    """
    Plots an `nstd` sigma ellipse based on the mean and covariance of a point
    "cloud" (points, an Nx2 array).

    Parameters
    ----------
        points : An Nx2 array of the data points.
        nstd : The radius of the ellipse in numbers of standard deviations.
            Defaults to 2 standard deviations.
        ax : The axis that the ellipse will be plotted on. Defaults to the 
            current axis.
        Additional keyword arguments are pass on to the ellipse patch.

    Returns
    -------
        A matplotlib ellipse artist
    """
    pos = points.mean(axis=0)
    cov = np.cov(points, rowvar=False)
    return plot_cov_ellipse(cov, pos, nstd, ax, **kwargs)

def plot_cov_ellipse(cov, pos, nstd=2, ax=None, **kwargs):
    """
    Plots an `nstd` sigma error ellipse based on the specified covariance
    matrix (`cov`). Additional keyword arguments are passed on to the 
    ellipse patch artist.

    Parameters
    ----------
        cov : The 2x2 covariance matrix to base the ellipse on
        pos : The location of the center of the ellipse. Expects a 2-element
            sequence of [x0, y0].
        nstd : The radius of the ellipse in numbers of standard deviations.
            Defaults to 2 standard deviations.
        ax : The axis that the ellipse will be plotted on. Defaults to the 
            current axis.
        Additional keyword arguments are pass on to the ellipse patch.

    Returns
    -------
        A matplotlib ellipse artist
    """
    def eigsorted(cov):
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        return vals[order], vecs[:,order]

    if ax is None:
        ax = plt.gca()

    vals, vecs = eigsorted(cov)
    theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))

    # Width and height are "full" widths, not radius
    width, height = 2 * nstd * np.sqrt(vals)
    ellip = Ellipse(xy=pos, width=width, height=height, angle=theta, **kwargs)

    ax.add_artist(ellip)
    return ellip

def bootstrapped_parameters():
    data_file = os.path.join(project_root, "plots", "hotfoot", 
                             "SMASH_new", "all_best_parameters.pickle")
    
    with open(data_file) as f:
        data = pickle.load(f)
    
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

def bootstrapped_parameters_transpose():
    data_file = os.path.join(project_root, "plots", "hotfoot", 
                             "SMASH_new", "all_best_parameters.pickle")
    
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

def parameter_errors():
    data_file = os.path.join(project_root, "plots", "hotfoot", 
                             "SMASH_new", "all_best_parameters.pickle")
    
    with open(data_file) as f:
        data = pickle.load(f)
    
    params = ['q1', 'qz', 'phi', 'v_halo']
    d = np.vstack(tuple([data[p] for p in params]))
    d[2] = np.degrees(d[2])
    d[3] = (d[3]*u.kpc/u.Myr).to(u.km/u.s).value
    cov = np.cov(d)

    errors = np.sqrt(np.diag(cov))
    
    for ii,p in enumerate(params):
        print("{0} = {1:.2f} + {2:.2f}".format(p, np.mean(d[ii]), errors[ii]))

if __name__ == '__main__':
    #gaia_spitzer_errors()
    #sgr()
    #phase_space_d_vs_time()
    #normed_objective_plot()
    #variance_projections()
    #bootstrapped_parameters()
    bootstrapped_parameters_transpose()
    parameter_errors()