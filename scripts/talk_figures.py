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
matplotlib.rc('font', family='Source Sans Pro', weight=200)
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


def bootstrapped_parameters_transpose():
    data_file = os.path.join(project_root, "plots", "hotfoot", 
                             "SMASH_new", "all_best_parameters.pickle")
    
    with open(data_file) as f:
        data = pickle.load(f)
    
    rcparams = {'axes.linewidth' : 3., 'axes.edgecolor' : '#cccccc', 
                'axes.facecolor' : '#888888', 'figure.facecolor' : '#666666',
                'xtick.major.size' : 10, 'xtick.minor.size' : 6, 'xtick.major.pad' : 8,
                'ytick.major.size' : 10, 'ytick.minor.size' : 6, 'ytick.major.pad' : 8,
                'xtick.color' : '#cccccc', 'ytick.color' : '#cccccc', 
                'text.color' : '#cccccc', 'axes.labelcolor' : '#cccccc'}
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
        
        axes[ii].axhline(true_y, linewidth=2, color='#ABDDA4', alpha=1., zorder=-1)
        axes[ii].axvline(true_x, linewidth=2, color='#ABDDA4', alpha=1., zorder=-1)
        
        points = np.vstack([xdata, ydata]).T
        plot_point_cov(points, nstd=2, ax=axes[ii], alpha=0.3, color='#ffffff')
        plot_point_cov(points, nstd=1, ax=axes[ii], alpha=0.55, color='#ffffff')
        plot_point_cov(points, nstd=2, ax=axes[ii], color='#000000', fill=False)
        plot_point_cov(points, nstd=1, ax=axes[ii], color='#000000', fill=False)
        
        axes[ii].plot(xdata, ydata, marker='.', markersize=7, alpha=0.75, 
                      color='#2B83BA', linestyle='none', markeredgewidth=1.,
                      markeredgecolor='#777777')
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
    
    fig.text(0.855, 0.07, "[deg]", fontsize=16, color=rcparams['axes.labelcolor'])
    fig.text(0.025, 0.49, "[km/s]", fontsize=16, color=rcparams['axes.labelcolor'])
    
    plt.tight_layout()
    fig.subplots_adjust(hspace=0., wspace=0.)
    fig.savefig(os.path.join(plot_path, "bootstrap.pdf"), facecolor=rcparams['figure.facecolor'])

def sgr():
    """ Top-down plot of Sgr particles, with selected stars and then 
        re-observed
    """
    
    cached_data_file = os.path.join('plots/paper1', 'sgr_kde.pickle')
    
    extent = {'x':(-90,55), 
              'y':(-58,68)}
    
    if not os.path.exists(cached_data_file):    
        # read in all particles as a table
        pdata = particle_table(N=0, expr="(Pcol<8) & (abs(Lmflag)==1)")
        Z = sgr_kde(pdata, extent=extent)
        
        with open(cached_data_file, 'w') as f:
            pickle.dump(Z, f)
    
    with open(cached_data_file, 'r') as f:
        Z = pickle.load(f)
    
    np.random.seed(44)
    pdata = particle_table(N=100, expr="(Pcol>-1) & (Pcol<8) & (abs(Lmflag)==1)")
    
    rcparams = {'axes.linewidth' : 3., 'axes.edgecolor' : '#cccccc', 
                'axes.facecolor' : '#eeeeee', 'figure.facecolor' : '#444444',
                'xtick.major.size' : 10, 'xtick.minor.size' : 6, 'xtick.major.pad' : 8,
                'ytick.major.size' : 10, 'ytick.minor.size' : 6, 'ytick.major.pad' : 8,
                'xtick.color' : '#cccccc', 'ytick.color' : '#cccccc', 
                'text.color' : '#cccccc', 'axes.labelcolor' : '#cccccc',
                'lines.linestyle' : 'none',  'lines.color' : 'k', 'lines.marker' : 'o'}
    
    with rc_context(rc=rcparams): 
        fig,ax = plt.subplots(1, 1, figsize=(8,8))
        ax.imshow(Z**0.5, interpolation="nearest", 
              extent=extent['x']+extent['y'],
              cmap=cm.bone, aspect=1, alpha=0.65)
        
        ax.plot(pdata['x'], pdata['z'], marker='.', alpha=0.85, ms=9)
        
        # add solar symbol
        ax.text(-8., 0., s=r"$\odot$", fontsize=32)
    
    ax.set_xlim(extent['x'])
    ax.set_ylim(extent['y'])
    
    # turn off right, left ticks respectively
    ax.yaxis.tick_left()
    
    # turn off top ticks
    ax.xaxis.tick_bottom()
    
    ax.set_aspect('equal')
    
    fig.savefig(os.path.join(plot_path, "lm10.pdf"), facecolor=rcparams['figure.facecolor'])

if __name__ == '__main__':
    #gaia_spitzer_errors()
    #dump_gaia_csv()
    #bootstrapped_parameters_transpose()
    sgr()