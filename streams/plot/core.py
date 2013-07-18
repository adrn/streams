# coding: utf-8

""" General plotting utilities. """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys
import cPickle as pickle

# Third-party
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as col
import matplotlib.cm as cm
import matplotlib.gridspec as gridspec
import astropy.units as u
from scipy.stats import gaussian_kde

from ..potential.lm10 import param_to_latex
from ..potential.lm10 import true_params
from ..inference.lm10 import param_ranges

__all__ = ["discrete_cmap", "emcee_plot", "plot_sampler_pickle", \
           "bootstrap_scatter_plot", "sgr_kde"]

def discrete_cmap(N=8):
    """create a colormap with N (N<15) discrete colors and register it"""
    # define individual colors as hex values
    # #9E0142
    cpool = ['#D085A4', '#D53E4F', '#F46D43', '#FDAE61', '#FEE08B',
             '#E6F598', '#ABDDA4', '#66C2A5', '#3288BD', '#5E4FA2']
    if N == 5:
        cmap3 = col.ListedColormap(cpool[::2], 'nice_spectral')
    else:
        cmap3 = col.ListedColormap(cpool[0:N], 'nice_spectral')
    cm.register_cmap(cmap=cmap3)

# TODO: make this more general...
def emcee_plot(sampler, params, converged_idx, 
                acceptance_fraction_bounds=(None,None), show_true=False):
    """ Plot posterior probability distributions and chain traces from the 
        chains associated with the given sampler. 
        
        Parameters
        ----------
        sampler : emcee.EnsembleSampler
            A sampler object that has already been run -- e.g. run_mcmc() has
            already been called.
        params : list
            A list of halo parameters.
        acceptance_fraction_bounds : tuple (optional)
            Only plot samples from chains that have an acceptance fraction
            within the provided range.
        show_true : bool (optional)
            Optionally show the true halo parameter value as a line.
    """
    
    if len(params) != sampler.chain.shape[2]:
        raise ValueError("Number of params doesn't match sampler shape[2]!")
    
    fig = plt.figure(figsize=(16,20.6))
    
    # I want the plot of individual walkers to span 2 columns
    gs = gridspec.GridSpec(len(params), 3)
    
    idx = np.ones_like(sampler.acceptance_fraction).astype(bool)
    
    if acceptance_fraction_bounds[0] != None:
        idx &= sampler.acceptance_fraction > acceptance_fraction_bounds[0]
    
    if acceptance_fraction_bounds[1] != None:
        idx &= sampler.acceptance_fraction < acceptance_fraction_bounds[1]
    
    chain = sampler.chain[idx]
        
    # For each parameter, I want to plot each walker on one panel, and a histogram
    #   of all links from all walkers past 150 steps (approximately when the chains
    #   converged)
    for ii,param in enumerate(params):
        these_chains = chain[:,:,ii]
        
        #if param == "v_halo":
        #    these_chains = (these_chains*u.kpc/u.Myr).to(u.km/u.s).value
        
        ax1 = plt.subplot(gs[ii, :2])     
        ax1.axvline(converged_idx, 
                    color="#67A9CF", 
                    alpha=0.7,
                    linewidth=2)
        
        for walker in these_chains:
            ax1.plot(np.arange(len(walker)), walker,
                    drawstyle="steps", 
                    color="#555555",
                    alpha=0.5)
                    
        ax1.set_ylabel(param_to_latex[param], 
                       fontsize=36,
                       labelpad=18,
                       rotation="horizontal",
                       color="k")
        
        # Don't show ticks on the y-axis
        ax1.yaxis.set_ticks([])
        
        # For the plot on the bottom, add an x-axis label. Hide all others
        if ii == len(params)-1:
            ax1.set_xlabel("step number", fontsize=24, labelpad=18, color="k")
        else:
            ax1.xaxis.set_visible(False)
        
        ax2 = plt.subplot(gs[ii, 2])
        
        # Same y-bounds as the walkers plot, so they line up
        #ax1.set_ylim(np.min(these_chains[:,ii]), np.max(these_chains[:,ii]))
        ax1.set_ylim(param_ranges[param])
        ax2.set_ylim(ax1.get_ylim())
        ax2.xaxis.set_visible(False)
        ax2.yaxis.tick_right()
        
        # Create a histogram of all values past the converged point. Make 100 bins
        #   between the y-axis bounds defined by the 'walkers' plot.
        ax2.hist(np.ravel(these_chains[:,converged_idx:]), 
                 bins=np.linspace(ax1.get_ylim()[0],ax1.get_ylim()[1],100),
                 orientation='horizontal',
                 facecolor="#67A9CF",
                 edgecolor="none")
        
        if show_true:
            val = true_params[param]
            if hasattr(val, "value"):
                val = val.value
                
            ax2.axhline(val, 
                        color="#555555",
                        linestyle="--")
        
        # For the first plot, add titles and shift them up a bit
        if ii == 0:
            t = ax1.set_title("Walkers", fontsize=30, color="k")
            t.set_y(1.01) 
            t = ax2.set_title("Posterior", fontsize=30, color="k")
            t.set_y(1.01) 
        
        if param == "v_halo":
            ax2.set_ylabel("km/s", 
                           fontsize=20,
                           rotation="horizontal",
                           color="k",
                           labelpad=16)
        elif param == "phi":
            ax2.set_ylabel("rad",
                           fontsize=20,
                           rotation="horizontal",
                           color="k",
                           labelpad=16)
        elif param == "r_halo":
            ax2.set_ylabel("kpc",
                           fontsize=20,
                           rotation="horizontal",
                           color="k",
                           labelpad=16)
        ax2.yaxis.set_label_position("right")
        
        # Adjust axis ticks, e.g. make them appear on the outside of the plots and
        #   change the padding / color.
        ax1.tick_params(axis='x', pad=2, direction='out', colors="#444444", labelsize=14)
        ax2.tick_params(axis='y', pad=2, direction='out', colors="#444444", labelsize=14)
        
        # Removes the top tick marks
        ax1.get_xaxis().tick_bottom()
        
        # Hack because the tick labels for phi are wonky... but this removed the 
        #   first and last tick labels so I can squash the plots right up against
        #   each other
        if param == "phi":
            #ax2.set_yticks(ax2.get_yticks()[1:-2])
            ax2.set_yticks(ax2.get_yticks()[1:-1])
        else:
            ax2.set_yticks(ax2.get_yticks()[1:-1])
    
    fig.subplots_adjust(hspace=0.02, wspace=0.0, bottom=0.075, top=0.9, left=0.12, right=0.88)
    return fig

def plot_sampler_pickle(filename, params, **kwargs):
    """ Given a pickled emcee Sampler object, generate an emcee_plot """
    
    if not hasattr(filename, "read"):
        if not os.path.exists(filename):
            raise IOError("File {0} doesn't exist!".format(filename))
        
        with open(filename) as f:
            sampler = pickle.load(f)
        
    else:
        sampler = pickle.load(f)
    
    # If chains converged, make mcmc plots
    fig = emcee_plot(sampler, params=params, 
                     converged_idx=0,
                     **kwargs)
    return fig

def bootstrap_scatter_plot(d, subtitle="", axis_lims=None):
    """ """
    matplotlib.rc('xtick', labelsize=16)
    matplotlib.rc('ytick', labelsize=16)
    fig,axes = plt.subplots(3, 3, figsize=(12,12))
    
    # Create dictionary of true parameter values, not quantities
    params = dict()
    for p in d.keys():
        params[p] = true_params[p]
        
        if p == 'phi':
            params[p] = params[p].value
        elif p == 'v_halo':
            params[p] = params[p].to(u.km/u.s).value
    
    # if no limits are provided, defined default axis limits
    axis_lims = dict(q1=(1.,2), 
                     qz=(1.,2.), 
                     phi=(np.pi/4, 3*np.pi/4), 
                     v_halo=(100, 150))
    
    # Create latex labels for the parameters
    labels = dict(q1="$q_1$", qz="$q_z$", 
                  phi="$\phi$ [rad]", v_halo="$v_{halo}$ [km/s]")
    
    # helper function for plotting single axes
    def plot_2params(ax, p1, p2):
        x = d[p1]
        y = d[p2]
        if p1 == 'v_halo':
            x = (d[p1]*u.kpc/u.Myr).to(u.km/u.s).value
        elif p2 == 'v_halo':
            y = (d[p2]*u.kpc/u.Myr).to(u.km/u.s).value
        
        ax.scatter(x, y, color='k', marker='.', alpha=0.5)
        ax.set_xlim(axis_lims[p1])
        ax.set_ylim(axis_lims[p2])
        ax.axvline(params[p1], linestyle="--", color="#555555")
        ax.axhline(params[p2], linestyle="--", color="#555555")
    
    q_ticks = np.arange(1.2,1.8,0.2)
    v_ticks = np.arange(105,155,10)
    phi_ticks = np.arange(1.0,2.2,0.2)
    plot_2params(axes[0,0], "q1", "qz")
    axes[0,0].set_ylabel(labels["qz"])
    axes[0,0].set_yticks(q_ticks)
    
    plot_2params(axes[1,0], "q1", "phi")
    axes[1,0].set_ylabel(labels["phi"])
    axes[1,0].set_yticks(phi_ticks)
    
    plot_2params(axes[2,0], "q1", "v_halo")
    axes[2,0].set_xlabel(labels["q1"])
    axes[2,0].set_ylabel(labels["v_halo"])
    axes[2,0].set_xticks(q_ticks)
    axes[2,0].set_yticks(v_ticks)
    
    plot_2params(axes[1,1], "qz", "phi")
    plot_2params(axes[2,1], "qz", "v_halo")
    axes[2,1].set_xlabel(labels["qz"])
    axes[2,1].set_xticks(q_ticks)
    
    plot_2params(axes[2,2], "phi", "v_halo")
    axes[2,2].set_xlabel(labels["phi"])
    axes[2,2].set_xticks(phi_ticks)

    axes[0,1].set_visible(False);axes[0,2].set_visible(False);axes[1,2].set_visible(False)
    axes[0,0].set_xticklabels([]);axes[1,0].set_xticklabels([]);axes[1,1].set_xticklabels([])
    axes[1,1].set_yticklabels([]);axes[2,1].set_yticklabels([]);axes[2,2].set_yticklabels([])
    
    for ax in fig.axes:
        if ax.get_xlabel() != "":
            ax.xaxis.label.set_size(24)
        if ax.get_ylabel() != "":
            ax.yaxis.label.set_size(24)
    
    fig.suptitle("Inferred halo parameters for 100 bootstrapped realizations", fontsize=28)
    fig.text(.5,.91, subtitle, fontsize=24, ha='center')
    fig.subplots_adjust(hspace=0.04, wspace=0.04)
    
    return fig

def sgr_kde(particle_data, ax=None):
    """ TODO """
    
    if ax is None:
        fig,ax = plt.subplots(1,1)
    
    # hack to get every 25th particle still bound to Sgr, otherwise this peak
    #   dominates the whole density field
    idx = (np.fabs(particle_data['Lmflag']) == 1) & (particle_data['dist'] < 70) \
            & (particle_data['Pcol'] > -1)
    still_bound = np.zeros_like(idx)
    w, = np.where(particle_data['Pcol']==-1)
    still_bound[w[::25]] = 1.
    idx = idx | still_bound.astype(bool)
    
    extent = dict(x=(-75,75), y=(-75,75))
    X,Y = np.mgrid[extent['x'][0]:extent['x'][1]:400j, 
                   extent['y'][0]:extent['y'][1]:400j]
    positions = np.vstack([X.ravel(),Y.ravel()])
    
    Xsun = 8. # kpc
    m1,m2 = -particle_data[idx]['xgc']+Xsun, particle_data[idx]['zgc']
    values = np.vstack([m1, m2])
    
    kernel = gaussian_kde(values)
    Z = np.reshape(kernel(positions).T, X.shape).T
    
    ax.imshow(Z, interpolation="nearest", 
              extent=extent['x']+extent['y'],
              cmap=cm.bone, aspect='auto')
    
    return fig, ax