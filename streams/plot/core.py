# coding: utf-8

""" General plotting utilities. """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys
import cPickle as pickle

# Third-party
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as col
import matplotlib.cm as cm
import matplotlib.gridspec as gridspec
import astropy.units as u

from ..potential.lm10 import param_to_latex, param_ranges
from ..potential.lm10 import halo_params as true_halo_params

__all__ = ["discrete_cmap", "emcee_plot", "plot_sampler_pickle"]

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
            val = true_halo_params[param]
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