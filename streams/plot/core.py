# coding: utf-8

""" General plotting utilities. """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys

# Third-party
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from ..potential.lm10 import param_to_latex
from ..potential.lm10 import halo_params as true_halo_params

__all__ = ["emcee_plot"]

# TODO: make this more general...
def emcee_plot(sampler, params, converged_idx, acceptance_fraction_bounds=(None,None)):
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
    flatchain = []
    for walker in chain:
        flatchain += list(walker)
    flatchain = np.array(flatchain)
    
    # The halo velocity (v_halo) parameter is stored in units of kpc/Myr, but I
    #   want to plot it in km/s
    if "v_halo" in params:
        ii = params.index("v_halo")
        #for xx in range(chain.shape[0]):
        #    for yy in range(chain.shape[1]):
        chain[:,:,ii] = (chain[:,:,ii]*u.kpc/u.Myr).to(u.km/u.s).value
    
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
        
        # Create a histogram of all values past the converged point. Make 100 bins
        #   between the y-axis bounds defined by the 'walkers' plot.
        ax2.hist(np.ravel(these_chains[:,converged_idx:]), 
                 bins=np.linspace(ax1.get_ylim()[0],ax1.get_ylim()[1],100),
                 orientation='horizontal',
                 facecolor="#67A9CF",
                 edgecolor="none")
        ax2.axhline(true_halo_params[param], 
                    color="#555555",
                    linestyle="--")
        
        # Same y-bounds as the walkers plot, so they line up
        ax1.set_ylim(np.min(these_chains[:,0]), np.max(these_chains[:,0]))
        ax2.set_ylim(ax1.get_ylim())
        ax2.xaxis.set_visible(False)
        ax2.yaxis.tick_right()
        
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
            ax2.set_yticks(ax2.get_yticks()[1:-2])
        else:
            ax2.set_yticks(ax2.get_yticks()[1:-1])
    
    fig.subplots_adjust(hspace=0.02, wspace=0.0, bottom=0.075, top=0.9, left=0.12, right=0.88)
    return fig