# coding: utf-8

""" Create a scatter-plot matrix using Matplotlib. """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys

# Third-party
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u

__all__ = ['scatter_plot_matrix']

def scatter_plot_matrix(obj, labels=None, axes=None, subplots_kwargs=dict(),
                            scatter_kwargs=dict(), triangle="lower"):
    """ Create a scatter plot matrix from the given data. 
        
        Parameters
        ----------
        obj : numpy.ndarray
            A numpy array containined the scatter data to plot. The data
            should be shape MxN where M is the number of dimensions and 
            and N data points.
        labels : numpy.ndarray (optional)
            A numpy array of length M containing the axis labels.
        axes : matplotlib Axes array (optional)
            If you've already created the axes objects, pass this in to
            plot the data on that.
        subplots_kwargs : dict (optional)
            A dictionary of keyword arguments to pass to the 
            matplotlib.pyplot.subplots call. Note: only relevant if axes=None.
        scatter_kwargs : dict (optional)
            A dictionary of keyword arguments to pass to the 
            matplotlib.pyplot.scatter function calls. 
        triangle : str (optional)
            Upper right triangle of plots or lower left triangle, since
            it's symmetric.
    """
    
    if isinstance(obj, u.Quantity):
        data = obj.value
    else:
        data = obj
    
    try:
        M,N = data.shape
    except ValueError: # too many values to unpack
        raise ValueError("Invalid data shape {0}. You must pass in an array of "
                         "shape (M, N) where N should be >> M.".format(data.shape))
    
    if labels == None:
        labels = [None]*M
    
    if axes == None:
        skwargs = subplots_kwargs.copy()
        skwargs["sharex"] = True if not skwargs.has_key("sharex") else skwargs["sharex"]
        skwargs["sharey"] = True if not skwargs.has_key("sharey") else skwargs["sharey"]
        
        fig, axes = plt.subplots(M-1, M-1, **skwargs)
    
    sc_kwargs = scatter_kwargs.copy()
    sc_kwargs["edgecolor"] = "none" if not sc_kwargs.has_key("edgecolor") else sc_kwargs["edgecolor"]
    sc_kwargs["c"] = "k" if not sc_kwargs.has_key("c") else sc_kwargs["c"]
    sc_kwargs["s"] = 10 if not sc_kwargs.has_key("s") else sc_kwargs["s"]
    
    xticks = yticks = None
    for ii in range(M-1):
        i = ii+1
        for jj in range(M-1):
            if triangle == "lower":
                if ii < jj:
                    axes[ii,jj].set_visible(False)
                    continue
            else:
                if ii > jj:
                    axes[ii,jj].set_visible(False)
                    continue
                
            axes[ii,jj].scatter(data[jj], data[i], **sc_kwargs)
            
            if yticks == None:
                yticks = axes[ii,jj].get_yticks()[1:-1]
            
            if xticks == None:
                xticks = axes[ii,jj].get_xticks()[1:-1]
            
            # first column
            if jj == 0 and i != 0:
                axes[ii,jj].set_ylabel(labels[i])
                
                # Hack so ticklabels don't overlap
                axes[ii,jj].yaxis.set_ticks(yticks)
            
            # last row
            if ii == M-2:
                axes[ii,jj].set_xlabel(labels[jj])

                # Hack so ticklabels don't overlap
                axes[ii,jj].xaxis.set_ticks(xticks)
    
    fig = axes[0,0].figure
    fig.subplots_adjust(hspace=0.05, wspace=0.05, left=0.08, bottom=0.08, top=0.9, right=0.9 )
    return fig, axes