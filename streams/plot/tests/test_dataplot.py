# coding: utf-8

from __future__ import absolute_import, unicode_literals, division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys

# Third-party
import numpy as np
import pytest

from ..data import scatter_plot_matrix

def test_scatter_plot_matrix():
    M = 6
    data = np.random.normal(size=(M,100))
        
    labels = ["x", "y", "z", "vx", "vy", "vz"]
    axes = scatter_plot_matrix(data, labels=labels)
    fig = axes[0,0].figure
    fig.savefig("plots/tests/scatterplot_matrix.png")
    
    axes = scatter_plot_matrix(data, labels=labels, 
                               subplots_kwargs={"figsize":(16,16)},
                               scatter_kwargs={"edgecolor":"none",  
                                               "c":"k",
                                               "s":10,
                                               "alpha":0.5})
    fig = axes[0,0].figure
    fig.savefig("plots/tests/scatterplot_matrix_sexier.png")
    