# coding: utf-8

from __future__ import absolute_import, unicode_literals, division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys

# Third-party
import numpy as np
import pytest

from ..data import scatter_plot_matrix

plot_path = "plots/tests/plot/"

if not os.path.exists(plot_path):
    os.mkdir(plot_path)

def test_scatter_plot_matrix():
    M = 6
    data = np.random.normal(size=(M,100))
        
    labels = ["x", "y", "z", "vx", "vy", "vz"]
    fig,axes = scatter_plot_matrix(data, labels=labels)
    fig = axes[0,0].figure
    fig.savefig(os.path.join(plot_path,"scatterplot_matrix.png"))
    
    fig,axes = scatter_plot_matrix(data, labels=labels, 
                               subplots_kwargs={"figsize":(16,16)},
                               scatter_kwargs={"edgecolor":"none",  
                                               "c":"k",
                                               "s":10,
                                               "alpha":0.5})
    fig = axes[0,0].figure
    fig.savefig(os.path.join(plot_path,"scatterplot_matrix_sexier.png"))
    