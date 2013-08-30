# coding: utf-8

""" """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys
import cPickle as pickle

# Third-party
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
from astropy.io.misc import fnunpickle
import triangle

from streams.plot.emcee import emcee_plot
    
if __name__ == "__main__":
    from argparse import ArgumentParser
    
    parser = ArgumentParser(description="")
    parser.add_argument("-f", "--file", dest="file", required=True,
                    help="Path to the sampler pickle file.")
    parser.add_argument("--params", dest="params", default=None, nargs='+', 
                    required=True, action='store', help="The halo parameters.")
    parser.add_argument("-s", "--source", dest="particle_source", required=True,
                        help="The source of the particles, e.g., lm10 or pal5")
    parser.add_argument("--show-true", dest="show_true", action="store_true",
                    default=False, help="Plot the true halo parameter values.")
    
    args = parser.parse_args()
    
    if args.particle_source.lower() == 'lm10':
        from streams.potential.lm10 import _true_params, param_to_latex
    elif args.particle_source.lower() == 'pal5':
        from streams.potential.pal5 import _true_params, param_to_latex
    else:
        raise ValueError("Invalid particle source {0}"
                         .format(config["particle_source"]))
    
    sampler = fnunpickle(args.file)
    xs = sampler.chain
    
    if args.show_true:
        truths = [_true_params[p] for p in args.params]
    else:
        truths = None
    
    # make triangle plot with 5-sigma ranges
    extents = []
    for ii,param in enumerate(args.params):
        mu = np.median(sampler.flatchain[:,ii])
        sigma = np.std(sampler.flatchain[:,ii])
        extents.append((mu-5*sigma,mu+5*sigma))
    
    labels = [param_to_latex[p] for p in args.params]
    
    fig1 = emcee_plot(xs, labels=labels, truths=truths, extents=extents)
    fig2 = triangle.corner(sampler.flatchain, labels=args.params,
                           extents=extents, truths=truths,
                           quantiles=[0.16,0.5,0.84])
    
    plt.show()