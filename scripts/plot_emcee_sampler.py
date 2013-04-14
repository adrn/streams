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
import matplotlib.gridspec as gridspec
import astropy.units as u

from streams.plot import plot_sampler_pickle
    
if __name__ == "__main__":
    from argparse import ArgumentParser
    
    parser = ArgumentParser(description="")
    parser.add_argument("-f", "--file", dest="file", required=True,
                    help="Path to the sampler pickle file.")
    parser.add_argument("--params", dest="params", default=[], nargs='+', 
                    required=True, action='store', help="The halo parameters.")
    parser.add_argument("-o", "--output-file", dest="output_file", required=True,
                    help="Path to save the plot.")
    parser.add_argument("-a", "--acceptance-fraction", dest="acc_frac", nargs=2,
                    type=float, default=(None,None), help="Allowed range of "
                                "acceptance fractions for 'good' walkers.")
    parser.add_argument("--show-true", dest="show_true", action="store_true",
                    default=False, help="Plot the true halo parameter values.")
    
    args = parser.parse_args()
    
    fig = plot_sampler_pickle(args.file, args.params, 
                              acceptance_fraction_bounds=args.acc_frac, 
                              show_true=args.show_true)
    fig.savefig(args.output_file)
    