# coding: utf-8

""" A script for making figures for our Streams Paper 1 """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys
import cPickle as pickle
import inspect

# Third-party
import astropy.units as u
from astropy.io.misc import fnpickle, fnunpickle
import daft
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib import rc_context, rcParams, cm
from matplotlib.patches import Rectangle, Ellipse
import scipy.optimize as so

from streams.util import project_root

matplotlib.rc('xtick', labelsize=18)
matplotlib.rc('ytick', labelsize=18)
#matplotlib.rc('text', usetex=True)
matplotlib.rc('axes', edgecolor='#444444', labelsize=24,
              labelweight=400, linewidth=2.0)
matplotlib.rc('lines', markeredgewidth=0)
matplotlib.rc('font', family='Source Sans Pro')
#matplotlib.rc('savefig', bbox='standard')

plot_path = "plots/paper2/"
if not os.path.exists(plot_path):
    os.mkdir(plot_path)

def graphical_model():
    pass


if __name__ == '__main__':
    from argparse import ArgumentParser
    import logging

    # Create logger
    logger = logging.getLogger(__name__)
    ch = logging.StreamHandler()
    formatter = logging.Formatter("%(name)s / %(levelname)s / %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # Define parser object
    parser = ArgumentParser(description="")
    parser.add_argument("-v", "--verbose", action="store_true",
                        dest="verbose", default=False,
                        help="Be chatty! (default = False)")
    parser.add_argument("-q", "--quiet", action="store_true", dest="quiet",
                        default=False, help="Be quiet! (default = False)")
    parser.add_argument("-l", "--list", action="store_true", dest="list",
                        default=False, help="List all functions")
    parser.add_argument("-o", "--overwrite", action="store_true",
                        dest="overwrite",  default=False,
                        help="Overwrite existing files.")
    parser.add_argument("-f", "--function", dest="function", type=str,
                        help="The name of the function to execute.")
    parser.add_argument("--kwargs", dest="kwargs", nargs="+", type=str,
                       help="kwargs passed in to whatever function you call.")

    args = parser.parse_args()
    try:
        kwargs = dict([tuple(k.split("=")) for k in args.kwargs])
    except TypeError:
        kwargs = dict()

    kwargs["overwrite"] = args.overwrite

    # Set logger level based on verbose flags
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    elif args.quiet:
        logger.setLevel(logging.ERROR)
    else:
        logger.setLevel(logging.INFO)

    def _print_funcs():
        fs = inspect.getmembers(sys.modules[__name__],
                                lambda member: inspect.isfunction(member) and member.__module__ == __name__ and not member.__name__.startswith("_"))
        print("\n".join([f[0] for f in fs]))

    if args.list:
        print("="*79)
        _print_funcs()
        print("="*79)
        sys.exit(0)

    func = getattr(sys.modules[__name__], args.__dict__.get("function"))
    func(**kwargs)

