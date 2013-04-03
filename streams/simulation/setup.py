# coding: utf-8

""" Import this script to setup the default arguments for a back-integration 
    simulation. It will handle the argparse stuff for you! 
"""

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys
import logging

# Third-party
import numpy as np
import astropy.units as u

from ..data import SgrSnapshot, SgrCen

def simulation_setup():
    from argparse import ArgumentParser

    parser = ArgumentParser(description="")
    parser.add_argument("-v", "--verbose", action="store_true", dest="verbose", 
                    default=False, help="Be chatty! (default = False)")
    parser.add_argument("-q", "--quiet", action="store_true", dest="quiet", 
                    default=False, help="Be quiet! (default = False)")
                    
    parser.add_argument("--mpi", action="store_true", dest="mpi", default=False,
                    help="Anticipate being run with MPI.")
    parser.add_argument("--threads", dest="nthreads", type=int,
                    help="If not using MPI, how many threads to spawn.")
    parser.add_argument("--with-errors", action="store_true", dest="with_errors", 
                    default=False, help="Run with observational errors!")
    parser.add_argument("--seed", dest="seed", default=42, type=int,
                    help="Seed the random number generator.")
                    
    parser.add_argument("--walkers", dest="nwalkers", default=100, type=int,
                    help="Number of walkers")
    parser.add_argument("--steps", dest="nsamples", default=100, type=int,
                    help="Number of steps to take")
    parser.add_argument("--burn-in", dest="nburn_in", type=int, default=100,
                    help="Number of steps to burn in")
    parser.add_argument("--particles", dest="nparticles", default=100, type=int,
                    help="Number of particles")

    parser.add_argument("--params", dest="params", default=[], nargs='+',
                    action='store', help="The halo parameters to vary.")
    parser.add_argument("--expr", dest="expr", default=[], 
                    action='append', help="Selection expression for particles.")
                    
    parser.add_argument("--output-path", dest="output_path", default="/tmp",
                    help="The path to store output.")
    parser.add_argument("--desc", dest="description", default="None",
                    help="An optional description to add to the run_parameters file.")

    args = parser.parse_args()
    config_dict = args.__dict__.copy()
    
    # Create logger
    logger = logging.getLogger(__name__)
    ch = logging.StreamHandler()
    formatter = logging.Formatter("%(name)s / %(levelname)s / %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    config_dict["logger"] = logger
    
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    elif args.quiet:
        logger.setLevel(logging.ERROR)
    else:
        logger.setLevel(logging.INFO)
        
    # Read in data from Kathryn's SGR_SNAP and SGR_CEN files
    sgr_cen = SgrCen()

    # Get timestep information from SGR_CEN
    t1 = min(sgr_cen.t)
    t2 = max(sgr_cen.t)
    dt = sgr_cen.dt[0]*10
    config_dict["dt"] = dt
    
    # Interpolate SgrCen data onto new times
    ts = np.arange(t2, t1, -dt)*u.Myr
    sgr_cen.interpolate(ts)
    config_dict["sgr_cen"] = sgr_cen
    
    np.random.seed(args.seed)
        
    # default expression is to only select unbound particles
    expr = "(tub > 10.)"
    if len(args.expr) > 0:
        expr += " & " + " & ".join(["({0})".format(x) for x in args.expr])
    config_dict["expr"] = expr
    
    sgr_snap = SgrSnapshot(num=args.nparticles, 
                           expr=expr)
    
    if args.errors:
        sgr_snap.add_errors()
    
    config_dict["sgr_snap"] = sgr_snap
    
    return config_dict