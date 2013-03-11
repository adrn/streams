#!/hpc/astro/users/amp2217/yt-x86_64/bin/python
# coding: utf-8

""" In this module, I'll use a simple function minimizer to minimize the var
    of the distribution of energy distances over various galaxy potential
    parameters.
"""

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys
import datetime
import logging

# Create logger
logger = logging.getLogger(__name__)
ch = logging.StreamHandler()
formatter = logging.Formatter("%(name)s / %(levelname)s / %(message)s")
ch.setFormatter(formatter)
logger.addHandler(ch)

# Third-party
import numpy as np
np.seterr(all="ignore")
import scipy
scipy.seterr(all="ignore")
from scipy import interpolate, optimize
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
import astropy.units as u
from scipy.optimize import fmin_powell

# Project
from streams.data import SgrSnapshot, SgrCen
from streams.potential import *
from streams.integrate import leapfrog
from streams.simulation import Particle, ParticleSimulation, run_back_integration

def minimize_potential(objective, **config):
    maxfev = config.get("maxfev", 10000)
    param_names = config.get("params", [])
    plot_path = config.get("plot_path", "plots")
    
    if len(param_names) == 0:
        raise ValueError("No parameters specified!")
    
    x0 = [np.random.uniform(param_ranges[p_name][0], param_ranges[p_name][1])
            for p_name in param_names]
    
    logger.info("Starting at: {0}".format(np.array(x0)))
    
    result = fmin_powell(objective, x0=x0, maxfun=maxfev, disp=True, retall=True, full_output=True)
    logger.info("Finished minimizing...")
    logger.info("Final parameters:")
    for ii,param in enumerate(param_names):
        logger.info("{0}: {1} // True: {2} ({3}% error)".format(param, result[0][ii], 
                true_halo_params[param], 
                100*abs(result[0][ii]-true_halo_params[param])/true_halo_params[param]))
    
    #allvecs = np.array(result[-1])
    #plt.plot([true_halo_params[param_names[0]]], [true_halo_params[param_names[0]]], color="r", marker="o")
    #plt.plot(allvecs[:,0], allvecs[:,1], color="k", marker="o", alpha=0.6)
    #plt.savefig(os.path.join(plot_path, "test_minimize_{0}_{1}.png".format(datetime.datetime.now().date(), "_".join(param_names))))
    
    return
    
    fig,axes = plt.subplots(len(x0), 1, figsize=(14,5*(len(x0)+1)))
    for ii in range(len(x0)):
        axes[ii].set_title(param_names[ii])
        axes[ii].axvline(true_halo_params[param_names[ii]], color="k", linestyle="-", linewidth=2)
        axes[ii].axvline(result.x[ii], color="r", linestyle="--", linewidth=2)

    #plt.savefig("/u/10/a/amp2217/public_html/plots/posterior_{0}.png".format(datetime.datetime.now().date()))
    plt.savefig(os.path.join(plot_path, "minimize_{0}_{1}.png".format(datetime.datetime.now().date(), "_".join(param_names))))

    return

if __name__ == "__main__":
    from argparse import ArgumentParser
    
    parser = ArgumentParser(description="")
    parser.add_argument("-v", "--verbose", action="store_true", dest="verbose", default=False,
                    help="Be chatty! (default = False)")
    parser.add_argument("-q", "--quiet", action="store_true", dest="quiet", default=False,
                    help="Be quiet! (default = False)")

    parser.add_argument("--maxfev", dest="maxfev", default=10000, type=int,
                    help="Number of function evaluations")
    
    parser.add_argument("--params", dest="params", default=[], nargs='+',
                    action='store', help="The halo parameters to vary.")
    parser.add_argument("--plot-path", dest="plot_path", default="plots",
                    help="The path to store plots.")
    
    args = parser.parse_args()
    
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    elif args.quiet:
        logger.setLevel(logging.ERROR)
    else:
        logger.setLevel(logging.INFO)
    
    # --------------------------------------------------------------------
    # This could get messy (TM)
    # --------------------------------------------------------------------
    
    # Read in data from Kathryn's SGR_SNAP and SGR_CEN files
    sgr_cen = SgrCen()
    
    # Get timestep information from SGR_CEN
    t1 = min(sgr_cen.data["t"])
    t2 = max(sgr_cen.data["t"])
    dt = sgr_cen.data["dt"][0]*10
    
    # Interpolate SgrCen data onto new times
    ts = np.arange(t2, t1, -dt)
    sgr_cen.interpolate(ts)
    
    np.random.seed(42)
    sgr_snap = SgrSnapshot(num=100, no_bound=True) # randomly sample 100 particles
    
    true_halo_params = dict(v_halo=(121.858*u.km/u.s).to(u.kpc/u.Myr).value,
                            q1=1.38,
                            q2=1.0,
                            qz=1.36,
                            phi=1.692969,
                            c=12.)
    
    # Define a mapping from parameter name to index
    param_map = dict(zip(range(len(args.params)), args.params))
    param_ranges = dict(qz=(0.5,2),
                        q1=(0.5,2),
                        q2=(0.5,2),
                        v_halo=((100*u.km/u.s).to(u.kpc/u.Myr).value, (200*u.km/u.s).to(u.kpc/u.Myr).value),
                        phi=(1, 2.5),
                        c=(5,20))
    
    def objective(p):    
        halo_params = true_halo_params.copy()
        for ii in range(len(p)):
            halo_params[param_map[ii]] = p[ii]
        halo_potential = LogarithmicPotentialLJ(**halo_params)
    
        return run_back_integration(halo_potential, sgr_snap, sgr_cen, dt)
    
    minimize_potential(objective, **args.__dict__)
    
