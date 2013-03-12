#!/hpc/astro/users/amp2217/yt-x86_64/bin/python
# coding: utf-8

""" In this module, I'll test how the distribution of 'energy distance' changes as I tweak
    various galaxy potential parameters. Ultimately, I want to come up with a way to evaluate
    the 'best' potential.
"""

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys
import datetime
import logging
import multiprocessing
    
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
from scipy import interpolate
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
import astropy.units as u
from astropy.io.misc import fnpickle, fnunpickle
import emcee
from emcee.utils import MPIPool

# Project
from streams.data import SgrSnapshot, SgrCen
from streams.potential import LawMajewski2010
from streams.potential.lm10 import halo_params as true_halo_params
from streams.simulation import back_integrate, back_integrate_with_errors

def ln_p_qz(qz):
    """ Prior on vertical (z) axis ratio """
    if qz <= 1 or qz >= 2:
        return -np.inf
    else:
        return 0.

def ln_p_q1(q1):
    """ Prior on axis ratio """
    if q1 <= 1 or q1 >= 2:
        return -np.inf
    else:
        return 0.

def ln_p_q2(q2):
    """ Prior on axis ratio """
    if q2 <= 0.5 or q2 >= 2:
        return -np.inf
    else:
        return 0.

def ln_p_v_halo(v):
    """ Prior on mass of the halo (v_halo). The range imposed is roughly a
        halo mass between 10^10 and 10^12 M_sun at 200 kpc
    """
    if v <= 0.01 or v >= 0.15:
        return -np.inf
    else:
        return 0.

def ln_p_phi(phi):
    """ Prior on orientation angle between DM halo and disk """
    if phi < 1. or phi > 2.5:
        return -np.inf
    else:
        return 0.

def ln_p_r_halo(r_halo):
    """ Prior on halo concentration parameter """
    if r_halo < 5. or r_halo > 20:
        return -np.inf
    else:
        return 0.

def ln_posterior(p):
    return ln_prior(p) + ln_likelihood(p)

def infer_potential(**config):
    nwalkers = config.get("nwalkers", 100)
    nsamples = config.get("nsamples", 1000)
    nburn_in = config.get("nburn_in", nsamples//10)
    param_names = config.get("params", [])
    plot_path = config.get("plot_path", "plots")
    data_path = config.get("data_path", "data")
    overwrite = config.get("overwrite", False)
    errors = config.get("errors", False)
    mpi = config.get("mpi", False)
    
    if errors:
        file_str = "{0}_{1}_w{2}_s{3}_errors".format(datetime.datetime.now().isoformat("_").replace(":","-"),
                                   "_".join(param_names), nwalkers, nsamples)
    else:
        file_str = "{0}_{1}_w{2}_s{3}".format(datetime.datetime.now().isoformat("_").replace(":","-"),
                                   "_".join(param_names), nwalkers, nsamples)
    
    if len(param_names) == 0:
        raise ValueError("No parameters specified!")
    
    logger.info("Inferring halo parameters: {0}".format(",".join(param_names)))
    logger.info("--> Starting simulation with {0} walkers.".format(nwalkers))
    logger.info("--> Will burn in for {0} steps, then take {1} steps.".format(nburn_in, nsamples))
    
    p0 = []
    for ii in range(nwalkers):
        p0.append([np.random.uniform(param_ranges[p_name][0], param_ranges[p_name][1])
                    for p_name in param_names])
    p0 = np.array(p0)
    ndim = p0.shape[1]
    
    if mpi:
        logger.info("Running with MPI!")
        # Initialize the MPI pool
        pool = MPIPool()
        
        # Make sure the thread we're running on is the master
        if not pool.is_master():
            pool.wait()
            sys.exit(0)
        
        sampler = emcee.EnsembleSampler(nwalkers, ndim, ln_posterior, pool=pool)
    else:
        logger.info("Running WITHOUT MPI on {0} cores".format(multiprocessing.cpu_count()))
        sampler = emcee.EnsembleSampler(nwalkers, ndim, ln_posterior, threads=multiprocessing.cpu_count())
    
    if nburn_in > 0:
        pos, prob, state = sampler.run_mcmc(p0, nburn_in)
        logger.debug("Burn in complete...")
        sampler.reset()
    else:
        pos = p0
        
    sampler.run_mcmc(pos, nsamples)
    logger.info("Median acceptance fraction: {0:.3f}".format(np.median(sampler.acceptance_fraction)))
    
    data_file = os.path.join(data_path, "{0}.pickle".format(file_str))
    if os.path.exists(data_file):
        os.remove(data_file)
    
    sampler_pickle = (sampler.acceptance_fraction,sampler.flatchain,sampler.chain)
    fnpickle(sampler_pickle, data_file)
    acceptance_fraction,flatchain,chain = sampler_pickle
    
    chain = chain[(acceptance_fraction > 0.1) & (acceptance_fraction < 0.6)] # rule of thumb, bitches
    flatchain = []
    for walker in chain:
        flatchain += list(walker)
    flatchain = np.array(flatchain)    
    
    posterior_fig,posterior_axes = plt.subplots(ndim, 1, figsize=(14,5*(ndim+1)))
    trace_fig,trace_axes = plt.subplots(ndim, 1, figsize=(14,5*(ndim+1)))
    
    posterior_fig.suptitle("Median acceptance fraction: {0:.3f}".format(np.median(acceptance_fraction)))
    trace_fig.suptitle("Median acceptance fraction: {0:.3f}".format(np.median(acceptance_fraction)))
    for ii in range(ndim):
        posterior_axes[ii].set_title(param_names[ii])
        posterior_axes[ii].hist(flatchain[:,ii], bins=25, color="k", histtype="step", alpha=0.75)
        posterior_axes[ii].axvline(true_halo_params[param_names[ii]], color="r", linestyle="--", linewidth=2)
        
        trace_axes[ii].set_title(param_names[ii])
        for k in range(chain.shape[0]):
            trace_axes[ii].plot(np.arange(len(chain[k,:,ii])),
                                chain[k,:,ii], color="k", drawstyle="steps", alpha=0.2)
    
    posterior_fig.savefig(os.path.join(plot_path, "posterior_{0}.png".format(file_str)), format="png")
    trace_fig.savefig(os.path.join(plot_path, "trace_{0}.png".format(file_str)), format="png")
    
    if mpi: pool.close()
    sys.exit(0)
    
    return

if __name__ == "__main__":
    from argparse import ArgumentParser
    
    parser = ArgumentParser(description="")
    parser.add_argument("-v", "--verbose", action="store_true", dest="verbose", default=False,
                    help="Be chatty! (default = False)")
    parser.add_argument("-q", "--quiet", action="store_true", dest="quiet", default=False,
                    help="Be quiet! (default = False)")
    parser.add_argument("-o", "--overwrite", action="store_true", dest="overwrite", default=False,
                    help="Nom nom nom, old files...")
    parser.add_argument("--mpi", action="store_true", dest="mpi", default=False,
                    help="Anticipate being run with MPI.")
    parser.add_argument("--errors", action="store_true", dest="errors", default=False,
                    help="Run with observational errors!")
    
    parser.add_argument("--walkers", dest="nwalkers", default=100, type=int,
                    help="Number of walkers")
    parser.add_argument("--steps", dest="nsamples", default=1000, type=int,
                    help="Number of steps to take")
    parser.add_argument("--burn-in", dest="nburn_in", type=int, default=100,
                    help="Number of steps to burn in")
    
    parser.add_argument("--params", dest="params", default=[], nargs='+',
                    action='store', help="The halo parameters to vary.")
    parser.add_argument("--plot-path", dest="plot_path", default="/u/10/a/amp2217/public_html/plots",
                    help="The path to store plots.")
    parser.add_argument("--data-path", dest="data_path", default="/hpc/astro/users/amp2217/projects/streams/data",
                    help="The path to store data files.")
    parser.add_argument("--sampler-file", dest="sampler_file", 
                        help="Specify a pickle file containing a pre-pickled sampler object.")
    
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
    
    # Define a mapping from parameter name to index
    param_map = dict(zip(range(len(args.params)), args.params))
    param_ranges = dict(qz=(0.5,2),
                        q1=(0.5,2),
                        q2=(0.5,2),
                        v_halo=((100*u.km/u.s).to(u.kpc/u.Myr).value, (200*u.km/u.s).to(u.kpc/u.Myr).value),
                        phi=(1, 2.5),
                        r_halo=(5,20))
    
    # Construct the prior based on the requested parameters
    prior_map = dict(qz=ln_p_qz, q1=ln_p_q1, q2=ln_p_q2, v_halo=ln_p_v_halo, phi=ln_p_phi, r_halo=ln_p_r_halo)
    
    def ln_prior(p):
        sum = 0
        for ii in range(len(p)):
            sum += prior_map[param_map[ii]](p[ii])
        return sum
    
    if args.errors:
        back_integrate = back_integrate_with_errors
        
    def ln_likelihood(p):    
        halo_params = true_halo_params.copy()
        for ii in range(len(p)):
            halo_params[param_map[ii]] = p[ii]
    
        mw_potential = LawMajewski2010(**halo_params)
        return -back_integrate(mw_potential, sgr_snap, sgr_cen, dt)
    
    infer_potential(**args.__dict__)
    
