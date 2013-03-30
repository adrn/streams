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
from streams.simulation import back_integrate

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

def plot_mcmc(param_names, acceptance_fraction, flatchain, chain):
    """ Make MCMC plots: posterior and trace of chains. """
    
    ndim = chain.shape[2]
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
    
    return (posterior_fig, trace_fig)

def infer_potential(**config):
    nwalkers = config.get("nwalkers", 100)
    nsamples = config.get("nsamples", 1000)
    nburn_in = config.get("nburn_in", nsamples//10)
    param_names = config.get("params", [])
    output_path = config.get("output_path", "/tmp")
    errors = config.get("errors", False)
    mpi = config.get("mpi", False)
    
    if len(param_names) == 0:
        raise ValueError("No parameters specified!")
        
    # Create list of strings to write to run_parameters file
    run_parameters.append("walkers: {0}".format(nwalkers))
    run_parameters.append("burn-in: {0}".format(nburn_in))
    run_parameters.append("samples: {0}".format(nsamples))
    run_parameters.append("halo parameters: {0}".format(", ".join(param_names)))
    run_parameters.append("used mpi? {0}".format(str(mpi)))
    run_parameters.append("with simulated observational errors? {0}".format(str(errors)))
    
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
    
    # Create a new path for the output
    path = os.path.join(output_path, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    
    if not os.path.exists(path):
        #raise IOError("Whoa, '{0}' already exists. What did you do, go back in time?".format(path))
        os.mkdir(path)
    
    fig = plot_sgr_snap(sgr_snap)
    fig.savefig(os.path.join(path, "particles.png"))
    
    if nburn_in > 0:
        pos, prob, state = sampler.run_mcmc(p0, nburn_in)
        logger.debug("Burn in complete...")
        sampler.reset()
    else:
        pos = p0

    sampler.run_mcmc(pos, nsamples)
    logger.info("Median acceptance fraction: {0:.3f}".format(np.median(sampler.acceptance_fraction)))
    run_parameters.append("median acceptance fraction: {0:.3f}".format(np.median(sampler.acceptance_fraction)))
    if mpi: pool.close()

    data_file = os.path.join(path, "sampler_data.pickle")
    if os.path.exists(data_file):
        os.remove(data_file)

    sampler_pickle = (sampler.acceptance_fraction,sampler.flatchain,sampler.chain)
    fnpickle(sampler_pickle, data_file)
    acceptance_fraction,flatchain,chain = sampler_pickle

    idx = (acceptance_fraction > 0.1) & (acceptance_fraction < 0.6) # rule of thumb, bitches
    run_parameters.append("{0} walkers ({1:.1f}%) converged".format(sum(idx), sum(idx)/nwalkers*100))
    
    chain = chain[idx] 
    flatchain = []
    for walker in chain:
        flatchain += list(walker)
    flatchain = np.array(flatchain)

    # If chains converged, make mcmc plots
    if len(flatchain) == 0:
        logger.warning("Not making plots -- no chains converged!")
    else:
        posterior_fig, trace_fig = plot_mcmc(param_names, acceptance_fraction, flatchain, chain)
        
        posterior_fig.savefig(os.path.join(path, "mcmc_posterior.png"), format="png")
        trace_fig.savefig(os.path.join(path, "mcmc_trace.png"), format="png")
    
    # Save the run parameters
    with open(os.path.join(path, "run_parameters"), "w") as f:
        f.write("\n".join(run_parameters))
        
    return

def plot_sgr_snap(sgr_snap):
    """ Plot 3 projections of the initial particle positions. """
    
    fig, axes = plt.subplots(2,2,figsize=(10,10))
    
    axes[0,0].scatter(sgr_snap.x, sgr_snap.y, edgecolor="none", c="k", s=10)
    axes[1,0].scatter(sgr_snap.x, sgr_snap.z, edgecolor="none", c="k", s=10)
    axes[1,1].scatter(sgr_snap.y, sgr_snap.z, edgecolor="none", c="k", s=10)
    
    axes[0,0].set_ylabel("{0} [{1}]".format("Y", sgr_snap.r_unit))
    axes[1,0].set_xlabel("{0} [{1}]".format("X", sgr_snap.r_unit))
    axes[1,0].set_ylabel("{0} [{1}]".format("Z", sgr_snap.r_unit))
    axes[1,1].set_xlabel("{0} [{1}]".format("Y", sgr_snap.r_unit))
    
    axes[0,1].set_visible(False)
    fig.subplots_adjust(hspace=0.1, wspace=0.1, left=0.08, bottom=0.08, top=0.9, right=0.9 )
    return fig

if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser(description="")
    parser.add_argument("-v", "--verbose", action="store_true", dest="verbose", default=False,
                    help="Be chatty! (default = False)")
    parser.add_argument("-q", "--quiet", action="store_true", dest="quiet", default=False,
                    help="Be quiet! (default = False)")
                    
    parser.add_argument("--mpi", action="store_true", dest="mpi", default=False,
                    help="Anticipate being run with MPI.")
    parser.add_argument("--errors", action="store_true", dest="errors", default=False,
                    help="Run with observational errors!")
    parser.add_argument("--seed", dest="seed", default=42, type=int,
                    help="Seed the random number generator.")
                    
    parser.add_argument("--walkers", dest="nwalkers", default=100, type=int,
                    help="Number of walkers")
    parser.add_argument("--steps", dest="nsamples", default=1000, type=int,
                    help="Number of steps to take")
    parser.add_argument("--burn-in", dest="nburn_in", type=int, default=100,
                    help="Number of steps to burn in")

    parser.add_argument("--params", dest="params", default=[], nargs='+',
                    action='store', help="The halo parameters to vary.")
    parser.add_argument("--expr", dest="expr", default=[], 
                    action='append', help="Selection expression for particles.")
                    
    parser.add_argument("--output-path", dest="output_path", default="/u/10/a/amp2217/public_html/plots",
                    help="The path to store output.")

    args = parser.parse_args()

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
    
    # Interpolate SgrCen data onto new times
    ts = np.arange(t2, t1, -dt)*u.Myr
    sgr_cen.interpolate(ts)

    np.random.seed(args.seed)
    run_parameters = []
    
    # default expression is to only select unbound particles
    expr = "(tub > 10.)"
    if len(args.expr) > 0:
        expr += " & " + " & ".join(["({0})".format(x) for x in args.expr])
    run_parameters.append("particle selection expr: {0}".format(expr))
    
    sgr_snap = SgrSnapshot(num=100, 
                           expr=expr)
    
    if args.errors:
        sgr_snap.add_errors()
    
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

    def ln_likelihood(p):
        halo_params = true_halo_params.copy()
        for ii in range(len(p)):
            halo_params[param_map[ii]] = p[ii]

        mw_potential = LawMajewski2010(**halo_params)
        return -back_integrate(mw_potential, sgr_snap, sgr_cen, dt)

    infer_potential(**args.__dict__)
    
    sys.exit(0)
