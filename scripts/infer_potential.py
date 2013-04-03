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
import multiprocessing

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
from streams.potential.lm10 import param_ranges
from streams.simulation import make_posterior
from streams.simulation.setup import simulation_setup

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

def infer_potential(ln_posterior, nwalkers=100, nsamples=100, nburn_in=100, 
                    params=[], output_path="/tmp", with_errors=True, mpi=False, 
                    nthreads=multiprocessing.cpu_count(), logger=None, 
                    description="", expr="", nparticles=100):
    
    """ Given a log posterior function, infer the parameters for the halo model. """
    
    if len(params) == 0:
        raise ValueError("No parameters specified!")
    
    tf_yn = dict(True="yes", False="no")
    # Create list of strings to write to run_parameters file
    run_parameters = []
    run_parameters.append("description: {0}".format(description))
    run_parameters.append("particle selection expr: {0}".format(expr))
    run_parameters.append("particles: {0}".format(nparticles))
    run_parameters.append("walkers: {0}".format(nwalkers))
    run_parameters.append("burn-in: {0}".format(nburn_in))
    run_parameters.append("samples: {0}".format(nsamples))
    run_parameters.append("halo parameters: {0}".format(", ".join(params)))
    run_parameters.append("use mpi? {0}".format(tf_yn[str(mpi)]))
    run_parameters.append("with simulated observational errors? {0}".\
                            format(tf_yn[str(with_errors)]))
    
    # Create the starting points for all walkers
    p0 = []
    for ii in range(nwalkers):
        p0.append([np.random.uniform(param_ranges[p_name][0], param_ranges[p_name][1])
                    for p_name in params])
    p0 = np.array(p0)
    ndim = p0.shape[1]

    if mpi:
        # Initialize the MPI pool
        pool = MPIPool()

        # Make sure the thread we're running on is the master
        if not pool.is_master():
            pool.wait()
            sys.exit(0)

        sampler = emcee.EnsembleSampler(nwalkers, ndim, ln_posterior, pool=pool)
    else:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, ln_posterior, threads=nthreads)
    
    # Create a new path for the output
    path = os.path.join(output_path, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    
    if not os.path.exists(path):
        os.mkdir(path)
    
    fig = plot_sgr_snap(sgr_snap)
    fig.savefig(os.path.join(path, "particles.png"))
    
    if nburn_in > 0:
        pos, prob, state = sampler.run_mcmc(p0, nburn_in)
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
        posterior_fig, trace_fig = plot_mcmc(params, acceptance_fraction, flatchain, chain)
        
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
    config_kwargs = simulation_setup()
    
    ln_posterior = make_posterior(config_kwargs["params"], 
                                  config_kwargs["sgr_snap"], 
                                  config_kwargs["sgr_cen"], 
                                  config_kwargs["dt"])
    infer_potential(ln_posterior, **config_kwargs)
    
    sys.exit(0)
