# coding: utf-8

""" """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys
import datetime
import cPickle as pickle

# Third-party
import numpy as np
np.seterr(all="ignore")
import scipy
scipy.seterr(all="ignore")
import astropy.units as u
from astropy.io.misc import fnpickle, fnunpickle
import emcee
from emcee.utils import MPIPool

# Project
from streams.potential.lm10 import param_ranges
from streams.inference import ln_posterior
from streams.plot import plot_sampler_pickle

__all__ = ["infer_potential"]

def infer_potential(particles, satellite_orbit, simulation_params):
    """ """
    
    # Shorthand!
    sp = simulation_params
    
    # Create the starting points for all walkers
    p0 = []
    for ii in range(sp["walkers"]):
        p0.append([np.random.uniform(param_ranges[p_name][0], param_ranges[p_name][1])
                    for p_name in sp["model_parameters"]])
    p0 = np.array(p0)
    ndim = len(sp["model_parameters"])
    
    # Construct the log posterior probability function to pass in to emcee
    args = sp["model_parameters"], particles, satellite_orbit
    
    if sp["mpi"]:
        # Initialize the MPI pool
        pool = MPIPool()

        # Make sure the thread we're running on is the master
        if not pool.is_master():
            pool.wait()
            sys.exit(0)

        sampler = emcee.EnsembleSampler(sp["walkers"], ndim, ln_posterior, 
                                        pool=pool, args=args)
    else:
        if "threads" in sp.keys():
            threads = sp["threads"]
        else:
            threads = 1
            
        sampler = emcee.EnsembleSampler(sp["walkers"], ndim, 
                                        ln_posterior, args=args,
                                        threads=threads)
    
    logger.info("About to start simulation with parameters: \n{0}"
                .format("\n\t".join(["{0}: {1}".format(k,v) for k,v in sp.items()])))
    
    # Create a new path for the output
    path = os.path.join(sp["output_path"], 
                        datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    
    logger.debug("Output path: {0}".format(path))
    
    if sp["make_plots"]:
        if not os.path.exists(path):
            os.mkdir(path)
    
        # Plot the initial positions of the particles in galactic XYZ coordinates
        fig,axes = particles.plot_positions()
        fig.savefig(os.path.join(path, "particles.png"))
    
    # If a burn-in period is requested, run the sampler for nburn_in steps then
    #   reset the walkers and use the end positions as new initial conditions
    if sp["burn_in"] > 0:
        pos, prob, state = sampler.run_mcmc(p0, sp["burn_in"])
        sampler.reset()
    else:
        pos = p0
    
    sampler.run_mcmc(pos, sp["steps"])
    #run_parameters.append("median acceptance fraction: {0:.3f}".\
    #                        format(np.median(sampler.acceptance_fraction)))
    
    # if we're running with MPI, we have to close the processor pool, otherwise
    #   the script will never finish running until the end of timmmmeeeee (echo)
    if sp["mpi"]: pool.close()
    
    data_file = os.path.join(path, "sampler_data.pickle")
    
    sampler.lnprobfn = None
    sampler.pool = None
    #sampler.pickle(data_file)
    fnpickle(sampler, data_file)

    #idx = (sampler.acceptance_fraction > 0.1) & \
    #        (sampler.acceptance_fraction < 0.6) # rule of thumb, bitches
    idx = np.ones_like(sampler.acceptance_fraction).astype(bool)

    logger.info("{0} walkers ({1:.1f}%) converged"
                .format(sum(idx), sum(idx)/sp["walkers"]*100))
    
    # Pluck out good chains, make a new flatchain from those...
    good_flatchain = []
    good_chains = sampler.chain[idx]
    for chain in good_chains:
        good_flatchain += list(chain)
    good_flatchain = np.array(good_flatchain)
    
    if sp["make_plots"]:
        fig = plot_sampler_pickle(os.path.join(path,data_file), params=sp["model_parameters"])
        fig.savefig(os.path.join(path, "emcee_sampler.png"), format="png")
    
    # Get "best" (mean) potential parameters:
    return dict(zip(sp["model_parameters"],np.mean(good_flatchain,axis=0)))