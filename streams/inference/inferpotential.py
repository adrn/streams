# coding: utf-8

""" """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys
from multiprocessing import Pool
import logging

# Third-party
import numpy as np
np.seterr(all="ignore")
import scipy
scipy.seterr(all="ignore")
import astropy.units as u
import emcee

# Project
from streams.potential.lm10 import param_ranges
from streams.inference import ln_posterior, ln_posterior_lm10

__all__ = ["infer_potential", "max_likelihood_parameters"]

logger = logging.getLogger(__name__)

def infer_potential(particles, satellite_ic, t, model_parameters, 
                    walkers=None, steps=100, burn_in=None, pool=None):
    
    """ Given a set of particles and the orbit of the progenitor system, 
        infer the model halo parameters by using MCMC to optimize the 
        generalized variance of the reduced, relative phase-space 
        coordinate distribution.
        
        Parameters
        ----------
        particles : TestParticle
            A set of particles -- positions and velocities -- to integrate
            backwards with the satellite orbit.
        satellite_ic : TestParticle
            The initial conditions of the progenitor satellite orbit.
        t : numpy.ndarray
            Array of times to integrate the particles / satellite over.
        model_parameters : list, tuple
            List of the names of the model parameters to infer.
        walkers : int
            Number of walkers to use in emcee.
        steps : int (optional)
            Number of steps for each walker to take through parameter space.
        burn_in : int (optional)
            Defaults to 1/10 the number of steps. 
        pool : multiprocessing.Pool, emcee.MPIPool
            A multiprocessing or MPI pool to pass to emcee for wicked awesome
            parallelization!
            
    """
    
    # If the number of walkers is not specified, default to twice the number 
    #   of model parameters
    if walkers == None:
        walkers = len(model_parameters) * 2
    
    if burn_in == None:
        burn_in = steps // 10
    
    # Create the starting points for all walkers
    for p_name in model_parameters:
        # sample initial parameter values from uniform distributions over 
        #   the ranges specified in lm10.py
        this_p = np.random.uniform(param_ranges[p_name][0], 
                                   param_ranges[p_name][1],
                                   size=walkers)
        try:
            p0 = np.vstack((p0, this_p))
        except NameError:
            p0 = this_p
    p0 = p0.T
    
    # Construct the log posterior probability function to pass in to emcee
    args = particles, satellite_ic, t
   
    # If no pool is specified, just create a single-processor pool
    if pool == None:
        pool = None
    
    # Construct an ensemble sampler to walk through dat model parameter space
    # 2013-05-01: changed ln_posterior to ln_posterior_lm10
    sampler = emcee.EnsembleSampler(nwalkers=walkers, 
                                    dim=len(model_parameters), 
                                    lnpostfn=ln_posterior_lm10, 
                                    pool=pool, 
                                    args=args)
    
    logger.debug("About to start simulation...")
    
    # If a burn-in period requested, run the sampler for 'burn_in' steps then
    #   reset the walkers and use the end positions as new initial conditions
    if burn_in > 0:
        pos, prob, state = sampler.run_mcmc(p0, burn_in)
        sampler.reset()
    else:
        pos = p0
    
    # Run the MCMC sampler and draw 'steps' samplers per walker
    pos, prob, state = sampler.run_mcmc(pos, steps)
    
    return sampler
    
def max_likelihood_parameters(sampler):
    """ Given an emcee Sampler object, find the maximum likelihood parameters
    
        Parameters
        ----------
        sampler : EnsembleSampler
    """
    
    nwalkers, nsteps, nparams = sampler.chain.shape
    
    # only use samplers that have reasonable acceptance fractions
    good_walkers = (sampler.acceptance_fraction > 0.15) & \
                   (sampler.acceptance_fraction < 0.6)
    
    logger.info("{0} walkers ({1:.1f}%) converged"
                .format(sum(good_walkers), sum(good_walkers)/nwalkers*100))
    
    good_chain = sampler.chain[good_walkers] # (sum(good_walkers), nsteps, nparams)
    good_probs = sampler.lnprobability[good_walkers] # (sum(good_walkers), nsteps)
    
    best_step_idx = np.argmax(np.ravel(good_probs))
    
    flatchain = []
    for walker_i in range(len(good_chain)):
        flatchain += list(good_chain[walker_i])
    flatchain = np.array(flatchain)

    best_params = flatchain[best_step_idx]
    
    return best_params
