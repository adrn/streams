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

__all__ = ["infer_potential", "max_likelihood_parameters"]

logger = logging.getLogger(__name__)

def infer_potential(ln_posterior, p0, steps=100, 
                    burn_in=None, pool=None, args=()):
    
    """ Given a set of particles and the orbit of the progenitor system, 
        infer the model halo parameters by using MCMC to optimize the 
        generalized variance of the reduced, relative phase-space 
        coordinate distribution.
        
        Parameters
        ----------
        ln_posterior : func
            Log-posterior function.
        p0 : array
            2D array of starting positions for all walkers.
        steps : int (optional)
            Number of steps for each walker to take through parameter space.
        burn_in : int (optional)
            Defaults to 1/10 the number of steps. 
        pool : multiprocessing.Pool, emcee.MPIPool
            A multiprocessing or MPI pool to pass to emcee for wicked awesome
            parallelization!
        args : (optional)
            Positional arguments to be passed to the posterior function.
    """
    
    if burn_in == None:
        burn_in = steps // 10
    
    assert p0.ndim == 2
    walkers = len(p0)
   
    # If no pool is specified, just create a single-processor pool
    if pool == None:
        pool = None
    
    # Construct an ensemble sampler to walk through dat model parameter space
    sampler = emcee.EnsembleSampler(nwalkers=walkers, 
                                    dim=p0.shape[1], 
                                    lnpostfn=ln_posterior, 
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
    good_walkers = (sampler.acceptance_fraction > 0.1) & \
                   (sampler.acceptance_fraction < 0.8)
    
    logger.info("{0} walkers ({1:.1f}%) converged"
                .format(sum(good_walkers), sum(good_walkers)/nwalkers*100))
    
    good_chain = sampler.chain[good_walkers] # (sum(good_walkers), nsteps, nparams)
    good_probs = sampler.lnprobability[good_walkers] # (sum(good_walkers), nsteps)
    flatchain = np.vstack(good_chain)
    
    # Find median values in each parameter
    best_params = []
    for ii in range(nparams):
        #xsorted = sorted(flatchain[:,ii])
        #best_params.append(xsorted[int(0.5 * len(xsorted))])
        best_params.append(np.median(flatchain[:,ii]))
    
    """
    best_step_idx = np.argmax(np.ravel(good_probs))
    
    flatchain = []
    for walker_i in range(len(good_chain)):
        flatchain += list(good_chain[walker_i])
    flatchain = np.array(flatchain)

    best_params = flatchain[best_step_idx]
    """
    
    return np.array(best_params)