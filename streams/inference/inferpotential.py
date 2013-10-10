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

__all__ = ["max_likelihood_parameters"]

logger = logging.getLogger(__name__)

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
    
    if np.all(~good_walkers):
        return np.array([])
    
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