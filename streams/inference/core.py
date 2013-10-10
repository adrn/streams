# coding: utf-8

""" Core functionality for doing stream inference """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys
import logging

# Third-party
import emcee
import numpy as np
import astropy.units as u

__all__ = ["StatisticalModel", "back_integrate_likelihood"]

logger = logging.getLogger(__name__)

class StatisticalModel(object):

    def __init__(self, parameters, ln_likelihood, likelihood_args=(),
                 parameter_bounds=dict(), prior_funcs=dict()):
        """ Right now this is tailored to my specific use case. If no specific
            prior function is specified, it creates a uniform prior around the
            specified bounds -- you have to specify either parameter bounds or
            prior functions.

            Parameters
            ----------
            parameters : list
                List of model parameters.
            ln_likelihood : func
                The likelihood function.
            likelihood_args : tuple
                Arguments to be passed in to the likelihood function.
            parameter_bounds : dict 
                Dictionary of tuples specifying min/max bounds for 
                each parameter.
            prior_funcs : dict (optional)
                Specify a custom prior here if you don't want to use
                a uniform prior between parameter bounds.
        """

        self.parameters = parameters
        self._prior_funcs = prior_funcs

        for p in self.parameters:
        # TODO: validate prior funcs / param bounds

    def ln_prior(self, p):
        """ Evaluate the prior functions """

        _sum = 0.
        for ii,param in enumerate(param_names):
            if self._prior_funcs.has_key(param):
                _sum += self._prior_funcs[param](p[ii])
            else:
                lo,hi = self.parameter_bounds[param]
                if p[ii] < lo or p[ii] > hi:
                    return -np.inf

        return _sum

    def ln_posterior(self, p):
        return self.ln_prior(p)+self.ln_likelihood(p, **self.likelihood_args)

    def run(self, p0, nsteps, nburn=None, pool=None):
        """ Use emcee to sample from the posterior.
            
            Parameters
            ----------
            p0 : array
                2D array of starting positions for all walkers.
            nsteps : int (optional)
                Number of steps for each walker to take through 
                parameter space.
            burn_in : int (optional)
                Defaults to 1/10 the number of steps. 
            pool : multiprocessing.Pool, emcee.MPIPool
                A multiprocessing or MPI pool to pass to emcee for 
                wicked awesome parallelization!
        """
        if nburn == None:
            nburn = nsteps // 10

        nwalkers, ndim = p0.shape

        if ndim != len(self.parameters):
            raise ValueError("Parameter initial conditions must have shape"
                             "(nwalkers,ndim) ({0},{1})".format(nwalkers,
                                len(self.parameters)))

        # make the ensemble sampler
        sampler = emcee.EnsembleSampler(nwalkers=nwalkers, dim=ndim, 
                                        lnpostfn=self.ln_posterior, 
                                        pool=pool)

        logger.debug("About to start walkers...")

        # If a burn-in period requested, run the sampler for 'burn_in' 
        #   steps then reset the walkers and use the end positions as 
        #   new initial conditions
        if nburn > 0:
            pos, prob, state = sampler.run_mcmc(p0, nburn)
            sampler.reset()
        else:
            pos = p0

        # Run the MCMC sampler and draw nsteps samples per walker
        pos, prob, state = sampler.run_mcmc(pos, nsteps)

        return sampler

def back_integrate_likelihood(p, param_names, particles, satellite, 
                              Potential, t1, t2):
    """ Evaluate the TODO """

    model_params = dict(zip(param_names, p))
    potential = Potential(**model_params)

    integrator = SatelliteParticleIntegrator(potential, satellite, particles)
    
    # not adaptive:
    s_orbit,p_orbits = integrator.run(t1=t1, t2=t2, dt=-1.)
    
    return #objective2(lm10, s_orbit, p_orbits, v_disp=0.0133)