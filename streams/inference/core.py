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

__all__ = []

logger = logging.getLogger(__name__)

class LogUniformPrior(object):

    def __call__(self, value):
        if np.any((value < self.a) | (value > self.b)):
            return -np.inf
        return 0.0

    def __init__(self, a, b):
        """ Return 0 if value is outside of the range 
            defined by a < value < b.
        """
        self.a = a
        self.b = b

class Parameter(object):

    def __init__(self, ln_prior=None, range=(None, None)):
        self.ln_prior = ln_prior
        if self.ln_prior is None:
            self.ln_prior = LogUniformPrior(*range)

class StreamModel(object):

    def __init__(self, Potential, satellite, particles, 
                 particle_data=None, satellite_data=None):
        """ ...

            Parameters
            ----------
            ...
        """
        parameters = []
        for p in Potential.parameters + \
                 satellite.parameters + \
                 particles.parameters:
            if not p.fixed:
                parameters.append(p)

        self.parameters = parameters
    
    def __call__(self, p):
        self.vector = p
        return self.ln_posterior()

    def ln_prior(self):
        lp = self.planetary_system.lnprior()
        if not np.isfinite(lp):
            return -np.inf
        pp = [l() for l in self.lnpriors]
        if not np.all(np.isfinite(pp)):
            return -np.inf
        ppar = [p.lnprior() for p in self.parameters]
        if not np.all(np.isfinite(ppar)):
            return -np.inf
        return lp + np.sum(pp) + np.sum(ppar)

    @property
    def vector(self):
        return np.concatenate(map(np.atleast_1d,
                                  [p.get() for p in self.parameters]))

    @vector.setter
    def vector(self, values):
        ind = 0
        for p in self.parameters:
            if len(p):
                p.set(values[ind:ind+len(p)])
                ind += len(p)
            else:
                p.set(values[ind])
                ind += 1

    # ---

    def ln_prior(self, p):
        """ Evaluate the prior functions """

        _sum = 0.
        for ii,param in enumerate(self.parameters):
            if self._prior_funcs.has_key(param):
                _sum += self._prior_funcs[param](p[ii])
            else:
                lo,hi = self.parameter_bounds[param]
                if p[ii] < lo or p[ii] > hi:
                    return -np.inf

        return _sum

    def ln_posterior(self, p):
        return self.ln_prior(p)+self.ln_likelihood(p, *self.likelihood_args)

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

        p0 = np.array(p0)
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
