# coding: utf-8

""" Core functionality for doing stream inference """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys

# Third-party
import numpy as np
import astropy.units as u
from astropy.constants import G

__all__ = []

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

    def run(self):
        pass

""" API:

    lm10_model = StatisticalModel(['q1', 'v_halo'], 
                                  lm10_likelihood, 
                                  likelihood_args=(...))
    
    lm10_model.run(p0, steps, burn_in, pool)

"""