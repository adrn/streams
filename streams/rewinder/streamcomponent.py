# coding: utf-8

""" Class for representing observed dynamical object """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
from collections import OrderedDict

# Third-party
import numpy as np
from scipy.stats import norm
from gary.units import galactic
from gary.inference import ModelParameter, NormalPrior

# Project
from .. import heliocentric_names

__all__ = ['StreamComponent', 'RewinderPotential']

class StreamComponent(object):

    def __init__(self, data, err=None, parameters=dict()):
        """ TODO """

        self.data = data
        self.err = err
        self.parameters = parameters

        nparticles = self.data.shape[0]

        default_priors = np.zeros_like(self.data)
        default_priors[:,0] = np.zeros(nparticles) - np.log(2*np.pi)
        default_priors[:,1] = np.log(np.cos(data[:,1])/2)

        # distance range: 1-200 kpc
        default_priors[:,2] = -np.log(np.log(200.)) - np.log(data[:,2])

        p = NormalPrior(0., 0.306814 / data[:,2])  # 300 km/s at d
        default_priors[:,3] = p.logpdf(data[:,3])
        default_priors[:,4] = p.logpdf(data[:,4])

        p = NormalPrior(0., 0.306814)  # 300 km/s
        default_priors[:,5] = p.logpdf(data[:,5])

        self.default_priors = default_priors

    def ln_data_prob(self, x):
        """ Compute the (log-)probability of phase-space positions (in heliocentric
            observed coordinates), given the data and uncertainties.

        """
        _dist = norm(self.data, self.err)
        lp = _dist.logpdf(x)
        for i in range(6):
            lp[np.isnan(lp[:,i]),i] = self.default_priors[np.isnan(lp[:,i]),i]

        return lp.sum(axis=1)

class RewinderPotential(object):

    def __init__(self, Potential, priors=dict(), fixed_pars=dict()):
        """ TODO """

        self.Potential = Potential
        self.fixed_pars = fixed_pars
        self.priors = priors

        self.parameters = OrderedDict()
        for name,prior in priors.items():
            self.parameters[name] = ModelParameter(name=name, prior=prior)

    def obj(self, **kwargs):
        """ Given kwargs for the potential parameters being varied, return
            a potential object with the given parameters plus any other
            fixed parameters.
        """
        pars = dict(kwargs.items() + self.fixed_pars.items())
        return self.Potential(units=galactic, **pars)

    def ln_prior(self, **kwargs):
        """ Evaluate the value of the log-prior over the potential parameters. """

        lp = 0.
        for k,prior in self.priors.items():
            lp += prior.logpdf(kwargs[k])

        return lp
