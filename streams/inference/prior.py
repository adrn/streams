# coding: utf-8

""" Priors """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys
import logging

# Third-party
import numpy as np
import astropy.units as u

__all__ = ["LogPrior", "LogUniformPrior", "LogNormalPrior"]

logger = logging.getLogger(__name__)

class LogPrior(object):

    def __call__(self, value):
        return 0.

class LogUniformPrior(LogPrior):

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

    def sample(self, size=1):
        return np.random.uniform(self.a, self.b, size=size)

class LogNormalPrior(LogPrior):

    def __call__(self, value):
        d = self.mu - value
        return -0.5 * (np.dot(d, d / self.sigma**2) + self._norm)

    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma

        if len(self.mu):
            pass
            # TODO: check shapes of mu and sigma

        k = len(self.sigma)
        self._norm = k*np.log(2*np.pi) + 2*np.sum(np.log(self.sigma))

    def sample(self, size=None):
        return np.random.normal(self.mu, self.sigma, size=size)