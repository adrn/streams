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
        q = np.diag(np.dot(d,np.dot(self._icov,d.T)))
        return self._norm - 0.5 * q

    def __init__(self, mu, sigma=None, cov=None):
        self.mu = mu

        if sigma is not None:
            if sigma.shape[0] != mu.shape[-1]:
                raise ValueError("Shape of std dev vector (sigma) must match shape of mean vector (mu) along axis=-1")

            cov = np.diag(sigma**2)

        if cov is None:
            raise ValueError("Must specify vector of sigmas or covariance matrix.")

        self.cov = cov
        self._icov = np.linalg.inv(self.cov)

        k,xx = self.cov.shape
        self._norm = -0.5*k*np.log(2*np.pi) - 0.5*np.log(np.linalg.det(self.cov))

    def sample(self, size=None):
        return np.random.multivariate_normal(self.mu, self.cov, size=size)