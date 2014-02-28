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

__all__ = ["LogPrior", "LogUniformPrior", "LogExponentialPrior", "LogNormalPrior"]

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
        self.a = np.array(a)
        self.b = np.array(b)

    def sample(self, size=None):
        return np.random.uniform(self.a, self.b, size=size)

class LogExponentialPrior(LogPrior):

    def __call__(self, value):
        return self.lna - self.a*value

    def __init__(self, a):
        """ Logarithm of exponential distribution with scale or
            rate parameter 'a'.
        """
        self.a = np.array(a)
        self.lna = np.log(a)

    def sample(self, size=None):
        x = np.random.uniform(0., 5./self.a, size=size)
        return x

class LogNormalCovPrior(LogPrior):

    def __call__(self, value):
        X = np.atleast_2d(self.mu - value)
        q = np.array([np.dot(d,np.dot(icov,d.T)) \
                      for icov,d in zip(self._icov, X)])

        return np.squeeze(self._norm - 0.5 * q)

    def __init__(self, mu, sigma=None, cov=None):
        self.mu = np.atleast_2d(mu)

        if sigma is not None:
            sigma = np.array(sigma)
            if sigma.shape[-1] != mu.shape[-1]:
                raise ValueError("Shape of std dev vector (sigma) must match"
                                 " shape of mean vector (mu) along axis=-1")
            cov = np.diag(sigma**2)

        if cov is None:
            raise ValueError("Must specify vector of sigmas or "
                             "covariance matrix.")

        if 2 > cov.ndim > 3:
            raise ValueError("covariance matrix must be 2D or 3D")

        self.cov = cov
        if self.cov.ndim == 2:
            self.cov = self.cov[np.newaxis]

        k = self.cov.shape[-1]

        self._icov = np.array([np.linalg.inv(c) for c in self.cov])
        self._norm = -0.5*k*np.log(2*np.pi)
        self._norm -=0.5*np.array([np.linalg.slogdet(c)[1] for c in self.cov])

    def sample(self, size=None):
        s = np.array([np.random.multivariate_normal(mu, cov, size=size) \
                      for mu,cov in zip(self.mu, self.cov)])
        if size is not None:
            return np.squeeze(np.rollaxis(s, 1))
        else:
            return np.squeeze(s)

class LogNormalPrior(LogPrior):

    def __call__(self, value):
        X = self.mu - np.atleast_1d(value)
        return self._norm - 0.5 * np.sum((X / self.sigma)**2)

    def __init__(self, mu, sigma):
        self.mu = np.atleast_1d(mu)
        self.sigma = np.atleast_1d(sigma)
        k = len(self.mu)
        self._norm = -0.5*k*np.log(2*np.pi) - np.sum(np.log(self.sigma))

    def sample(self, size=None):
        s = np.random.multivariate_normal(self.mu, np.diag(self.sigma**2), size=size)
        if size is not None:
            return np.squeeze(np.rollaxis(s, 1))
        else:
            return np.squeeze(s)