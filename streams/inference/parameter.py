# coding: utf-8

""" Parameter classes """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys
import logging

# Third-party
import numpy as np
import astropy.units as u
from astropy.utils import isiterable

# Project
from .prior import LogPrior

__all__ = ["ModelParameter"]

logger = logging.getLogger(__name__)

class ModelParameter(u.Quantity):

    def __new__(cls, name, value=np.nan, prior=None, truth=None):
        """ """

        #super(ModelParameter, self).__init__(v)
        try:
            value = value.decompose(usys)
        except:
            pass

        self = super(ModelParameter, cls).__new__(cls, value)

        # make sure input prior is a Prior, or a list of Prior objects
        if prior is None:
            prior = LogPrior()

        # TODO: type check
        self._prior = prior
        self.truth = truth
        self.name = str(name)

        print(name, prior, value)

        return self

    def prior(self, value):
        """ """

        if isiterable(self._prior):
            return np.array([p(value) for p,value in zip(self._prior,value)])
        else:
            return self._prior(value)

    def sample(self, size=None):
        """ """

        if isiterable(self._prior):
            return np.array([p.sample(size=size) for p in self._prior]).T
        else:
            return self._prior.sample(size=size)