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

        return self

    def copy(self):
        """ Return a copy of this `ModelParameter` instance """
        return ModelParameter(name=self.name, value=self.value*self.unit,
                              prior=self._prior, truth=self.truth)

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

    def __reduce__(self):
        # patch to pickle ModelParameter objects (ndarray subclasses),
        # see http://www.mail-archive.com/numpy-discussion@scipy.org/msg02446.html

        object_state = list(super(ModelParameter, self).__reduce__())
        object_state[2] = (object_state[2], self.__dict__)
        return tuple(object_state)

    def __setstate__(self, state):
        # patch to unpickle ModelParameter objects (ndarray subclasses),
        # see http://www.mail-archive.com/numpy-discussion@scipy.org/msg02446.html

        nd_state, own_state = state
        super(ModelParameter, self).__setstate__(nd_state)
        self.__dict__.update(own_state)