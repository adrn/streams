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

from .prior import LogPrior

__all__ = ["ModelParameter", "Parameter"]

logger = logging.getLogger(__name__)

class ModelParameter(object):

    def __init__(self, target, attr, ln_prior=None):
        """ This object represents an abstract concept -- the idea of
            a parameter unbound from actual instances of objects. For
            example, a ModelParamter could be a mass, which maps to
            several different target objects that all have associated
            masses.

            Parameters
            ----------
            target : iterable
            attr : str
            ln_prior : LogPrior
        """

        self.target = target
        self.attr = attr

        if ln_prior is None:
            ln_prior = LogPrior()
        self._ln_prior = ln_prior

        self.shape = self.get().shape

    def __str__(self):
        return "{0}".format(self.attr)

    def __unicode__(self):
        return self.__str__()

    def __repr__(self):
        return "<{0} {1}>".format(self.__class__.__name__,
                                  str(self))

    def get(self):
        return np.array(getattr(self.target, self.attr))

    def set(self, value):
        setattr(self.target, self.attr, value.reshape(self.shape))

    def ln_prior(self):
        return self._ln_prior(self.get())

    def sample(self, size=None):
        return self._ln_prior.sample(size)

    def __len__(self):
        return self.get().size

class Parameter(object):
    pass