# coding: utf-8

""" Parameter classes """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys
import logging
import collections

# Third-party
import numpy as np
import astropy.units as u

__all__ = ["ModelParameter", "Parameter"]

logger = logging.getLogger(__name__)

class ModelParameter(object):

    def __init__(self, targets, attr, ln_prior=None):
        """ This object represents an abstract concept -- the idea of
            a parameter unbound from actual instances of objects. For
            example, a ModelParamter could be a mass, which maps to
            several different target objects that all have associated
            masses.

            Parameters
            ----------
            targets : iterable
            attr : str
            ln_prior : LogPrior
        """

        if isinstance(targets, collections.Iterable):
            self.targets = list(targets)
        else:
            self.targets = [targets]

        self.attr = attr

        if ln_prior is None:
            ln_prior = LogPrior()
        self._ln_prior = ln_prior

    def __str__(self):
        return "{0}".format(self.attr)

    def __unicode__(self):
        return self.__str__()

    def __repr__(self):
        return "<{0} {1}>".format(self.__class__.__name__,
                                  str(self))

    def get(self):
        #return getattr(self.target, self.attr)
        return np.array([getattr(t, self.attr) for t in self.targets])

    def set(self, value):
        #setattr(self.target, self.attr, value)
        [setattr(t, self.attr, value) for t in self.targets]

    def ln_prior(self):
        return np.array([self._ln_prior(v) for v in self.get()])

    def __len__(self):
        return self.get().size

class Parameter(object):
    pass