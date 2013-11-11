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

__all__ = ["ModelParameter"]

logger = logging.getLogger(__name__)

class ModelParameter(object):

    def __init__(self, targets, attr, ln_prior=None):
        if isinstance(targets, collections.Iterable):
            self.targets = targets
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

class PotentialParameter(object):

    def __init__(self, value=None, truth=None, range=(),
                 latex="", units=usys):

        if value is None and truth is None:
            raise ValueError("If value not specified, must specify truth.")

        elif value is None:
            value = truth

        if hasattr(value, "unit"):
            q = value.decompose(units)
            self._value = q.value
            self._unit = q.unit

            t = truth.decompose(units)
            self._truth = t.value

            lo,hi = range
            self._range = (lo.decompose(units).value,
                           hi.decompose(units).value)

        else:
            self._value = value
            self._truth = truth
            self._unit = u.dimensionless_unscaled
            self._range = range

        self.latex = latex

    @property
    def value(self):
        return self._value*self._unit

    @value.setter
    def value(self, v):
        self._value = v.to(self._unit).value

    @property
    def truth(self):
        return self._truth*self._unit

    @truth.setter
    def truth(self, v):
        self._truth = v.to(self._unit).value

    @property
    def range(self):
        return (self._range[0]*self._unit, self._range[1]*self._unit)

    def __float__(self):
        return self._value

class ParticleParameter(object):

    def __init__(self, value=None, N=1, truth=None, range=(),
                 latex="", units=usys):

        if value is None and truth is None:
            raise ValueError("If value not specified, must specify truth.")

        elif value is None:
            value = truth

        if hasattr(value, "unit"):
            q = value.decompose(units)
            self._value = q.value
            self._unit = q.unit

            t = truth.decompose(units)
            self._truth = t.value

            lo,hi = range
            self._range = (lo.decompose(units).value,
                           hi.decompose(units).value)

        else:
            self._value = value
            self._truth = truth
            self._unit = u.dimensionless_unscaled
            self._range = range

        self.latex = latex

    @property
    def value(self):
        return self._value*self._unit

    @value.setter
    def value(self, v):
        self._value = v.to(self._unit).value

    @property
    def truth(self):
        return self._truth*self._unit

    @truth.setter
    def truth(self, v):
        self._truth = v.to(self._unit).value

    @property
    def range(self):
        return (self._range[0]*self._unit, self._range[1]*self._unit)

    def __float__(self):
        return self._value
