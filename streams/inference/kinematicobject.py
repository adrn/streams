# coding: utf-8

""" Class for representing observed dynamical object """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys
from collections import OrderedDict

# Third-party
import numpy as np
from astropy import log as logger
import astropy.units as u

# Project
from .. import heliocentric_names
from streamteam.inference import ModelParameter
from .util import log_normal

__all__ = ['ObservedQuantity', 'KinematicObject']

class ObservedQuantity(ModelParameter):

    def __init__(self, name, value, error, truth=None, prior=None):
        """ Represents an observed quantity, e.g., distance. """
        self.value = value
        self.error = error
        super(ObservedQuantity,self).__init__(name, truth, prior, value.shape)

    def ln_likelihood(self, value):
        """ """
        return log_normal(value, self.value, self.error)

    def __str__(self):
        return "<ObservedQuantity '{}'>".format(self.name)


class KinematicObject(object):

    def __init__(self, coords, errors, truths=dict()):
        """ """

        ps = OrderedDict()

        # TODO: still need to be able to define mappings for parameters...
        for name in heliocentric_names:
            ps[name] = ObservedQuantity(name, coords[name], errors[name],
                                        truth=truths.get(name,None))

        self.parameters = ps
        self.n = len(self.parameters[name].value)

    def __len__(self):
        return self.n

    @property
    def values(self):
        return np.vstack([self.parameters[n].value for n in heliocentric_names]).T.copy()

    @property
    def errors(self):
        return np.vstack([self.parameters[n].error for n in heliocentric_names]).T.copy()

    @property
    def truths(self):
        return np.vstack([self.parameters[n].truth for n in heliocentric_names]).T.copy()