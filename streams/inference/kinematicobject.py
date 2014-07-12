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

__all__ = ['KinematicObject']

# TODO: or could do something like this?
# class ObservedQuantity(ModelParameter):
#     def __init__(self, value, uncertainty, parameter=None):

class ObservedQuantity(object):

    def __init__(self, value, error, truth=None):
        self.value = value
        self.error = error
        self.truth = truth

class KinematicObject(object):

    def __init__(self, hel, errors, truths=dict()):
        """ """

        pars = OrderedDict()
        qs = OrderedDict()

        # TODO: still need to be able to define mappings for parameters...
        for name in heliocentric_names:
            pars[name] = ModelParameter(name, truth=truths.get(name,None))
            qs[name] = ObservedQuantity(hel[name], errors[name],
                                        truths.get(name,None))
        self.parameters = pars
        self.quantities = qs

        self.n = len(self.quantities[name].value)

    def __len__(self):
        return self.n

    @property
    def values(self):
        return np.vstack([self.quantities[n].value for n in heliocentric_names]).T.copy()

    @property
    def errors(self):
        return np.vstack([self.quantities[n].error for n in heliocentric_names]).T.copy()

    @property
    def truths(self):
        return np.vstack([self.quantities[n].truth for n in heliocentric_names]).T.copy()