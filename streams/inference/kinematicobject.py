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

class KinematicObject(object):

    def __init__(self, hel, errors, truths=dict()):
        """ """

        pars = OrderedDict()

        # TODO: still need to be able to define mappings for parameters...
        for name in heliocentric_names:
            pars[name] = ModelParameter(name, truth=truths.get(name,None))
        self.parameters = pars

        self.values = OrderedDict()
        self.errors = OrderedDict()
        for par_name in self.parameters.keys():
            self.values[par_name] = hel[par_name]
            self.errors[par_name] = errors[par_name]
            self.n = len(hel[par_name])

    def __len__(self):
        return self.n