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

    def __init__(self, data, errors, truths=dict()):
        """ """

        ps = OrderedDict()

        # TODO: still need to be able to define mappings for parameters...
        for name in heliocentric_names:
            ps[name] = ObservedQuantity(name, data[name], errors[name],
                                        truth=truths.get(name,None))

        self.parameters = ps

        self.data = np.vstack([ps[n].value for n in heliocentric_names]).T.copy()
        self.errors = np.vstack([ps[n].error for n in heliocentric_names]).T.copy()
        self.truths = np.vstack([ps[n].truth for n in heliocentric_names]).T.copy()
        self.n = len(self.data)

    def __len__(self):
        return self.n

    def ln_prior(self, coords):
        """ TODO: """
        coords = np.atleast_2d(coords)
        l,b,d,mul,mub,vr = coords.T

        ln_p_lb = np.log(np.cos(b) / (4*np.pi))
        # HACK: distance range 1-200 kpc
        ln_p_d = np.log((1 / np.log(200./1.))) - np.log(d)
        ln_p_mul = log_normal(mul, 0., 0.306814 / d) # 300 km/s
        ln_p_mub = log_normal(mub, 0., 0.306814 / d) # 300 km/s
        ln_p_vr = log_normal(vr, 0., 0.306814) # 300 km/s

        return ln_p_lb + ln_p_d + ln_p_mul + ln_p_mub + ln_p_vr
