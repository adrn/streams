# coding: utf-8

""" Class for representing observed dynamical object """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
from collections import OrderedDict

# Third-party
import numpy as np
from astropy import log as logger

# Project
from .. import heliocentric_names
from streamteam.inference import ModelParameter
from .util import log_normal

__all__ = ['ObservedParameter', 'Stars', 'Progenitor']

class ObservedParameter(ModelParameter):

    def __init__(self, name, value, error, truth=None, prior=None):
        """ Represents an observed quantity, e.g., distance. """
        self.value = value
        self.error = error
        super(ObservedParameter,self).__init__(name, truth, prior, value.shape)

    def ln_likelihood(self, value):
        """ """
        return log_normal(value, self.value, self.error)

    def __str__(self):
        return "<ObservedParameter '{}'>".format(self.name)

class Base(object):

    def __init__(self, data, errors, truths=None):
        """ """

        ps = OrderedDict()

        # TODO: still need to be able to define mappings for parameters...
        for name in heliocentric_names:
            try:
                truth = np.asarray(truths[name])
            except:
                truth = None
            ps[name] = ObservedParameter(name, np.asarray(data[name]),
                                         np.asarray(errors[name]), truth=truth)
        self.parameters = ps

        self.data = np.vstack([ps[n].value for n in heliocentric_names]).T.copy()
        self.errors = np.vstack([ps[n].error for n in heliocentric_names]).T.copy()
        if truth is None:
            self.truths = None
        else:
            self.truths = np.vstack([ps[n].truth for n in heliocentric_names]).T.copy()
        self.n = len(self.data)

    def __len__(self):
        return self.n

    def ln_prior(self, data):
        """ """

        data = np.atleast_2d(data)
        l,b,d,mul,mub,vr = data.T

        ln_p_l = -np.log(2*np.pi)  # isotropic
        ln_p_b = np.log(np.cos(b) / 2.)  # isotropic

        # distance range 1-200 kpc
        ln_p_d = np.log((1 / np.log(200./1.))) - np.log(d)
        ln_p_mul = log_normal(mul, 0., 0.306814 / d)  # 300 km/s at d
        ln_p_mub = log_normal(mub, 0., 0.306814 / d)  # 300 km/s at d
        ln_p_vr = log_normal(vr, 0., 0.306814)  # 300 km/s

        return ln_p_l, ln_p_b, ln_p_d, ln_p_mul, ln_p_mub, ln_p_vr

class Stars(Base):
    pass

class Progenitor(Base):
    pass