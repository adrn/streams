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

__all__ = ['Stars', 'Progenitor']

class StreamComponentBase(object):

    def __init__(self, data, err=None, **kwargs):
        """ """

        self.data = data
        self.err = err
        self.parameters = kwargs

    # def ln_prior(self, data):
    #     """ """

    #     data = np.atleast_2d(data)
    #     l,b,d,mul,mub,vr = data.T

    #     ln_p_l = -np.log(2*np.pi)  # isotropic
    #     ln_p_b = np.log(np.cos(b) / 2.)  # isotropic

    #     # distance range 1-200 kpc
    #     ln_p_d = np.log((1 / np.log(200./1.))) - np.log(d)
    #     ln_p_mul = log_normal(mul, 0., 0.306814 / d)  # 300 km/s at d
    #     ln_p_mub = log_normal(mub, 0., 0.306814 / d)  # 300 km/s at d
    #     ln_p_vr = log_normal(vr, 0., 0.306814)  # 300 km/s

    #     return ln_p_l, ln_p_b, ln_p_d, ln_p_mul, ln_p_mub, ln_p_vr

class Stars(StreamComponentBase):

    def __init__(self, data, err=None, tail=None):
        """ TODO: """

        if isinstance(tail, ModelParameter):

        elif tail is None:

        else:



class Progenitor(StreamComponentBase):

    def __init__(self, data, err=None, m0=None, mdot=0.):
        pass
