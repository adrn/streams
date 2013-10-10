# coding: utf-8
""" Test the special fucntions used to compute our scalar objective """

from __future__ import absolute_import, unicode_literals, division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys
import time as pytime

# Third-party
import numpy as np
import pytest
import astropy.units as u
import matplotlib.pyplot as plt

from ..core import *

plot_path = "plots/tests/inference"
if not os.path.exists(plot_path):
    os.makedirs(plot_path)


def test_statistical_model():
    """ Test with example from main Emcee docs """

    def lnprob(x, ivar=1.7):
        return -0.5 * np.sum(ivar * x ** 2)

    parameters = ['x']
    ndim, nwalkers = 1, 100
    ivar = 1. / np.random.rand(ndim)
    p0 = [np.random.rand(ndim) for i in range(nwalkers)]

    sm = StatisticalModel(parameters, 
                          ln_likelihood=lnprob,
                          likelihood_args=(1.7,), 
                          parameter_bounds={'x':(-10,100)})

    sampler = sm.run(p0, 100, 0)