# coding: utf-8

""" Test inference utilities """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys

# Third-party
import numpy as np
from astropy import log as logger
import astropy.units as u
from scipy.stats import norm

# Project
from ..util import *

def test_log_normal():
    n = 128

    x = np.random.uniform(-100, 100, size=n)
    sigma = np.random.uniform(1., 5., size=n)
    mu = x + np.random.normal(0., sigma)

    U = log_normal(x, mu, sigma)
    V = np.log(norm.pdf(x, loc=mu, scale=sigma))

    assert np.allclose(U, V)