# coding: utf-8
"""
    Test error models
"""

from __future__ import absolute_import, unicode_literals, division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys

# Third-party
import astropy.units as u
import numpy as np
import pytest

from ..error_model import *

def test_error_model():
    pass

def test_spitzer_gaia():
    test_data = np.random.random((100,6))

    em = SpitzerGaiaErrorModel()
    err = em(test_data)
    assert err.shape == test_data.shape

    em = SpitzerGaiaErrorModel(D_err=0.05)
    err = em(test_data)
    assert err.shape == test_data.shape

    em = SpitzerGaiaErrorModel(mul_err=1.*u.mas/u.yr)
    err = em(test_data)
    assert err.shape == test_data.shape