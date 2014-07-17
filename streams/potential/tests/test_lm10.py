# coding: utf-8

""" Test the LM10Potential. """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys

# Third-party
import numpy as np
from astropy import log as logger
import astropy.units as u

# Project
from ..lm10 import LM10Potential

def test_simple():
    potential = LM10Potential(q1=1.41)
    assert potential.parameter_values['q1'] == 1.41

    # test evaluation
    xyz = np.random.uniform(15., 60, size=(4,6))
    w = np.hstack((xyz,np.random.uniform(15., 60, size=(4,6))))
    r = np.linalg.norm(xyz, axis=-1)

    # TODO: check these against David Law's code or something?
    pot = potential.evaluate(xyz)

    # can do some minimal consistency check here
    acc = potential.acceleration(xyz)
    var_acc = potential.var_acceleration(w)
    assert np.allclose(acc, var_acc[:,3:6])

    r_tide = potential.tidal_radius(2.5E8, r)
