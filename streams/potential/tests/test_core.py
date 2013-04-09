# coding: utf-8
"""
    Test the core Potential classes
"""

from __future__ import absolute_import, unicode_literals, division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

import os
import pytest
import numpy as np
from astropy.constants.si import G
import astropy.units as u
import matplotlib.pyplot as plt

from ..core import CartesianPotential

#np.testing.assert_array_almost_equal(new_quantity.value, 130.4164, decimal=5)

def test_basic():
    p = CartesianPotential(unit_bases=[u.kpc, u.kg, u.s], origin=[0.,0.,0.]*u.kpc)

def test_bases():
    
    # Not enough unit bases supplied
    with pytest.raises(ValueError):
        CartesianPotential(unit_bases=[u.kpc, u.s], origin=[0.,0.,0.]*u.kpc)
    
    with pytest.raises(ValueError):
        CartesianPotential(unit_bases=[u.kg, u.s], origin=[0.,0.,0.]*u.kpc)
    
    with pytest.raises(ValueError):
        CartesianPotential(unit_bases=[u.kpc, u.kg], origin=[0.,0.,0.]*u.kpc)

def test_origin():
    
    # not a quantity
    with pytest.raises(TypeError):
        CartesianPotential(unit_bases=[u.kpc, u.kg, u.s], origin=[0.,0.,0.])
    
    # quantity with wrong units
    with pytest.raises(ValueError):
        CartesianPotential(unit_bases=[u.kpc, u.kg, u.s], origin=[0.,0.,0.]*u.s)