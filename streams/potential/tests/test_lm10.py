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

from ..lm10 import LawMajewski2010, CLawMajewski2010

def test_simple():
    p = LawMajewski2010()
    p = LawMajewski2010(v_halo=121*u.km/u.s)

def test_cython():
    p = CLawMajewski2010()
    p = CLawMajewski2010(v_halo=121*u.km/u.s)
    
def test_compare():
    py = LawMajewski2010(v_halo=121*u.km/u.s)
    cy = CLawMajewski2010(v_halo=121*u.km/u.s)
    
    
    r = np.array([[10.,10.,0.],
                  [20.,20.,0.]])*u.kpc
    
    print()
    print(py.acceleration_at(r))
    print(cy.acceleration_at(r))