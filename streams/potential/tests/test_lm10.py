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

from ..lm10 import LawMajewski2010

def test_simple():
    p = LawMajewski2010()
    p = LawMajewski2010(v_halo=121*u.km/u.s)

def test_tidal_radius():
    p = LawMajewski2010()
    
    r = np.array([[10., 0., 2.]]) * u.kpc
    m = 2.5E8 * u.M_sun
    r_tide = p.tidal_radius(m, r)
    _r_tide = p._tidal_radius(m.value, r.value)
    
    assert _r_tide == r_tide.value

def test_escape_velocity():
    p = LawMajewski2010()
    
    r = np.array([[10., 0., 2.]]) * u.kpc
    m = 2.5E8 * u.M_sun
    
    v_esc = p.escape_velocity(m, r=r)
    r_tide = p.tidal_radius(m, r)
    v_esc = p.escape_velocity(m, r_tide=r_tide)
    
    _r_tide = p._tidal_radius(m.value, r.value)
    _v_esc = p._escape_velocity(m.value, r_tide=_r_tide)
    
    assert _v_esc == v_esc.value