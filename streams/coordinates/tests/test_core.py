# coding: utf-8
"""
    Test conversions in core.py
"""

from __future__ import absolute_import, division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

import os
import pytest
import numpy as np

import astropy.coordinates as coord
import astropy.units as u
from astropy.io import ascii

from ..core import *

this_path = os.path.split(__file__)[0]
data = ascii.read(os.path.join(this_path, "idl_vgsr_vhel.txt"))

def test_gsr_to_hel():
    for row in data:
        l = row["lon"] * u.degree
        b = row["lat"] * u.degree
        v_gsr = row["vgsr"] * u.km/u.s
        v_sun_lsr = [row["vx"],row["vy"],row["vz"]]*u.km/u.s
        
        v_hel = vgsr_to_vhel(l, b, v_gsr, 
                             v_sun_lsr=v_sun_lsr, 
                             v_circ=row["vcirc"]*u.km/u.s)
        
        np.testing.assert_almost_equal(v_hel.value, row['vhelio'], decimal=4)

def test_hel_to_gsr():
    for row in data:
        l = row["lon"] * u.degree
        b = row["lat"] * u.degree
        v_hel = row["vhelio"] * u.km/u.s
        v_sun_lsr = [row["vx"],row["vy"],row["vz"]]*u.km/u.s
        
        v_gsr = vhel_to_vgsr(l, b, v_hel, 
                             v_sun_lsr=v_sun_lsr, 
                             v_circ=row["vcirc"]*u.km/u.s)
        
        np.testing.assert_almost_equal(v_gsr.value, row['vgsr'], decimal=4)
        