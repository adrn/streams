# coding: utf-8
"""
    Test UnitSystem
"""

from __future__ import absolute_import, unicode_literals, division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys

# Third-party
import astropy.units as u
import numpy as np
import pytest

from ..units import UnitSystem

def test_create():
    usys = UnitSystem(u.km, u.s, u.M_sun)

def test_dict():
    usys = UnitSystem(u.km, u.s, u.M_sun)
    print("length", usys['length'])
    
    print("keys", usys.keys())
    print("values", usys.values())
    print("bases", usys.bases)

def test_to_list():
    usys = UnitSystem(u.km, u.s, u.M_sun)
    print("to list", list(usys))