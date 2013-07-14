# coding: utf-8
""" """

from __future__ import absolute_import, unicode_literals, division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys

# Third-party
import astropy.units as u
import numpy as np
import pytest

from ...misc.units import UnitSystem
from ..particles import *
from ..orbits import *

usys = UnitSystem(u.kpc, u.Myr, u.M_sun)

def test_orbitcollection_init():
    t = np.linspace(0., 10., 100)*u.yr
    r = np.random.random((100,10,3))*u.kpc
    v = np.random.random((100,10,3))*u.kpc/u.yr
    m = np.random.random(10)*u.M_sun
    
    oc = OrbitCollection(t=t, r=r, v=v, m=m)
    
    # Should fail -- different length units, no usys:
    t = np.linspace(0., 10., 100)*u.yr
    r = np.random.random((100,10,3))*u.kpc
    v = np.random.random((100,10,3))*u.km/u.s
    m = np.random.random(10)*u.M_sun
    with pytest.raises(ValueError):
        oc = OrbitCollection(t=t, r=r, v=v, m=m)
    
    # should pass bc we give a usys
    oc = OrbitCollection(t=t, r=r, v=v, m=m, unit_system=usys)

def test_to():
    t = np.arange(0., 100, 0.1)*u.Myr
    r = np.random.random(size=(len(t),10,3))*u.kpc
    v = np.random.random(size=(len(t),10,3))*u.kpc/u.Myr
    m = np.random.random(10)*u.M_sun
    
    pc = OrbitCollection(t=t, r=r, v=v, m=m, unit_system=usys)
    
    usys2 = UnitSystem(u.km, u.s, u.kg)    
    pc2 = pc.to(usys2)
    
    assert np.allclose(pc2._t, t.to(u.s).value)
    assert np.allclose(pc2._r, r.to(u.km).value)
    assert np.allclose(pc2._v, v.to(u.km/u.s).value)
    assert np.allclose(pc2._m, m.to(u.kg).value)