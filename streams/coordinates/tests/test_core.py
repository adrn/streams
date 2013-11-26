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
        l = coord.Angle(row["lon"] * u.degree)
        b = coord.Angle(row["lat"] * u.degree)
        v_gsr = row["vgsr"] * u.km/u.s
        v_sun_lsr = [row["vx"],row["vy"],row["vz"]]*u.km/u.s

        v_hel = vgsr_to_vhel(l, b, v_gsr,
                             v_sun_lsr=v_sun_lsr,
                             v_sun_circ=row["vcirc"]*u.km/u.s)

        np.testing.assert_almost_equal(v_hel.value, row['vhelio'], decimal=4)

def test_hel_to_gsr():
    for row in data:
        l = coord.Angle(row["lon"] * u.degree)
        b = coord.Angle(row["lat"] * u.degree)
        v_hel = row["vhelio"] * u.km/u.s
        v_sun_lsr = [row["vx"],row["vy"],row["vz"]]*u.km/u.s

        v_gsr = vhel_to_vgsr(l, b, v_hel,
                             v_sun_lsr=v_sun_lsr,
                             v_sun_circ=row["vcirc"]*u.km/u.s)

        np.testing.assert_almost_equal(v_gsr.value, row['vgsr'], decimal=4)

def test_roundtrip():
    np.random.seed(43)
    N = 100
    x1 = np.random.uniform(-50.,50.,size=N)*u.kpc
    y1 = np.random.uniform(-50.,50.,size=N)*u.kpc
    z1 = np.random.uniform(-50.,50.,size=N)*u.kpc

    vx1 = np.random.uniform(-100.,100.,size=N)*u.km/u.s
    vy1 = np.random.uniform(-100.,100.,size=N)*u.km/u.s
    vz1 = np.random.uniform(-100.,100.,size=N)*u.km/u.s

    l,b,d,mul,mub,vr = gc_to_hel(x1,y1,z1,vx1,vy1,vz1)
    x2,y2,z2,vx2,vy2,vz2 = hel_to_gc(l,b,d,mul,mub,vr)

    np.all(np.round((x1-x2)/x1*100.,2) == 0)
    np.all(np.round((y1-y2)/y1*100.,2) == 0)
    np.all(np.round((z1-z2)/z1*100.,2) == 0)
    np.all(np.round((vx1-vx2)/vx1*100.,2) == 0)
    np.all(np.round((vy1-vy2)/vy1*100.,2) == 0)
    np.all(np.round((vz1-vz2)/vz1*100.,2) == 0)

def test_roundtrip_unitless():
    np.random.seed(43)
    N = 100
    X = np.random.random((N,6))

    O = _gc_to_hel(X)
    X2 = _hel_to_gc(O)

    assert np.allclose(X, X2)