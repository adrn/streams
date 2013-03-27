# coding: utf-8
"""
    Test the helper classes for reading in project data.
"""

from __future__ import absolute_import, unicode_literals, division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys

# Third-party
import astropy.coordinates as coord
import astropy.units as u
import numpy as np
import pytest

from ..core import LM10, LINEAR, QUEST, SgrCen, SgrSnapshot
    
def test_sgrcen():
    sgr_cen = SgrCen()
    
    assert isinstance(sgr_cen.x, u.Quantity)
    assert isinstance(sgr_cen.y, u.Quantity)
    assert isinstance(sgr_cen.z, u.Quantity)
    assert isinstance(sgr_cen.xyz, u.Quantity)
    assert sgr_cen.xyz.value.shape == (3,len(sgr_cen.z))
    
    assert isinstance(sgr_cen.vx, u.Quantity)
    assert isinstance(sgr_cen.vy, u.Quantity)
    assert isinstance(sgr_cen.vz, u.Quantity)
    assert isinstance(sgr_cen.vxyz, u.Quantity)
    assert sgr_cen.vxyz.value.shape == (3,len(sgr_cen.vz))

def test_sgrsnap():
    sgr_snap = SgrSnapshot(num=100, 
                           expr="sqrt(x**2 + y**2 + z**2) < 20.")
    
    assert np.max(np.sqrt(sgr_snap.x**2+sgr_snap.y**2+sgr_snap.z**2)) < 20.
    assert len(sgr_snap) == 100
    
    sgr_snap = SgrSnapshot(num=100, 
                           expr="(sqrt(x**2 + y**2 + z**2) < 20.) & (tub.value > 0)")
    assert np.max(np.sqrt(sgr_snap.x**2+sgr_snap.y**2+sgr_snap.z**2)) < 20.
    assert np.all(sgr_snap.tub.value > 0)
    assert len(sgr_snap) == 100
    
def test_coordinates():
    lm10 = LM10()
    linear = LINEAR()
    quest = QUEST()
    
    assert isinstance(lm10.ra[0], coord.RA)
    assert isinstance(lm10.dec[0], coord.Dec)
    
    assert isinstance(linear.ra[0], coord.RA)
    assert isinstance(linear.dec[0], coord.Dec)
    
    assert isinstance(quest.ra[0], coord.RA)
    assert isinstance(quest.dec[0], coord.Dec)

def test_sgrcen_interpolate(): 
    sgr_cen = SgrCen()
    
    new_ts = np.linspace(0, 500, 100)*u.Myr
    sgr_cen.interpolate(new_ts)
    
    assert (sgr_cen.t == new_ts).all()
    