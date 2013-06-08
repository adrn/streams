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

def test_particlecollection_init():

    # Init with individual arrays of ndim=1
    r = np.random.random(3)*u.kpc
    v = np.random.random(3)*u.km/u.s
    m = np.random.random()*u.M_sun
    
    with pytest.raises(ValueError):
        pc = ParticleCollection(r=r, v=v, m=m, unit_system=usys)
    
    r = np.random.random(size=(10,3))*u.kpc
    v = np.random.random(size=(10,3))*u.km/u.s
    m = np.random.random(10)*u.M_sun
    
    pc = ParticleCollection(r=r, v=v, m=m, unit_system=usys)
    assert np.all(pc.r.value == r.value)
    assert np.all(pc.v.value == v.value)
    assert np.all(pc.m.value == m.value)
    
    assert np.all(pc._r == r.value)
    assert np.all(pc._v == v.value)
    assert np.all(pc._m == m.value)

def test_acceleration():
    r = np.array([[1.,0.],
                  [0, 1.],
                  [-1., 0.],
                  [0., -1.]])*u.kpc
    v = np.zeros_like(r.value)*u.km/u.s
    m = np.random.random()*u.M_sun
    
    pc = ParticleCollection(r=r, v=v, m=m, unit_system=usys)

    pc.acceleration_at(np.array([0.,0.])*u.kpc, m=1.*u.M_sun)
    
    a = pc.acceleration_at(np.array([[0.5,0.5], [0.0,0.0], [-0.5, -0.5]])*u.kpc,
                           m=[1.,1.,1.]*u.M_sun)

def test_merge():
    # test merging two particle collections
    
    r = np.random.random(size=(10,3))*u.kpc
    v = np.random.random(size=(10,3))*u.km/u.s
    m = np.random.random(10)*u.M_sun
    
    pc1 = ParticleCollection(r=r, v=v, m=m, unit_system=usys)
    
    r = np.random.random(size=(10,3))*u.kpc
    v = np.random.random(size=(10,3))*u.km/u.s
    m = np.random.random(10)*u.M_sun
    
    pc2 = ParticleCollection(r=r, v=v, m=m, unit_system=usys)
    
    pc_merged = pc1.merge(pc2)
    
    assert pc_merged._r.shape == (20,3)