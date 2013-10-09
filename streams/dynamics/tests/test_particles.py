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

from ..particles import *

units = (u.kpc, u.Myr, u.M_sun)

def test_Particle_init():

    # Init with individual arrays of ndim=1
    r = np.random.random(3)*u.kpc
    v = np.random.random(3)*u.km/u.s
    m = np.random.random()*u.M_sun
    
    with pytest.raises(ValueError):
        pc = Particle(r=r, v=v, m=m, units=units)
    
    r = np.random.random(size=(10,3))*u.kpc
    v = np.random.random(size=(10,3))*u.kpc/u.Myr
    m = np.random.random(10)*u.M_sun
    
    pc = Particle(r=r, v=v, m=m, units=units)
    
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
    
    pc = Particle(r=r, v=v, m=m, units=units)

    pc.acceleration_at(np.array([0.,0.])*u.kpc)
    
    a = pc.acceleration_at(np.array([[0.5,0.5], [0.0,0.0], [-0.5, -0.5]])*u.kpc)

def test_merge():
    # test merging two particle collections
    
    r = np.random.random(size=(10,3))*u.kpc
    v = np.random.random(size=(10,3))*u.km/u.s
    m = np.random.random(10)*u.M_sun
    
    pc1 = Particle(r=r, v=v, m=m, units=units)
    
    r = np.random.random(size=(10,3))*u.kpc
    v = np.random.random(size=(10,3))*u.km/u.s
    m = np.random.random(10)*u.M_sun
    
    pc2 = Particle(r=r, v=v, m=m, units=units)
    
    pc_merged = pc1.merge(pc2)
    
    assert pc_merged._r.shape == (20,3)

def test_to():
    r = np.random.random(size=(10,3))*u.kpc
    v = np.random.random(size=(10,3))*u.kpc/u.Myr
    m = np.random.random(10)*u.M_sun
    
    pc = Particle(r=r, v=v, m=m, units=units)
    
    usys2 = (u.km, u.s, u.kg)    
    pc2 = pc.to(usys2)
    
    assert np.allclose(pc2._r, r.to(u.km).value)
    assert np.allclose(pc2._v, v.to(u.km/u.s).value)
    assert np.allclose(pc2._m, m.to(u.kg).value)

def test_getitem():
    r = np.random.random(size=(100,3))*u.kpc
    v = np.random.random(size=(100,3))*u.kpc/u.Myr
    m = np.random.random(100)*u.M_sun
    
    pc = Particle(r=r, v=v, m=m, units=units)
    pc2 = pc[15:30]
    
    assert pc2.nparticles == 15
    assert (pc2._r[0] == pc._r[15]).all()