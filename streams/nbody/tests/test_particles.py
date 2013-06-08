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

from ..core import *

def test_particle_init():
    # Should work. Vanilla initialization
    p = Particle(r=np.array([1., 3., 5.])*u.kpc, 
                 v=np.array([50., 117., 90.])*u.km/u.s, 
                 m=1*u.M_sun)
    assert p.ndim == 3
    
    assert p.r.unit == u.kpc
    assert p.v.unit == u.km/u.s
    assert p.m.unit == u.M_sun
    
    # Vectors of different shape
    with pytest.raises(ValueError):
        p = Particle(r=np.array([1., 3., 5.])*u.kpc, 
                     v=np.array([50., 117.])*u.km/u.s, 
                     m=1*u.M_sun)
    with pytest.raises(ValueError):
        p = Particle(r=np.array([1., 3.])*u.kpc, 
                     v=np.array([50., 117., 90.])*u.km/u.s, 
                     m=1*u.M_sun)
    
    # Two-dimensional vectors
    with pytest.raises(ValueError):
        p = Particle(r=np.random.random(size=(3,5))*u.kpc, 
                     v=np.random.random(size=(3,5))*u.km/u.s, 
                     m=1*u.M_sun)
    
    # unit errors
    r = np.array([1., 3., 5.])*u.kpc
    v = np.array([50., 117., 90.])*u.km/u.s
    m = 1*u.M_sun
    
    with pytest.raises(TypeError): p = Particle(r=r.value,v=v,m=m)
    with pytest.raises(TypeError): p = Particle(r=r,v=v.value,m=m)
    with pytest.raises(TypeError): p = Particle(r=r,v=v,m=m.value)
    
def test_particlecollection_init():
    
    # Init with a list of Particle object
    particles = []
    for ii in range(10):
        p = Particle(r=np.random.random(3)*u.kpc, 
                     v=np.random.random(3)*u.kpc/u.Myr,
                     m=np.random.random()*u.M_sun)
        particles.append(p)
    
    # with and without a specified unit system
    usys = UnitSystem(u.kpc, u.Myr, u.M_sun)
    pc = ParticleCollection(particles=particles, unit_system=usys)
    assert pc.ndim == 3
    assert pc.r.unit == u.kpc
    assert pc.v.unit == u.kpc/u.Myr
    assert pc.m.unit == u.M_sun
    
    pc = ParticleCollection(particles=particles)
    assert pc.ndim == 3
    assert pc.r.unit == u.kpc
    assert pc.v.unit == u.kpc/u.Myr
    assert pc.m.unit == u.M_sun
    
    assert pc._r.shape == (10, 3)
    assert pc._v.shape == (10, 3)
    
    # position in kpc, velocity in km/s : should throw error if we don't give it
    #   a unit system because of 2 length units
    particles = []
    for ii in range(10):
        p = Particle(r=np.random.random(3)*u.kpc, 
                     v=np.random.random(3)*u.km/u.s,
                     m=np.random.random()*u.M_sun)
        particles.append(p)
    
    with pytest.raises(ValueError):
        pc = ParticleCollection(particles=particles)
    
    # one particle with wrong dimensions
    particles.append(Particle(r=np.random.random(2)*u.kpc, 
                              v=np.random.random(2)*u.km/u.s,
                              m=np.random.random()*u.M_sun))
    
    with pytest.raises(ValueError):
        pc = ParticleCollection(particles=particles)
    
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
    
    with pytest.raises(ValueError): pc = ParticleCollection(r=r, v=v, unit_system=usys)
    with pytest.raises(ValueError): pc = ParticleCollection(r=r, m=m, unit_system=usys)
    with pytest.raises(ValueError): pc = ParticleCollection(v=v, m=m, unit_system=usys)

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