# coding: utf-8
"""
    Test the core simulation code, e.g. Particle and ParticleCollection
"""

from __future__ import absolute_import, unicode_literals, division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys

# Third-party
import numpy as np
import pytest
import astropy.units as u

from ..core import Particle

# Tests for the Particle class -- *not* a test particle!!!
class TestParticle(object):
    
    def test_creation(self):
        # Not quantities
        with pytest.raises(TypeError):
            Particle(15., 16., 1.*u.M_sun)
            
        with pytest.raises(TypeError):
            Particle(15.*u.kpc, 16., 1.*u.M_sun)
        
        with pytest.raises(TypeError):
            Particle(15., 16.*u.kpc/u.Myr, 1.*u.M_sun)
        
        # make 1D case vectors
        p = Particle(15.*u.kpc, 16.*u.kpc/u.Myr, m=1.*u.M_sun)
        assert len(p.r) == 1 and len(p.v) == 1
        
        # 2D case
        p = Particle([15.,10.]*u.kpc, [160.,110.]*u.kpc/u.Myr, m=1.*u.M_sun)
        assert len(p.r) == 2 and len(p.v) == 2 and len(p.m) == 1
        
        # 3D case
        p = Particle([0.,8.,0.2]*u.kpc, 
                     [10.,200.,-15.]*u.km/u.s,
                     m=1.*u.M_sun)
        
        with pytest.raises(ValueError):
            p = Particle([[15.,11.,13],[15.,11.,13]]*u.kpc, 
                     [[15.,11.,13],[15.,11.,13]]*u.kpc/u.Myr,
                     m=[1.,1.,1.]*u.M_sun)
        
        # 3D, multiple particles
        p = Particle([[15.,11.,13],[15.,11.,13]]*u.kpc, 
                     [[15.,11.,13],[15.,11.,13]]*u.kpc/u.Myr,
                     m=[1.,1.]*u.M_sun)
        
        pc = Particle(r=np.random.random(size=500)*u.kpc,
                      v=np.random.random(size=500)*u.kpc,
                      m=np.ones(500)*u.M_sun)
        
        
        pc = Particle(r=np.random.random(size=(500,3))*u.kpc,
                      v=np.random.random(size=(500,3))*u.kpc,
                      m=np.ones(500)*u.M_sun)
        
        # Size mismatch
        with pytest.raises(ValueError):
            pc = Particle(r=np.random.random(size=(500,3))*u.kpc,
                          v=np.random.random(size=(501,3))*u.kpc,
                          m=np.ones(500)*u.M_sun)
        
        with pytest.raises(ValueError):
            pc = Particle(r=np.random.random(size=(501,3))*u.kpc,
                          v=np.random.random(size=(501,3))*u.kpc,
                          m=np.ones(500)*u.M_sun)
    
    def test_slicing(self):
        pc = Particle(r=np.random.random(size=500)*u.kpc,
                      v=np.random.random(size=500)*u.km/u.s,
                      m=np.ones(500)*u.M_sun)
        
        assert isinstance(pc[0], Particle)
        
        assert isinstance(pc[0:15], Particle)
        assert len(pc[0:15]) == 15
    