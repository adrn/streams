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

from ..core import Particle, ParticleCollection

# Tests for the Particle class -- *not* a test particle!!!
class TestParticle(object):
    
    def test_creation(self):
        
        # Not quantities
        with pytest.raises(TypeError):
            Particle(15., 16.)
            
        with pytest.raises(TypeError):
            Particle(15.*u.kpc, 16.)
        
        with pytest.raises(TypeError):
            Particle(15., 16.*u.kpc/u.Myr)
        
        # make 1D case vectors
        p = Particle(15.*u.kpc, 16.*u.kpc/u.Myr)
        assert len(p.r) == 1 and len(p.v) == 1
        
        # don't support arrays
        with pytest.raises(ValueError):
            Particle([[15.,11.,13],[15.,11.,13]]*u.kpc, 
                     [[15.,11.,13],[15.,11.,13]]*u.kpc/u.Myr)
        
        with pytest.raises(ValueError):
            Particle([[15.,11.,13],[15.,11.,13]]*u.kpc, 
                     [[15.,11.,13],[15.,11.,13]]*u.kpc/u.Myr)
        
        p = Particle([0.,8.,0.2]*u.kpc, 
                     [10.,200.,-15.]*u.km/u.s)


class TestParticleCollection(object):
    
    def test_creation(self):
        
        particles = [Particle(r=np.random.random(size=3)*u.kpc, 
                              v=np.random.random(size=3)*u.kpc) for ii in range(100)]
        
        pc = ParticleCollection(particles=particles)
        
        pc = ParticleCollection(r=np.random.random(size=500)*u.kpc,
                                v=np.random.random(size=500)*u.kpc)
        
        # Size mismatch
        with pytest.raises(ValueError):
            pc = ParticleCollection(r=np.random.random(size=500)*u.kpc,
                                    v=np.random.random(size=501)*u.kpc)
        
        with pytest.raises(ValueError):
            pc = ParticleCollection(r=np.random.random(size=501)*u.kpc,
                                    v=np.random.random(size=500)*u.kpc)
        
        # test adding particle with wrong type/dimensionality
        particles = [Particle(r=np.random.random(size=3)*u.kpc, 
                              v=np.random.random(size=3)*u.kpc) for ii in range(10)]
        
        with pytest.raises(TypeError):
            pc = ParticleCollection(particles=particles+[16])
        
        with pytest.raises(ValueError):
            pc = ParticleCollection(particles=particles+[Particle([0]*u.kpc,[0]*u.kpc)])
    
    def test_slicing(self):
        pc = ParticleCollection(r=np.random.random(size=500)*u.kpc,
                                v=np.random.random(size=500)*u.km/u.s)
        
        assert isinstance(pc[0], Particle)
        
        assert isinstance(pc[0:15], ParticleCollection)
        