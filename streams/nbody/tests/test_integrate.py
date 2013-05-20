# coding: utf-8

from __future__ import absolute_import, unicode_literals, division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys

# Third-party
import numpy as np
import pytest
import astropy.units as u
from astropy.constants import G
import matplotlib.pyplot as plt

from ..core import *
from ..integrate import *
from ..integrate import _nbody_acceleration

def test_acceleration():
    r = np.array([[1.,0.],
                  [0, 1.],
                  [-1., 0.],
                  [0., -1.]])
    m = np.array([1.,1.,1.,1.])
    _G = G.decompose(bases=[u.kpc,u.Myr,u.M_sun]).value
    
    a = _nbody_acceleration(_G, r, m)
    assert a.shape == r.shape
    
    print(a)

def test_api():
    
    # Create particles
    particles = []
    earth = Particle(r=np.array([1., 0., 0.])*u.au,
                     v=np.array([0., 2.*np.pi, 0.])*u.au/u.yr,
                     m=1.E-6*u.M_sun)
    particles.append(earth)
    
    sun = Particle(r=np.array([0.,0.,0.])*u.au,
                   v=np.array([0.,0.,0.])*u.au/u.yr,
                     m=1.*u.M_sun)
    particles.append(sun)
    
    pc = ParticleCollection(particles=particles, units=[u.au, u.yr, u.M_sun])
    
    # Create time grid to integrate on
    t = np.arange(0., 10., 0.02) * u.yr
    r,v = nbody_integrate(pc, time_steps=t, merge_length=1.E-3*u.au)
    
    plt.figure(figsize=(8,8))
    plt.plot(r[:,0,0], r[:,0,1], 'b-')
    plt.plot(r[:,1,0], r[:,1,1], 'b-')
    plt.show()

def test_collection():
    
    # Create particles
    particles = []
    p1 = Particle(r=np.array([1., 0., 0.])*u.au,
                  v=np.array([0., 2.*np.pi, 0.])*u.au/u.yr,
                  m=1.*u.M_sun)
    particles.append(p1)
    
    p2 = Particle(r=np.array([0., 1., 0.])*u.au,
                  v=np.array([-2.*np.pi, 0., 0.])*u.au/u.yr,
                  m=1.*u.M_sun)
    particles.append(p2)
    
    p3 = Particle(r=np.array([-1., 0., 0.])*u.au,
                  v=np.array([0., -2.*np.pi, 0.])*u.au/u.yr,
                  m=1.*u.M_sun)
    particles.append(p3)
    
    p4 = Particle(r=np.array([0., -1., 0.])*u.au,
                  v=np.array([2.*np.pi, 0., 0.])*u.au/u.yr,
                  m=1.*u.M_sun)
    particles.append(p4)
    
    p5 = Particle(r=np.array([0., 0., 0.])*u.au,
                  v=np.array([0., 0., 0.])*u.au/u.yr,
                  m=10.*u.M_sun)
    particles.append(p5)
    
    pc = ParticleCollection(particles=particles, units=[u.au, u.yr, u.M_sun])
    
    # Create time grid to integrate on
    t = np.arange(0., 10., 0.01) * u.yr
    r,v = nbody_integrate(pc, time_steps=t, merge_length=1.E-3*u.au)
    
    plt.figure(figsize=(8,8))
    plt.plot(r[:,0,0], r[:,0,1], 'k-')
    plt.plot(r[:,1,0], r[:,1,1], 'b-')
    plt.plot(r[:,2,0], r[:,2,1], 'r-')
    plt.plot(r[:,3,0], r[:,3,1], 'g-')
    plt.show()