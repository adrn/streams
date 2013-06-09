# coding: utf-8
""" Test the special fucntions used to compute our scalar objective """

from __future__ import absolute_import, unicode_literals, division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys
import time

# Third-party
import numpy as np
import pytest
import astropy.units as u

from ...misc.units import UnitSystem
from ...potential.lm10 import LawMajewski2010
from ...nbody import OrbitCollection
from ..backintegrate import *

potential = LawMajewski2010()

usys = UnitSystem(u.kpc, u.Myr, u.M_sun, u.radian)

t = np.arange(0., 1000., 1.)*u.Myr
r = np.zeros((len(t),1,3))
r[:,0,0] = np.linspace(1., 50., len(t))
r = r*u.kpc

v = np.zeros((len(t),1,3))
v[:,0,0] = np.linspace(200., 0., len(t))
v = v*u.km/u.s

m = [2.5E8] * u.M_sun

satellite_orbit = OrbitCollection(t=t, r=r, v=v, m=m, 
                                  unit_system=usys)

Nparticles = 1000
t = np.arange(0., 1000., 1.)*u.Myr
r = np.zeros((len(t),Nparticles,3))
r[:,:,0] = np.random.uniform(size=(len(t),Nparticles)) * np.linspace(1., 50., len(t)).reshape((len(t),1))
r = r*u.kpc

v = np.zeros((len(t),Nparticles,3))
v[:,:,0] = np.random.uniform(size=(len(t),Nparticles)) * np.linspace(200., 0., len(t)).reshape((len(t),1))
v = v*u.km/u.s
                                  
particle_orbits = OrbitCollection(t=t, r=r, v=v, m=m, 
                                  unit_system=usys)

def test_relative_normalized_coordinates():
    a = time.time()
    R,V = relative_normalized_coordinates(potential, 
                                          particle_orbits, 
                                          satellite_orbit)
    print("R,V: {0:.3f} ms".format(1000.*(time.time()-a)))

'''
def test_relative_normalized_coordinates_sgr():
    from ...data.sgr import lm10_satellite, lm10_particles
    
    # TODO: Need to integrate both, get orbits
    particles = lm10_particles(N=1000)
    satellite = lm10_satellite()
    
    R,V = relative_normalized_coordinates(potential, 
                                          particles, 
                                          satellite)
    
    print(R.shape, V.shape)
'''
 
def test_minimum_distance_matrix():
    #a = time.time()
    minimum_distance_matrix(potential, particle_orbits, satellite_orbit)
    #print("min. dist. matrix: {0:.3f} ms".format(1000.*(time.time()-a)))

def test_generalized_variance():
    a = time.time()
    generalized_variance(potential, particle_orbits, satellite_orbit)
    print("gen. variance: {0:.3f} ms".format(1000.*(time.time()-a)))