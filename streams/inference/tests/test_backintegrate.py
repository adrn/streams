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
from ...integrate import SatelliteParticleIntegrator
from ..backintegrate import *
from ...data.simulation import lm10_time, lm10_particles, lm10_satellite

t1,t2 = lm10_time()
particles = lm10_particles(N=100, expr="Pcol > 0")
satellite = lm10_satellite()

potential = LawMajewski2010()

usys = UnitSystem(u.kpc, u.Myr, u.M_sun, u.radian)
integrator = SatelliteParticleIntegrator(potential, satellite, particles)
timestep = lambda *args, **kwargs: -1

satellite_orbit,particle_orbits = integrator.run(timestep_func=timestep,
                                                 timestep_args=(potential, satellite.m.value),
                                                 resolution=3.,
                                                 t1=t1, t2=t2)

def test_relative_normalized_coordinates():
    a = time.time()
    R,V = relative_normalized_coordinates(potential, 
                                          particle_orbits, 
                                          satellite_orbit)
    print("R,V: {0:.3f} ms".format(1000.*(time.time()-a)))

def test_minimum_distance_matrix():
    #a = time.time()
    minimum_distance_matrix(potential, particle_orbits, satellite_orbit)
    #print("min. dist. matrix: {0:.3f} ms".format(1000.*(time.time()-a)))

def test_generalized_variance():
    a = time.time()
    v = generalized_variance(potential, particle_orbits, satellite_orbit)
    print(v)
    print("gen. variance: {0:.3f} ms".format(1000.*(time.time()-a)))

def test_vary_potential():
    
    for q1 in np.linspace(1.,2.,10):
        potential = LawMajewski2010(q1=q1)
    
        usys = UnitSystem(u.kpc, u.Myr, u.M_sun, u.radian)
        integrator = SatelliteParticleIntegrator(potential, satellite, particles)
        timestep = lambda *args, **kwargs: -1
        
        satellite_orbit,particle_orbits = integrator.run(timestep_func=timestep,
                                                         timestep_args=(potential, satellite.m.value),
                                                         resolution=3.,
                                                         t1=t1, t2=t2)
        
        v = generalized_variance(potential, particle_orbits, satellite_orbit)
        print (q1, v)
