# coding: utf-8
""" Test the special fucntions used to compute our scalar objective """

from __future__ import absolute_import, unicode_literals, division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys
import time as pytime

# Third-party
import numpy as np
import pytest
import astropy.units as u
import matplotlib.pyplot as plt

from ...potential.lm10 import LawMajewski2010, true_params
from ...integrate import satellite_particles_integrate
from .. import *
from ...io.lm10 import time, particles_today, satellite_today

plot_path = "plots/tests/inference"
if not os.path.exists(plot_path):
    os.makedirs(plot_path)

t1,t2 = time()
particles = particles_today(N=100, expr="Pcol > 0")
satellite = satellite_today()

potential = LawMajewski2010()
Nparticles = len(particles)
acc = np.zeros((Nparticles+1,3)) # placeholder
satellite_orbit,particle_orbits = satellite_particles_integrate(satellite, particles,
                                                 potential,
                                                 potential_args=(Nparticles+1, acc), \
                                                 time_spec=dict(t1=t1, t2=t2, dt=-1.))

def test_relative_normalized_coordinates():
    a = pytime.time()
    R,V = relative_normalized_coordinates(potential, satellite_orbit, particle_orbits)
    print("R,V: {0:.3f} ms".format(1000.*(pytime.time()-a)))

def test_minimum_distance_matrix():
    #a = pytime.time()
    minimum_distance_matrix(potential, satellite_orbit, particle_orbits)
    #print("min. dist. matrix: {0:.3f} ms".format(1000.*(pytime.time()-a)))

def test_generalized_variance():
    a = pytime.time()
    v = generalized_variance(potential, satellite_orbit, particle_orbits)
    print(v)
    print("gen. variance: {0:.3f} ms".format(1000.*(pytime.time()-a)))