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

from ...misc.units import UnitSystem
from ...potential.lm10 import LawMajewski2010, true_params
from ...integrate import SatelliteParticleIntegrator
from ..core import *
from ...io.lm10 import time, particles_today, satellite_today

plot_path = "plots/tests/inference"
if not os.path.exists(plot_path):
    os.makedirs(plot_path)

t1,t2 = time()
particles = particles_today(N=100, expr="Pcol > 0")
satellite = satellite_today()

potential = LawMajewski2010()

usys = UnitSystem(u.kpc, u.Myr, u.M_sun, u.radian)
integrator = SatelliteParticleIntegrator(potential, satellite, particles)
timestep = lambda *args, **kwargs: -1

satellite_orbit,particle_orbits = integrator.run(t1=t1, t2=t2, dt=-1.)

def test_relative_normalized_coordinates():
    a = pytime.time()
    R,V = relative_normalized_coordinates(potential, 
                                          particle_orbits, 
                                          satellite_orbit)
    print("R,V: {0:.3f} ms".format(1000.*(pytime.time()-a)))

def test_minimum_distance_matrix():
    #a = pytime.time()
    minimum_distance_matrix(potential, particle_orbits, satellite_orbit)
    #print("min. dist. matrix: {0:.3f} ms".format(1000.*(pytime.time()-a)))

def test_generalized_variance():
    a = pytime.time()
    v = generalized_variance(potential, particle_orbits, satellite_orbit)
    print(v)
    print("gen. variance: {0:.3f} ms".format(1000.*(pytime.time()-a)))

@pytest.mark.parametrize(("param", ), [('q1',),('qz',),('v_halo',),('phi',)])
def test_vary_potential(param):
    """ Vary each LM10 potential parameter, plot the generalized variance 
        vs. the parameter values.
    """
    usys = UnitSystem(u.kpc, u.Myr, u.M_sun, u.radian)
    Nbins = 21
    
    variances = []
    params = true_params.copy()
    for val in np.linspace(true_params[param]*0.75, 
                           true_params[param]*1.25, Nbins):
        params[param] = val
        
        # create potential with all 'true' params, except the one we're varying
        potential = LawMajewski2010(**params)

        integrator = SatelliteParticleIntegrator(potential, satellite, particles)        
        satellite_orbit,particle_orbits = integrator.run(t1=t1, t2=t2, dt=-1.)
        v = generalized_variance(potential, particle_orbits, satellite_orbit)
        variances.append(v)
    
    plt.clf()
    plt.plot(np.linspace(0.75, 1.25, Nbins), variances)
    plt.title(param)
    plt.savefig(os.path.join(plot_path, "vary_{0}.png".format(param)))