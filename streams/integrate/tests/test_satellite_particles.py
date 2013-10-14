# coding: utf-8
"""
    Test the ...
"""

from __future__ import absolute_import, unicode_literals, division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

import cProfile
import pstats
import os
import pytest
import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt

from streams.dynamics import Particle
from streams.potential.lm10 import LawMajewski2010
from streams.integrate.satellite_particles import satellite_particles_integrate
from streams.integrate.leapfrog import LeapfrogIntegrator

plot_path = "plots/tests/integrate/satellite_particles"
if not os.path.exists(plot_path):
    os.makedirs(plot_path)

potential = LawMajewski2010()
def test_random_particles():
    Nparticles = 100
    acc = np.zeros((Nparticles+1,3))

    satellite = Particle(r=[0.,0.,5.]*u.kpc, v=[0.,0.121,0.]*u.kpc/u.Myr, m=1E8*u.M_sun)
    particles = Particle(r=np.random.random((Nparticles,3))*u.kpc, 
                         v=0.1*np.random.random((Nparticles,3))*u.kpc/u.Myr,
                         units=[u.kpc,u.Myr,u.M_sun])
    
    s, p = satellite_particles_integrate(satellite, particles, potential,
                                         potential_args=(Nparticles+1, acc), \
                                         time_spec=dict(t1=0., t2=6000., dt=1.))
    
    plt.plot(s.t, np.sum(s._r**2, axis=-1), alpha=0.75, marker=None)
    plt.savefig(os.path.join(plot_path,"satellite.png"))

    plt.clf()
    for ii in range(5):
        plt.plot(s.t, np.sum(p._r[:,ii]**2, axis=-1), 
                 color='k', alpha=0.25, marker=None)
    plt.savefig(os.path.join(plot_path,"particles.png"))
    