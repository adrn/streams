# coding: utf-8

""" """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys

# Third-party
import numpy as np
import astropy.units as u

class ParticleSimulation(object):
    pass

# API?:

simulation = ParticleSimulation(length_unit=u.kpc, time_unit=u.Myr, mass_unit=u.solMass)
simulation.set_potential(potential)

for ii in range(100):
    p = Particle(position=(np.random.uniform(1., 10.), 0., np.random.uniform(-0.5, 0.5)), # kpc
                 velocity=((np.random.uniform(-10., 10.)*u.km/u.s).to(u.kpc/u.Myr), (200*u.km/u.s).to(u.kpc/u.Myr), (np.random.uniform(-10., 10.)*u.km/u.s).to(u.kpc/u.Myr)), # kpc/Myr
                 mass=1.) # M_sol
    simulation.add_particle(p)

simulation.run(t1=0., t2=1000., dt=0.1)