# coding: utf-8
"""
    Test the ...
"""

from __future__ import absolute_import, unicode_literals, division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

import pytest
import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt

from .._leapfrog import leapfrog2
from ..leapfrog import *
from ... import potential as pot

class TestIntegrate(object):

    def test1(self):
        potential = pot.HernquistPotential(M=1E10, c=0.7)

        initial_position = np.random.uniform(0.5, 1.5, size=(200,3)) # kpc
        initial_velocity = np.random.uniform(0.01, 0.05, size=(200,3)) # kpc/Myr

        import time

        time1 = time.time()
        ts, xs, vs = leapfrog(potential.acceleration_at, initial_position, initial_velocity, 0., 1000., 0.1)
        print("python: ", time.time()-time1)

        time2 = time.time()
        ts, xs, vs = leapfrog2(potential.acceleration_at, initial_position, initial_velocity, 0., 1000., 0.1)
        print("cython: ", time.time()-time2)

        assert xs.shape == (len(ts),) + initial_position.shape
        assert vs.shape == (len(ts),) + initial_position.shape

        return

    def test_energy(self):
        potential = pot.HernquistPotential(M=1E10, c=0.7)

        initial_position = np.array([1.0, 0.0, 0.]) # kpc
        initial_velocity = np.array([0.0, (100.*u.km/u.s).to(u.kpc/u.Myr).value, 0.]) # kpc/Myr

        ts, xs, vs = leapfrog(potential.acceleration_at, initial_position, initial_velocity, 0., 1000., 1.)

        E_kin = 0.5*np.sum(vs**2, axis=2)
        E_pot = potential.value_at(xs)

        energies = potential.energy_at(xs, vs)

        fig, axes = plt.subplots(3,1,sharex=True,sharey=True)
        axes[0].plot(ts, E_kin)
        axes[0].set_ylabel(r"$E_{kin}$")

        axes[1].plot(ts, E_pot.T)
        axes[1].set_ylabel(r"$E_{pot}$")

        axes[2].plot(ts, (E_kin + E_pot.T), 'k.')
        axes[2].set_ylabel(r"$E_{tot}$")

        grid = np.linspace(-2., 2., 100)
        fig, axes = potential.plot(grid,grid,grid)
        axes[0,0].plot(xs[:,0,0], xs[:,0,1])
        axes[1,0].plot(xs[:,0,0], xs[:,0,2])
        axes[1,1].plot(xs[:,0,1], xs[:,0,2])
