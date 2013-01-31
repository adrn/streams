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

show_plots = True

class TestBoxOnSpring(object):

    def test_time_series(self):

        k = 2.E1 # g/s^2
        m = 10. # g
        g = 980. # cm/s^2

        acceleration = lambda x: -k/m * x + g

        #ts, xs, vs = leapfrog(acceleration, [4.], [0.], 0., 100., 0.01)

        dt = 0.01
        times = np.arange(0, 100, dt)

        Ntimesteps = len(times)
        xs = np.zeros(Ntimesteps)
        vs = np.zeros(Ntimesteps)

        x_i = np.array([1.])
        v_i = np.array([0.])

        for ii in range(Ntimesteps):
            t = times[ii]
            a_i = acceleration(t, x_i)

            x_ip1 = x_i + v_i*dt + 0.5*a_i*dt*dt
            a_ip1 = acceleration(x_ip1)
            v_ip1 = v_i + 0.5*(a_i + a_ip1)*dt

            xs[ii] = x_i
            vs[ii] = v_i

            a_i = a_ip1
            x_i = x_ip1
            v_i = v_ip1

        plt.plot(times, xs, 'k-')
        if show_plots: plt.show()

def plot_energies(potential, ts, xs, vs, axes1=None):
    E_kin = 0.5*np.sum(vs**2, axis=2)
    E_pot = potential.value_at(xs)

    energies = potential.energy_at(xs, vs)

    if axes1 == None:
        fig1, axes1 = plt.subplots(3,1,sharex=True,sharey=True)

    axes1[0].plot(ts, E_kin)
    axes1[0].set_ylabel(r"$E_{kin}$")

    axes1[1].plot(ts, E_pot.T)
    axes1[1].set_ylabel(r"$E_{pot}$")

    axes1[2].plot(ts, (E_kin + E_pot.T))
    axes1[2].set_ylabel(r"$E_{tot}$")

    grid = np.linspace(-20., 20., 100)
    fig2, axes2 = potential.plot(grid,grid,grid)
    axes2[0,0].plot(xs[:,0,0], xs[:,0,1])
    axes2[1,0].plot(xs[:,0,0], xs[:,0,2])
    axes2[1,1].plot(xs[:,0,1], xs[:,0,2])

class TestIntegrate(object):

    def test_speed(self):
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

    def test_hernquist_energy(self):
        potential = pot.HernquistPotential(M=1E10*u.solMass, c=0.7)

        initial_position = np.array([10.0, 0.0, 0.]) # kpc
        initial_velocity = np.array([0.0, (30.*u.km/u.s).to(u.kpc/u.Myr).value, 0.]) # kpc/Myr

        ts, xs, vs = leapfrog(potential.acceleration_at, initial_position, initial_velocity, 0., 1000., 1.)
        plot_energies(potential,ts, xs, vs)
        if show_plots: plt.show()

    def test_miyamoto_energy(self):
        potential = pot.MiyamotoNagaiPotential(M=1E11*u.solMass, a=6.5, b=0.26)

        initial_position = np.array([8.0, 0.0, 0.]) # kpc
        initial_velocity = np.array([0.0, (200.*u.km/u.s).to(u.kpc/u.Myr).value, 0.]) # kpc/Myr

        ts, xs, vs = leapfrog(potential.acceleration_at, initial_position, initial_velocity, 0., 1000., 1.)
        plot_energies(potential,ts, xs, vs)
        if show_plots: plt.show()

    def test_log_potential(self):
        potential = pot.LogarithmicPotential(v_circ=(181.*u.km/u.s).to(u.kpc/u.Myr).value, p=1., q=0.75, c=12.)

        initial_position = np.array([14.0, 0.0, 0.]) # kpc
        initial_velocity = np.array([0.0, (160.*u.km/u.s).to(u.kpc/u.Myr).value, 0.]) # kpc/Myr

        ts, xs, vs = leapfrog(potential.acceleration_at, initial_position, initial_velocity, 0., 6000., 1.)
        plot_energies(potential,ts, xs, vs)
        if show_plots: plt.show()

    def test_three_component(self):
        potential1 = pot.HernquistPotential(M=1E10*u.solMass, c=0.7)
        potential2 = pot.MiyamotoNagaiPotential(M=1E11*u.solMass, a=6.5, b=0.26)
        potential3 = pot.LogarithmicPotential(v_circ=(181.*u.km/u.s).to(u.kpc/u.Myr).value, p=1., q=1., c=12.)
        potential = potential1 + potential2 + potential3

        initial_position = np.array([14.0, 0.0, 5.]) # kpc
        initial_velocity = np.array([(40.*u.km/u.s).to(u.kpc/u.Myr).value,
                                     (130.*u.km/u.s).to(u.kpc/u.Myr).value,
                                     (40.*u.km/u.s).to(u.kpc/u.Myr).value]) # kpc/Myr

        fig, axes = plt.subplots(3, 1, sharex=True, sharey=True)
        ts, xs, vs = leapfrog(potential.acceleration_at, initial_position, initial_velocity, 0., 6000., 1.)
        plot_energies(potential, ts, xs, vs, axes)

        if show_plots: plt.show()