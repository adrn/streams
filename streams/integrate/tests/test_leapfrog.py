# coding: utf-8
"""
    Test the ...
"""

from __future__ import absolute_import, unicode_literals, division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

import os
import pytest
import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt

from ..leapfrog import *
from ... import potential as pot

plot_path = "plots/tests/simulation"
animation_path = os.path.join(plot_path, "animation")

if not os.path.exists(plot_path):
    os.mkdir(plot_path)

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
            a_i = acceleration(x_i)

            x_ip1 = x_i + v_i*dt + 0.5*a_i*dt*dt
            a_ip1 = acceleration(x_ip1)
            v_ip1 = v_i + 0.5*(a_i + a_ip1)*dt

            xs[ii] = x_i
            vs[ii] = v_i

            a_i = a_ip1
            x_i = x_ip1
            v_i = v_ip1

        plt.plot(times, xs, 'k-')
        plt.savefig(os.path.join(plot_path,"box_spring.png"))

def plot_energies(potential, ts, xs, vs, axes1=None):
    E_kin = 0.5*np.sum(vs**2, axis=2)
    E_pot = potential.value_at(xs[:,0,:])
    
    if axes1 == None:
        fig1, axes1 = plt.subplots(3,1,sharex=True,sharey=True,figsize=(12,8))

    axes1[0].plot(ts, E_kin[:,0])
    axes1[0].set_ylabel(r"$E_{kin}$")

    axes1[1].plot(ts, E_pot)
    axes1[1].set_ylabel(r"$E_{pot}$")

    axes1[2].plot(ts, (E_kin[:,0] + E_pot))
    axes1[2].set_ylabel(r"$E_{tot}$")

    grid = np.linspace(-20., 20., 100)*u.kpc
    fig2, axes2 = potential.plot(grid,grid,grid)
    axes2[0,0].plot(xs[:,0,0], xs[:,0,1])
    axes2[1,0].plot(xs[:,0,0], xs[:,0,2])
    axes2[1,1].plot(xs[:,0,1], xs[:,0,2])
    
    return fig1,fig2 

class TestIntegrate(object):
    
    def test_point_mass_energy(self):
        potential = pot.PointMassPotential(unit_bases=[u.M_sun, u.au, u.yr],
                                           m=1*u.M_sun)

        initial_position = np.array([1.0, 0.0, 0.]) # au
        initial_velocity = np.array([0.0, 2*np.pi, 0.]) # au/yr

        ts, xs, vs = leapfrog(potential.acceleration_at, 
                              initial_position,
                              initial_velocity, 
                              t1=0., t2=5., dt=0.05)
        fig1,fig2 = plot_energies(potential,ts, xs, vs)
        fig1.savefig(os.path.join(plot_path,"point_mass_energy.png"))
    
    def test_hernquist_energy(self):
        potential = pot.HernquistPotential(unit_bases=[u.M_sun, u.kpc, u.Myr],
                                           m=1E10*u.M_sun, 
                                           c=0.7*u.kpc)

        initial_position = np.array([10.0, 0.0, 0.]) # kpc
        initial_velocity = np.array([0.0, (30.*u.km/u.s).to(u.kpc/u.Myr).value, 0.]) # kpc/Myr

        ts, xs, vs = leapfrog(potential.acceleration_at, 
                              initial_position,
                              initial_velocity, 
                              t1=0., t2=1000., dt=1.)
        fig1,fig2 = plot_energies(potential,ts, xs, vs)
        fig1.savefig(os.path.join(plot_path,"hernquist_energy.png"))

    def test_miyamoto_energy(self):
        potential = pot.MiyamotoNagaiPotential(unit_bases=[u.M_sun, u.kpc, u.Myr],
                                               m=1E11*u.M_sun, 
                                               a=6.5*u.kpc, 
                                               b=0.26*u.kpc)

        initial_position = np.array([8.0, 0.0, 0.]) # kpc
        initial_velocity = np.array([0.0, (200.*u.km/u.s).to(u.kpc/u.Myr).value, 0.]) # kpc/Myr

        ts, xs, vs = leapfrog(potential.acceleration_at, 
                              initial_position, initial_velocity, 
                              t1=0., t2=1000., dt=1.)
        fig1,fig2 = plot_energies(potential,ts, xs, vs)
        fig1.savefig(os.path.join(plot_path,"miyamoto_energy.png"))

    def test_log_potential(self):
        potential = pot.LogarithmicPotentialLJ(unit_bases=[u.kpc,u.Myr,u.M_sun,u.radian],
                                               v_halo=(121.858*u.km/u.s).to(u.kpc/u.Myr),
                                               q1=1.38,
                                               q2=1.0,
                                               qz=1.36,
                                               phi=1.692969*u.radian,
                                               r_halo=12.*u.kpc)

        initial_position = np.array([14.0, 0.0, 0.]) # kpc
        initial_velocity = np.array([0.0, (160.*u.km/u.s).to(u.kpc/u.Myr).value, 0.]) # kpc/Myr

        ts, xs, vs = leapfrog(potential.acceleration_at, 
                              initial_position, 
                              initial_velocity, 
                              t1=0., t2=6000., dt=1.)
        plot_energies(potential,ts, xs, vs)
        fig1,fig2 = plot_energies(potential,ts, xs, vs)
        fig1.savefig(os.path.join(plot_path,"logarithmic_energy.png"))

    def test_three_component(self):
        bulge = pot.HernquistPotential(unit_bases=[u.M_sun, u.kpc, u.Myr],
                                           m=1E10*u.M_sun, 
                                           c=0.7*u.kpc)
        disk = pot.MiyamotoNagaiPotential(unit_bases=[u.M_sun, u.kpc, u.Myr],
                                               m=1E11*u.M_sun, 
                                               a=6.5*u.kpc, 
                                               b=0.26*u.kpc)
        
        halo = pot.LogarithmicPotentialLJ(unit_bases=[u.kpc,u.Myr,u.M_sun,u.radian],
                                               v_halo=(121.858*u.km/u.s).to(u.kpc/u.Myr),
                                               q1=1.38,
                                               q2=1.0,
                                               qz=1.36,
                                               phi=1.692969*u.radian,
                                               r_halo=12.*u.kpc)
        
        potential = disk + bulge + halo

        initial_position = np.array([14.0, 0.0, 5.]) # kpc
        initial_velocity = np.array([(40.*u.km/u.s).to(u.kpc/u.Myr).value,
                                     (130.*u.km/u.s).to(u.kpc/u.Myr).value,
                                     (40.*u.km/u.s).to(u.kpc/u.Myr).value]) # kpc/Myr

        fig, axes = plt.subplots(3, 1, sharex=True, sharey=True)
        ts, xs, vs = leapfrog(potential.acceleration_at, 
                              initial_position, 
                              initial_velocity, 
                              t1=0., t2=6000., dt=1.)
        plot_energies(potential,ts, xs, vs)
        fig1,fig2 = plot_energies(potential,ts, xs, vs)
        fig1.savefig(os.path.join(plot_path,"three_component_energy.png"))