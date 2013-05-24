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

from ..verlet import verlet
from ..leapfrog import leapfrog
from ...potential import *

plot_path = "plots/tests/integrate"
animation_path = os.path.join(plot_path, "animation")

if not os.path.exists(plot_path):
    os.makedirs(plot_path)

gal_units = [u.kpc,u.Myr,u.M_sun,u.radian]

class TestBoxOnSpring(object):
    
    @pytest.mark.parametrize(("name","integrator"), [('leapfrog',leapfrog), 
                                                ('verlet',verlet)])
    def test_time_series(self, name, integrator):

        k = 2.E1 # g/s^2
        m = 10. # g
        g = 980. # cm/s^2

        acceleration = lambda x: -k/m * x + g
        
        dt = 0.1        
        t = np.arange(0, 100, dt)
        ts, xs, vs = integrator(acceleration, [4.], [0.], t=t)
        
        plt.clf()
        plt.plot(ts, xs[:,0,0], 'k-')
        plt.savefig(os.path.join(plot_path,"box_spring_{0}.png".format(name)))

def plot_energies(potential, ts, xs, vs, axes1=None):
    E_kin = 0.5*np.sum(vs**2, axis=2)
    E_pot = potential._value_at(xs[:,0,:])
    
    if axes1 == None:
        fig1, axes1 = plt.subplots(3,1,sharex=True,sharey=True,figsize=(12,8))

    axes1[0].plot(ts, E_kin[:,0])
    axes1[0].set_ylabel(r"$E_{kin}$")

    axes1[1].plot(ts, E_pot)
    axes1[1].set_ylabel(r"$E_{pot}$")

    axes1[2].plot(ts, (E_kin[:,0] + E_pot))
    axes1[2].set_ylabel(r"$E_{tot}$")

    grid = np.linspace(-10., 10., 100)*u.kpc
    fig2, axes2 = potential.plot(ndim=3, grid=grid)
    axes2[0,0].plot(xs[:,0,0], xs[:,0,1], color='w')
    axes2[1,0].plot(xs[:,0,0], xs[:,0,2], color='w')
    axes2[1,1].plot(xs[:,0,1], xs[:,0,2], color='w')
    
    return fig1,fig2 

class TestIntegrate(object):

    @pytest.mark.parametrize(("name","integrator"), [('leapfrog',leapfrog), 
                                                     ('verlet',verlet)])    
    def test_point_mass_energy(self, name, integrator):
        potential = PointMassPotential(unit_system=UnitSystem(u.M_sun, u.au, u.yr),
                                       m=1*u.M_sun,
                                       r_0=[0.,0.,0.]*u.au)

        initial_position = np.array([1.0, 0.0, 0.]) # au
        initial_velocity = np.array([0.0, 2*np.pi, 0.]) # au/yr

        ts, xs, vs = integrator(potential._acceleration_at, 
                                initial_position,
                                initial_velocity, 
                                t1=0., t2=5., dt=0.05)
        fig1,fig2 = plot_energies(potential,ts, xs, vs)
        fig1.savefig(os.path.join(plot_path,"point_mass_energy_{0}.png".format(name)))
        fig2.savefig(os.path.join(plot_path,"point_mass_{0}.png".format(name)))
        
    @pytest.mark.parametrize(("name","integrator"), [('leapfrog',leapfrog), 
                                                     ('verlet',verlet)])  
    def test_hernquist_energy(self, name, integrator):
        potential = HernquistPotential(units=[u.M_sun, u.kpc, u.Myr],
                                           m=1E10*u.M_sun, 
                                           c=0.7*u.kpc)

        initial_position = np.array([10.0, 0.0, 0.]) # kpc
        initial_velocity = np.array([0.0, (30.*u.km/u.s).to(u.kpc/u.Myr).value, 0.]) # kpc/Myr

        ts, xs, vs = integrator(potential.acceleration_at, 
                              initial_position,
                              initial_velocity, 
                              t1=0., t2=1000., dt=1.)
        fig1,fig2 = plot_energies(potential,ts, xs, vs)
        fig1.savefig(os.path.join(plot_path,"hernquist_energy.png"))
    
    @pytest.mark.parametrize(("name","integrator"), [('leapfrog',leapfrog), 
                                                     ('verlet',verlet)])  
    def test_miyamoto_energy(self, name, integrator):
        potential = MiyamotoNagaiPotential(units=[u.M_sun, u.kpc, u.Myr],
                                               m=1E11*u.M_sun, 
                                               a=6.5*u.kpc, 
                                               b=0.26*u.kpc)

        initial_position = np.array([8.0, 0.0, 0.]) # kpc
        initial_velocity = np.array([0.0, (200.*u.km/u.s).to(u.kpc/u.Myr).value, 0.]) # kpc/Myr

        ts, xs, vs = integrator(potential.acceleration_at, 
                              initial_position, initial_velocity, 
                              t1=0., t2=1000., dt=1.)
        fig1,fig2 = plot_energies(potential,ts, xs, vs)
        fig1.savefig(os.path.join(plot_path,"miyamoto_energy.png"))
    
    @pytest.mark.parametrize(("name","integrator"), [('leapfrog',leapfrog), 
                                                     ('verlet',verlet)])  
    def test_log_potential(self, name, integrator):
        potential = LogarithmicPotentialLJ(units=[u.kpc,u.Myr,u.M_sun,u.radian],
                                               v_halo=(121.858*u.km/u.s).to(u.kpc/u.Myr),
                                               q1=1.38,
                                               q2=1.0,
                                               qz=1.36,
                                               phi=1.692969*u.radian,
                                               r_halo=12.*u.kpc)

        initial_position = np.array([14.0, 0.0, 0.]) # kpc
        initial_velocity = np.array([0.0, (160.*u.km/u.s).to(u.kpc/u.Myr).value, 0.]) # kpc/Myr

        ts, xs, vs = integrator(potential.acceleration_at, 
                              initial_position, 
                              initial_velocity, 
                              t1=0., t2=6000., dt=1.)
        plot_energies(potential,ts, xs, vs)
        fig1,fig2 = plot_energies(potential,ts, xs, vs)
        fig1.savefig(os.path.join(plot_path,"logarithmic_energy.png"))