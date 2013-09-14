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

from ...misc.units import UnitSystem
from ..leapfrog import LeapfrogIntegrator
from ...potential import *
from ...potential.pal5 import Palomar5

plot_path = "plots/tests/integrate"
animation_path = os.path.join(plot_path, "animation")

if not os.path.exists(plot_path):
    os.makedirs(plot_path)

gal_units = UnitSystem(u.M_sun, u.kpc, u.Myr, u.radian)

def plot_energies(potential, ts, xs, vs, axes1=None):
    E_kin = 0.5*np.sum(vs**2, axis=-1)
    if E_kin.ndim > 1:
        E_kin = E_kin[:,0]
        
    E_pot = potential._value_at(xs[:,0,:])    
    if E_pot.ndim > 1:
        E_pot = E_pot[:,0]
    
    if axes1 == None:
        fig1, axes1 = plt.subplots(3,1,sharex=True,figsize=(12,8))
    
    print(E_kin.shape, E_pot.shape)
    
    axes1[0].plot(ts, E_kin, marker=None)
    axes1[0].set_ylabel(r"$E_{kin}$")
    
    axes1[1].plot(ts, E_pot, marker=None)
    axes1[1].set_ylabel(r"$E_{pot}$")
    
    E_total = (E_kin + E_pot)
    axes1[2].semilogy(ts[1:], np.fabs((E_total[1:]-E_total[0])/E_total[0]), marker=None)
    axes1[2].set_ylabel(r"$\Delta E/E \times 100$")
    axes1[2].set_ylim(-1., 1.)
    
    grid = np.linspace(np.min(xs)-1., np.max(xs)+1., 100)*u.kpc
    fig2, axes2 = potential.plot(ndim=3, grid=grid)
    axes2[0,0].plot(xs[:,0,0], xs[:,0,1], color='w', marker=None)
    axes2[1,0].plot(xs[:,0,0], xs[:,0,2], color='w', marker=None)
    axes2[1,1].plot(xs[:,0,1], xs[:,0,2], color='w', marker=None)
    
    return fig1,fig2 

class TestBoxOnSpring(object):
    
    @pytest.mark.parametrize(("name","Integrator"), [('leapfrog',LeapfrogIntegrator), ])
    def test_time_series(self, name, Integrator):
        
        k = 2.E1 # g/s^2
        m = 10. # g
        g = 980. # cm/s^2

        acceleration = lambda x: -k/m * x + g
        
        dt = 0.1
        integrator = Integrator(acceleration, [[4.]], [[0.]])
        ts, xs, vs = integrator.run(dt=dt, Nsteps=1000)
        
        plt.clf()
        plt.plot(ts, xs[:,0,0], 'k-')
        plt.savefig(os.path.join(plot_path,"box_spring_{0}.png".format(name)))

class TestIntegrate(object):

    @pytest.mark.parametrize(("name","Integrator"), [('leapfrog',LeapfrogIntegrator), ])   
    def test_point_mass_energy(self, name, Integrator):
        potential = PointMassPotential(unit_system=UnitSystem(u.M_sun, u.au, u.yr),
                                       m=1*u.M_sun,
                                       r_0=[0.,0.,0.]*u.au)

        initial_position = np.array([[1.0, 0.0, 0.]]) # au
        initial_velocity = np.array([[0.0, 2*np.pi, 0.]]) # au/yr
        
        integrator = Integrator(potential._acceleration_at, 
                                initial_position, initial_velocity)
        ts, xs, vs = integrator.run(t1=0., t2=10., dt=0.01)
        
        fig1,fig2 = plot_energies(potential,ts, xs, vs)
        fig1.savefig(os.path.join(plot_path,"point_mass_energy_{0}.png".format(name)))
        fig2.savefig(os.path.join(plot_path,"point_mass_{0}.png".format(name)))
        
    @pytest.mark.parametrize(("name","Integrator"), [('leapfrog',LeapfrogIntegrator), ]) 
    def test_hernquist_energy(self, name, Integrator):
        potential = HernquistPotential(unit_system=gal_units,
                                           m=1E10*u.M_sun, 
                                           c=0.7*u.kpc)

        initial_position = np.array([10.0, 0.0, 0.]) # kpc
        initial_velocity = np.array([0.0, (30.*u.km/u.s).to(u.kpc/u.Myr).value, 0.]) # kpc/Myr
        
        integrator = Integrator(potential._acceleration_at, 
                                initial_position, initial_velocity)
        ts, xs, vs = integrator.run(t1=0., t2=1000., dt=1.)
        
        fig1,fig2 = plot_energies(potential,ts, xs, vs)
        fig1.savefig(os.path.join(plot_path,"hernquist_energy_{0}.png".format(name)))
        fig2.savefig(os.path.join(plot_path,"hernquist_{0}.png".format(name)))
    
    @pytest.mark.parametrize(("name","Integrator"), [('leapfrog',LeapfrogIntegrator), ]) 
    def test_miyamoto_energy(self, name, Integrator):
        potential = MiyamotoNagaiPotential(unit_system=gal_units,
                                               m=1E11*u.M_sun, 
                                               a=6.5*u.kpc, 
                                               b=0.26*u.kpc)

        initial_position = np.array([8.0, 0.0, 0.]) # kpc
        initial_velocity = np.array([0.0, (200.*u.km/u.s).to(u.kpc/u.Myr).value, 0.]) # kpc/Myr
        
        integrator = Integrator(potential._acceleration_at, 
                                initial_position, initial_velocity)
        ts, xs, vs = integrator.run(t1=0., t2=1000., dt=1.)
        
        fig1,fig2 = plot_energies(potential,ts, xs, vs)
        fig1.savefig(os.path.join(plot_path,"miyamoto_energy_{0}.png".format(name)))
        fig2.savefig(os.path.join(plot_path,"miyamoto_{0}.png".format(name)))
    
    @pytest.mark.parametrize(("name","Integrator"), [('leapfrog',LeapfrogIntegrator), ]) 
    def test_log_potential(self, name, Integrator):
        potential = LogarithmicPotentialLJ(unit_system=gal_units,
                                               v_halo=(121.858*u.km/u.s).to(u.kpc/u.Myr),
                                               q1=1.38,
                                               q2=1.0,
                                               qz=1.36,
                                               phi=1.692969*u.radian,
                                               r_halo=12.*u.kpc)

        initial_position = np.array([14.0, 0.0, 0.]) # kpc
        initial_velocity = np.array([0.0, (160.*u.km/u.s).to(u.kpc/u.Myr).value, 0.]) # kpc/Myr
        
        integrator = Integrator(potential._acceleration_at, 
                                initial_position, initial_velocity)
        ts, xs, vs = integrator.run(t1=0., t2=6000., dt=1.)
        
        plot_energies(potential,ts, xs, vs)
        fig1,fig2 = plot_energies(potential,ts, xs, vs)
        fig1.savefig(os.path.join(plot_path,"logarithmic_energy_{0}.png".format(name)))
        fig2.savefig(os.path.join(plot_path,"logarithmic_{0}.png".format(name)))
    
    @pytest.mark.parametrize(("name","Integrator"), [('leapfrog',LeapfrogIntegrator), ]) 
    def test_log_potential_adaptive(self, name, Integrator):
        potential = LogarithmicPotentialLJ(unit_system=gal_units,
                                               v_halo=(121.858*u.km/u.s).to(u.kpc/u.Myr),
                                               q1=1.38,
                                               q2=1.0,
                                               qz=1.36,
                                               phi=1.692969*u.radian,
                                               r_halo=12.*u.kpc)

        initial_position = np.array([[14.0, 0.0, 0.]]) # kpc
        initial_velocity = np.array([[0.0, (160.*u.km/u.s).to(u.kpc/u.Myr).value, 0.]]) # kpc/Myr
        
        def timestep():
            return np.random.random()*2.
        
        integrator = Integrator(potential._acceleration_at, 
                                initial_position, initial_velocity)
        
        t1 = 0.
        t2 = 6000.
        dt_i = dt_im1 = timestep()
        integrator._prime(dt_i)
        
        times = [t1]
        xs,vs = initial_position[np.newaxis], initial_velocity[np.newaxis]
        while times[-1] < t2:
            dt = 0.5*(dt_im1 + dt_i)
            r_i,v_i = integrator.step(dt)
            
            xs = np.vstack((xs,r_i[np.newaxis]))
            vs = np.vstack((vs,v_i[np.newaxis]))
            
            dt_i = timestep()
            times.append(times[-1] + dt)
            dt_im1 = dt_i
        
        ts = np.array(times)
        print(xs.shape, vs.shape, ts.shape)
        
        plot_energies(potential, ts, xs, vs)
        fig1,fig2 = plot_energies(potential,ts, xs, vs)
        fig1.savefig(os.path.join(plot_path,"logarithmic_energy_adaptive_{0}.png".format(name)))
        fig2.savefig(os.path.join(plot_path,"logarithmic_adaptive_{0}.png".format(name)))