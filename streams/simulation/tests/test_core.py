# coding: utf-8
"""
    Test the core simulation code, e.g. Particle and ParticleCollection
"""

from __future__ import absolute_import, unicode_literals, division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys

# Third-party
import numpy as np
import pytest
import astropy.units as u
import matplotlib.pyplot as plt

from ..core import Particle
from ...potential import *

plot_path = "plots/tests/simulation"
animation_path = os.path.join(plot_path, "animation")

if not os.path.exists(plot_path):
    os.mkdir(plot_path)

# Tests for the Particle class -- *not* a test particle!!!
class TestParticle(object):
    
    def test_creation(self):
        # Not quantities
        with pytest.raises(TypeError):
            Particle(15., 16., 1.*u.M_sun)
            
        with pytest.raises(TypeError):
            Particle(15.*u.kpc, 16., 1.*u.M_sun)
        
        with pytest.raises(TypeError):
            Particle(15., 16.*u.kpc/u.Myr, 1.*u.M_sun)
        
        # make 1D case vectors
        p = Particle(15.*u.kpc, 16.*u.kpc/u.Myr, m=1.*u.M_sun)
        assert len(p.r) == 1 and len(p.v) == 1
        
        # 2D case
        p = Particle([15.,10.]*u.kpc, [160.,110.]*u.kpc/u.Myr, m=1.*u.M_sun)
        assert len(p.r) == 2 and len(p.v) == 2 and len(p.m) == 1
        
        # 3D case
        p = Particle([0.,8.,0.2]*u.kpc, 
                     [10.,200.,-15.]*u.km/u.s,
                     m=1.*u.M_sun)
        
        with pytest.raises(ValueError):
            p = Particle([[15.,11.,13],[15.,11.,13]]*u.kpc, 
                     [[15.,11.,13],[15.,11.,13]]*u.kpc/u.Myr,
                     m=[1.,1.,1.]*u.M_sun)
        
        # 3D, multiple particles
        p = Particle([[15.,11.,13],[15.,11.,13]]*u.kpc, 
                     [[15.,11.,13],[15.,11.,13]]*u.kpc/u.Myr,
                     m=[1.,1.]*u.M_sun)
        
        pc = Particle(r=np.random.random(size=500)*u.kpc,
                      v=np.random.random(size=500)*u.kpc,
                      m=np.ones(500)*u.M_sun)
        
        
        pc = Particle(r=np.random.random(size=(500,3))*u.kpc,
                      v=np.random.random(size=(500,3))*u.kpc,
                      m=np.ones(500)*u.M_sun)
        
        # Size mismatch
        with pytest.raises(ValueError):
            pc = Particle(r=np.random.random(size=(500,3))*u.kpc,
                          v=np.random.random(size=(501,3))*u.kpc,
                          m=np.ones(500)*u.M_sun)
        
        with pytest.raises(ValueError):
            pc = Particle(r=np.random.random(size=(501,3))*u.kpc,
                          v=np.random.random(size=(501,3))*u.kpc,
                          m=np.ones(500)*u.M_sun)
    
    def test_slicing(self):
        pc = Particle(r=np.random.random(size=500)*u.kpc,
                      v=np.random.random(size=500)*u.km/u.s,
                      m=np.ones(500)*u.M_sun)
        
        assert isinstance(pc[0], Particle)
        
        assert isinstance(pc[0:15], Particle)
        assert len(pc[0:15]) == 15
    
    def test_integrate(self):
        
        potential = LawMajewski2010()
        
        t = np.linspace(0., 5000., 2000)*u.Myr
        pc = Particle(r=[8.,0.,0.]*u.kpc,
                      v=[0.,220.,0.]*u.km/u.s,
                      m=1*u.M_sun)
        pc.integrate(potential, t=t)
    
    def test_animate_earth(self):
        
        this_path = os.path.join(animation_path, "earth")
        if not os.path.exists(this_path):
            os.mkdir(this_path)
        
        #potential = LawMajewski2010()
        potential = PointMassPotential([u.au, u.yr, u.M_sun],
                                       m=1.*u.M_sun)
        
        t = np.linspace(0., 5., 250)*u.yr
        
        rs = [1.,0.,0.]*u.au
        vs = [0, 2*np.pi, -0.1]*u.au/u.yr
        ms = 3E-6*u.M_sun
        pc = Particle(r=rs,
                      v=vs,
                      m=ms)
        t,r,v = pc.integrate(potential, t=t)
        
        grid = np.linspace(-2,2,50)*u.au
        fig,axes = potential.plot(grid,grid,grid)
        for ii in range(len(t)):
            cxy = axes[0,0].scatter(r[ii,:,0], r[ii,:,1], c='k', alpha=0.5, s=10)
            cxz = axes[1,0].scatter(r[ii,:,0], r[ii,:,2], c='k', alpha=0.5, s=10)
            cyz = axes[1,1].scatter(r[ii,:,1], r[ii,:,2], c='k', alpha=0.5, s=10)
            
            fig.savefig(os.path.join(this_path, "{0}.png".format(ii)))
    
    def test_animate_galaxy(self):
        
        this_path = os.path.join(animation_path, "galaxy")
        if not os.path.exists(this_path):
            os.mkdir(this_path)
        
        bulge = HernquistPotential(unit_bases=[u.M_sun, u.kpc, u.Myr],
                                           m=1E10*u.M_sun, 
                                           c=0.7*u.kpc)
        disk = MiyamotoNagaiPotential(unit_bases=[u.M_sun, u.kpc, u.Myr],
                                               m=1E11*u.M_sun, 
                                               a=6.5*u.kpc, 
                                               b=0.26*u.kpc)
        
        halo = LogarithmicPotentialLJ(unit_bases=[u.kpc,u.Myr,u.M_sun,u.radian],
                                               v_halo=(121.858*u.km/u.s).to(u.kpc/u.Myr),
                                               q1=1.38,
                                               q2=1.0,
                                               qz=1.36,
                                               phi=1.692969*u.radian,
                                               r_halo=12.*u.kpc)
        
        potential = disk + bulge + halo
        
        t = np.linspace(0., 1000., 500)*u.Myr
        
        N = 100
        trails = True
        #rs = np.array([14.0, 0.0, 5.]) *u.kpc
        #vs = np.array([40., 130., 30.]) * u.km/u.s
        
        r = np.sqrt(np.random.uniform(size=(N,1))*10.**2)
        phi = np.random.uniform(size=(N,1))*2.*np.pi
        
        x = r*np.cos(phi)
        y = r*np.sin(phi)
        z = np.random.uniform(-0.2, 0.2, size=(N,1))
        rs = np.hstack((x,y,z))*u.kpc
        
        mag_v = 150.
        vx = -mag_v * np.sin(phi)# + (Math.random.uniform(N)-0.5)*(dispersion/100.0/1.41),
        vy = mag_v * np.cos(phi)# + (np.random.uniform(N)-0.5)*(dispersion/100.0/1.41);
        vz = np.random.uniform(-10., 10., size=(N,1))
        vs = np.hstack((vx,vy,vz))*u.km/u.s
        ms = np.ones(N)*u.M_sun
        
        pc = Particle(r=rs,
                      v=vs,
                      m=ms)
        t,r,v = pc.integrate(potential, t=t)
        
        grid = np.linspace(-10,10,50)*u.kpc
        fig,axes = potential.plot(grid,grid,grid)
        for ii in range(len(t)):
            if trails:
                cxy = axes[0,0].scatter(r[ii,:,0], r[ii,:,1], c='k', alpha=0.75, s=8, edgecolor="none")
                cxz = axes[1,0].scatter(r[ii,:,0], r[ii,:,2], c='k', alpha=0.75, s=8, edgecolor="none")
                cyz = axes[1,1].scatter(r[ii,:,1], r[ii,:,2], c='k', alpha=0.75, s=8, edgecolor="none")
            else:
                try:
                    cxy.set_offsets((r[ii,:,0], r[ii,:,1]))
                    cxz.set_offsets((r[ii,:,0], r[ii,:,2]))
                    cyz.set_offsets((r[ii,:,1], r[ii,:,2]))
                except NameError:
                    cxy = axes[0,0].scatter(r[ii,:,0], r[ii,:,1], c='k', alpha=0.75, s=8, edgecolor="none")
                    cxz = axes[1,0].scatter(r[ii,:,0], r[ii,:,2], c='k', alpha=0.75, s=8, edgecolor="none")
                    cyz = axes[1,1].scatter(r[ii,:,1], r[ii,:,2], c='k', alpha=0.75, s=8, edgecolor="none")
            
            #print(potential.acceleration_at(r[ii,:,:]))
            if ii % 10 == 0:
                fig.savefig(os.path.join(this_path, "{0}.png".format(ii)))

def leapfrog_hack(potential, satellite_potential, initial_position, initial_velocity, 
                  t=None, t1=None, t2=None, dt=None):
             
    if initial_position.shape != initial_velocity.shape:
        raise ValueError("initial_position shape must match initial_velocity "
                         "shape! {0} != {1}"
                         .format(initial_position.shape, 
                                 initial_velocity.shape))

    if initial_position.ndim == 1:
        # r_i just stands for positions, it's actually a vector
        r_i = np.array(initial_position)\
                .reshape(1, len(initial_position))
        v_i = np.array(initial_velocity)\
                .reshape(1, len(initial_position))
    else:
        r_i = initial_position
        v_i = initial_velocity
    
    if t == None:           
        times = np.arange(t1, t2+dt, dt)
        #times = np.arange(t1, t2, dt)
    else:
        times = t
        dt = times[1]-times[0]
    
    Ntimesteps = len(times)

    # Shape of final object should be (Ntimesteps, Ndim, Nparticles)
    rs = np.zeros((Ntimesteps,) + r_i.shape, dtype=np.float64)
    vs = np.zeros((Ntimesteps,) + v_i.shape, dtype=np.float64)
    
    initial_sat_velocity = ([0., 100., 50.]*u.km/u.s).to(u.kpc/u.Myr).value
    r_i_sat = np.array(satellite_potential.parameters.values()[0]["origin"])\
            .reshape(1, 3)
    v_i_sat = np.array(initial_sat_velocity)\
            .reshape(1, 3)
    rs_sat = np.zeros((Ntimesteps,) + r_i_sat.shape, dtype=np.float64)
    vs_sat = np.zeros((Ntimesteps,) + v_i_sat.shape, dtype=np.float64)
    
    axes = None
    grid = np.linspace(-80,80,100)*u.kpc
    for ii in range(Ntimesteps):
        t = times[ii]
        
        # particles
        a_i = (potential+satellite_potential).acceleration_at(r_i).T
        
        r_ip1 = r_i + v_i*dt + 0.5*a_i*dt*dt        
        a_ip1 = (potential+satellite_potential).acceleration_at(r_ip1).T
        v_ip1 = v_i + 0.5*(a_i + a_ip1)*dt

        rs[ii,:,:] = r_i
        vs[ii,:,:] = v_i

        a_i = a_ip1
        r_i = r_ip1
        v_i = v_ip1
        
        # satellite
        a_i_sat = potential.acceleration_at(r_i_sat).T
        
        r_ip1_sat = r_i_sat + v_i_sat*dt + 0.5*a_i_sat*dt*dt        
        a_ip1_sat = potential.acceleration_at(r_ip1_sat).T
        v_ip1_sat = v_i_sat + 0.5*(a_i_sat + a_ip1_sat)*dt

        rs_sat[ii,:,:] = r_i_sat
        vs_sat[ii,:,:] = v_i_sat

        a_i_sat = a_ip1_sat
        r_i_sat = r_ip1_sat
        v_i_sat = v_ip1_sat
        
        satellite_potential.parameters.values()[0]["origin"] = r_i_sat[0]
        
        if axes == None:
            fig,axes = (potential+satellite_potential).plot(grid,grid,grid)
        else:
            for ax in axes.flat: 
                ax.cla()
            fig,axes = (potential+satellite_potential).plot(grid,grid,grid,axes=axes)
            
        axes[0,0].scatter(rs[ii,:,0], rs[ii,:,1], c='#F1A340', s=6, alpha=0.7, edgecolor='none')
        axes[1,0].scatter(rs[ii,:,0], rs[ii,:,2], c='#F1A340', s=6, alpha=0.7, edgecolor='none')
        axes[1,1].scatter(rs[ii,:,1], rs[ii,:,2], c='#F1A340', s=6, alpha=0.7, edgecolor='none')
        fig.savefig(os.path.join(animation_path, "{0:04d}.png".format(ii)), facecolor="k")
        
        if ii % 50 == 0:
            print(ii)
        
    return times, (rs, vs), (rs_sat, vs_sat)

def test_mw_plus_sat():
    bulge = HernquistPotential(unit_bases=[u.M_sun, u.kpc, u.Myr],
                                       m=1E10*u.M_sun, 
                                       c=0.7*u.kpc)
    disk = MiyamotoNagaiPotential(unit_bases=[u.M_sun, u.kpc, u.Myr],
                                           m=1E11*u.M_sun, 
                                           a=6.5*u.kpc, 
                                           b=0.26*u.kpc)
    
    halo = LogarithmicPotentialLJ(unit_bases=[u.kpc,u.Myr,u.M_sun,u.radian],
                                           v_halo=(121.858*u.km/u.s).to(u.kpc/u.Myr),
                                           q1=1.38,
                                           q2=1.0,
                                           qz=1.36,
                                           phi=1.692969*u.radian,
                                           r_halo=12.*u.kpc)
    
    mw_potential = disk + bulge + halo

    sat_potential = HernquistPotential(unit_bases=[u.M_sun, u.kpc, u.Myr],
                                       m=2.5E9*u.M_sun, 
                                       c=1.*u.kpc,
                                       origin=[40.,0.,0.]*u.kpc)
    
    N = 1000
    
    r = np.sqrt(np.random.uniform(size=(N,1))*4.**2)
    phi = np.random.uniform(size=(N,1))*2.*np.pi
    
    x = r*np.cos(phi) + 40.
    y = r*np.sin(phi)
    z = np.random.uniform(-0.2, 0.2, size=(N,1))
    initial_position = np.hstack((x,y,z))
    
    mag_v = 40.
    vx = -mag_v * np.sin(phi) + 0.
    vy = mag_v * np.cos(phi) + 100.
    vz = np.random.uniform(-5., 5., size=(N,1)) + 50.
    initial_velocity = (np.hstack((vx,vy,vz))*u.km/u.s).to(u.kpc/u.Myr).value
    
    t, (rs,vs), (rs_sat,vs_sat) = leapfrog_hack(mw_potential, sat_potential, 
                  initial_position, initial_velocity, 
                  t1=0., t2=6000., dt=5.)
    
    return
    
    fig, axes = plt.subplots(3, 1, sharex=True, sharey=True)
    ts, xs, vs = leapfrog(potential.acceleration_at, 
                          initial_position, 
                          initial_velocity, 
                          t1=0., t2=6000., dt=1.)
    plot_energies(potential,ts, xs, vs)
    fig1,fig2 = plot_energies(potential,ts, xs, vs)
    fig1.savefig(os.path.join(plot_path,"three_component_energy.png"))
