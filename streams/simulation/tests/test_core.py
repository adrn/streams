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
