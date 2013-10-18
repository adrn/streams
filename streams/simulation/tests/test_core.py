# coding: utf-8
"""
    Test the core simulation code, e.g. Particle
"""

from __future__ import absolute_import, unicode_literals, division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys
import glob

# Third-party
import numpy as np
import pytest
import astropy.units as u
import matplotlib.pyplot as plt

from ..core import TestParticle, TestParticleOrbit
from ...potential import *

plot_path = "plots/tests/simulation"
animation_path = os.path.join(plot_path, "animation")
ffmpeg_cmd = "ffmpeg -i {0} -r 12 -b 5000 -vcodec libx264 -vpre medium -b 3000k {1}"

if not os.path.exists(plot_path):
    os.mkdir(plot_path)

# Tests for the TestParticle class -- *not* a test particle!!!
class TestTestParticle(object):
    
    def test_creation(self):
        # Not quantities
        with pytest.raises(TypeError):
            TestParticle(15., 16.)
            
        with pytest.raises(TypeError):
            TestParticle(15.*u.kpc, 16.)
        
        with pytest.raises(TypeError):
            TestParticle(15., 16.*u.kpc/u.Myr)
        
        # make 1D case vectors
        p = TestParticle(15.*u.kpc, 16.*u.kpc/u.Myr)
        assert len(p.r) == 1 and len(p.v) == 1
        
        # 2D case
        p = TestParticle([15.,10.]*u.kpc, [160.,110.]*u.kpc/u.Myr)
        assert len(p.r) == 2 and len(p.v) == 2
        
        # 3D case
        p = TestParticle([0.,8.,0.2]*u.kpc, 
                     [10.,200.,-15.]*u.km/u.s)
                        
        pc = TestParticle(r=np.random.random(size=500)*u.kpc,
                      v=np.random.random(size=500)*u.kpc)
        
        
        pc = TestParticle(r=np.random.random(size=(500,3))*u.kpc,
                      v=np.random.random(size=(500,3))*u.kpc)
        
        # Size mismatch
        with pytest.raises(ValueError):
            pc = TestParticle(r=np.random.random(size=(500,3))*u.kpc,
                          v=np.random.random(size=(501,3))*u.kpc)
        
        with pytest.raises(ValueError):
            pc = TestParticle(r=np.random.random(size=(501,3))*u.kpc,
                          v=np.random.random(size=(500,3))*u.kpc)
    
    def test_slicing(self):
        pc = TestParticle(r=np.random.random(size=500)*u.kpc,
                      v=np.random.random(size=500)*u.km/u.s)
        
        assert isinstance(pc[0], TestParticle)
        
        assert isinstance(pc[0:15], TestParticle)
        assert len(pc[0:15]) == 15
    
    def test_array_slicing(self):
        pc = TestParticle(r=np.random.random(size=(500,3))*u.kpc,
                      v=np.random.random(size=(500,3))*u.km/u.s)
        
        assert isinstance(pc[0], TestParticle)
        
        assert isinstance(pc[0:15], TestParticle)
        assert len(pc[0:15]) == 15
    
    @pytest.mark.xfail
    def test_acceleration_from(self):        
        N = 500
        pc = TestParticle(r=np.random.random(size=(N,3))*u.kpc,
                      v=np.random.random(size=(N,3))*u.km/u.s)
        
        for ii in range(N):
            idx = np.delete(np.arange(N),ii)
            print(pc[idx].acceleration_at(pc[ii].r))


class TestTestParticleOrbit(object):
    
    def test_create(self):
        r = np.random.random(size=(1000,100,3))
        t = np.arange(0.,1000.)
        orbit1 = TestParticleOrbit(t*u.Myr,
                                   r=r*u.kpc,
                                   v=r*u.km/u.s)
   
    def test_slicing(self):
        r = np.random.random(size=(1000,100,3))
        t = np.arange(0.,1000.)
        orbit1 = TestParticleOrbit(t*u.Myr,
                                   r=r*u.kpc,
                                   v=r*u.km/u.s)
        
        assert isinstance(orbit1[15], TestParticle)
   
    def test_psd(self):
        N = 2
        t = np.arange(0.,1000)
        r = np.random.random(size=(1000,N,3))
        v = np.random.random(size=(1000,N,3))
        particle_orbits = TestParticleOrbit(t*u.Myr,
                                   r=r*u.kpc,
                                   v=v*u.km/u.s)
        
        t = np.arange(0.,1000.)
        r = np.random.random(size=(1000,3))
        v = np.random.random(size=(1000,3))
        satellite_orbit = TestParticleOrbit(t*u.Myr,
                                   r=r*u.kpc,
                                   v=r*u.km/u.s)
        
        r_tide = np.random.uniform(size=(1000,))*u.kpc
        v_esc = np.random.uniform(size=(1000,))*u.km/u.s
        
        from ...simulation import relative_normalized_coordinates, phase_space_distance
        
        R,V = relative_normalized_coordinates(particle_orbits, satellite_orbit, 
                                              r_tide, v_esc)
        psd = phase_space_distance(R,V)
        min_time_idx = psd.argmin(axis=0)
        assert len(min_time_idx) == N
        
        _min = psd.min(axis=0)
        for jj,ii in zip(min_time_idx, range(N)):
            assert _min[ii] == np.sqrt(np.sum(R[jj,ii,:].value**2) + np.sum(V[jj,ii,:].value**2))
    
    def test_min_psd(self):
        N = 100
        
        gal_units = [u.kpc, u.Myr, u.M_sun, u.radian]
        potential = CompositePotential(units=gal_units, 
                                       origin=[0.,0.,0.]*u.kpc)
        potential["disk"] = MiyamotoNagaiPotential(gal_units,
                                      m=1.E11*u.M_sun, 
                                      a=6.5*u.kpc,
                                      b=0.26*u.kpc,
                                      origin=[0.,0.,0.]*u.kpc)
        
        potential["bulge"] = HernquistPotential(gal_units,
                                   m=3.4E10*u.M_sun,
                                   c=0.7*u.kpc,
                                   origin=[0.,0.,0.]*u.kpc)
        
        potential["halo"] = LogarithmicPotentialLJ(gal_units,
                                      v_halo=(121.858*u.km/u.s),
                                      q1=1.38,
                                      q2=1.0,
                                      qz=1.36,
                                      phi=1.692969*u.radian,
                                      r_halo=12.*u.kpc,
                                      origin=[0.,0.,0.]*u.kpc)
        
        t = np.arange(0.,1000)
        r = np.random.random(size=(1000,N,3))
        v = np.random.random(size=(1000,N,3))
        particle_orbits = TestParticleOrbit(t*u.Myr,
                                   r=r*u.kpc,
                                   v=v*u.km/u.s)
        
        t = np.arange(0.,1000.)
        r = np.random.random(size=(1000,3))
        v = np.random.random(size=(1000,3))
        satellite_orbit = TestParticleOrbit(t*u.Myr,
                                   r=r*u.kpc,
                                   v=r*u.km/u.s)
        
        from ...simulation import minimum_distance_matrix, generalized_variance
        
        min_ps = minimum_distance_matrix(potential, particle_orbits, satellite_orbit)
        assert min_ps.shape == (N, 6)
        var = generalized_variance(potential, particle_orbits, satellite_orbit)
        print(var)
        
        
def test_animate_earth():
    
    this_path = os.path.join(animation_path, "earth")
    if not os.path.exists(this_path):
        os.mkdir(this_path)
    
    potential = PointMassPotential(units=[u.au, u.yr, u.M_sun],
                                   m=1.*u.M_sun)
    
    t = np.linspace(0., 5., 250)*u.yr
    
    rs = [1.,0.,0.]*u.au
    vs = [0, 2*np.pi, -0.1]*u.au/u.yr
    ms = 3E-6*u.M_sun
    pc = TestParticle(r=rs,
                  v=vs)
    t,r,v = pc.integrate(potential, t=t)
    
    grid = np.linspace(-2,2,50)*u.au
    fig,axes = potential.plot(grid,grid,grid)
    for ii in range(len(t)):
        cxy = axes[0,0].scatter(r[ii,:,0], r[ii,:,1], c='r', alpha=0.75, s=10, edgecolor='none')
        cxz = axes[1,0].scatter(r[ii,:,0], r[ii,:,2], c='r', alpha=0.75, s=10, edgecolor='none')
        cyz = axes[1,1].scatter(r[ii,:,1], r[ii,:,2], c='r', alpha=0.75, s=10, edgecolor='none')
        
        fig.savefig(os.path.join(this_path, "{0:03d}.png".format(ii)))
    
    os.system(ffmpeg_cmd
              .format(os.path.join(this_path, "%3d.png"), 
                      os.path.join(this_path, "anim.mp4")))
    
    for png in glob.glob(os.path.join(this_path,"*.png")):
        os.remove(png)

def test_animate_binary():
    
    this_path = os.path.join(animation_path, "binary")
    if not os.path.exists(this_path):
        os.mkdir(this_path)
    else:
        for f in glob.glob(os.path.join(this_path, "*")):
            os.remove(f)
    
    units = [u.au, u.yr, u.M_sun]
    potential = CompositePotential(units=units,
                                   origin=[0.,0.,0.]*u.au)
    potential["one"] = PointMassPotential(units=potential.units,
                                          m=1.*u.M_sun,
                                          origin=[0.,0.,0.]*u.au)
    potential["two"] = PointMassPotential(units=potential.units,
                                          m=0.8*u.M_sun,
                                          origin=[0.,2.,0.]*u.au)
    
    t = np.linspace(0., 10., 250)*u.yr
    
    rs = [1.,0.,0.]*u.au
    vs = [0, 2*np.pi, -1.5]*u.au/u.yr
    ms = 3E-6*u.M_sun
    pc = TestParticle(r=rs,
                  v=vs)
    t,r,v = pc.integrate(potential, t=t)
    
    grid = np.linspace(-4,4,100)*u.au
    fig,axes = potential.plot(grid,grid,grid)
    for ii in range(len(t)):
        cxy = axes[0,0].scatter(r[ii,:,0], r[ii,:,1], c='r', alpha=0.75, s=10, edgecolor='none')
        cxz = axes[1,0].scatter(r[ii,:,0], r[ii,:,2], c='r', alpha=0.75, s=10, edgecolor='none')
        cyz = axes[1,1].scatter(r[ii,:,1], r[ii,:,2], c='r', alpha=0.75, s=10, edgecolor='none')
        
        fig.savefig(os.path.join(this_path, "{0:03d}.png".format(ii)))
    
    out = os.system(ffmpeg_cmd
              .format(os.path.join(this_path, "%3d.png"), 
                      os.path.join(this_path, "anim.mp4")))
        
    for png in glob.glob(os.path.join(this_path,"*.png")):
        os.remove(png)

def test_animate_ternary():
    
    this_path = os.path.join(animation_path, "ternary")
    if not os.path.exists(this_path):
        os.mkdir(this_path)
    else:
        for f in glob.glob(os.path.join(this_path, "*")):
            os.remove(f)
    
    units = [u.au, u.yr, u.M_sun]
    potential = CompositePotential(units=units,
                                   origin=[0.,0.,0.]*u.au)
    potential["one"] = PointMassPotential(units=potential.units,
                                          m=1.*u.M_sun,
                                          origin=[0.,0.,0.]*u.au)
    potential["two"] = PointMassPotential(units=potential.units,
                                          m=0.1*u.M_sun,
                                          origin=[0.,4.,0.]*u.au)
    #potential["three"] = PointMassPotential(units=potential.units,
    #                                      m=0.1*u.M_sun,
    #                                      origin=[-1.,-3.,0.]*u.au)
    
    t = np.arange(0., 50., 0.05)*u.yr
    
    rs = [1.,0.,0.]*u.au
    vs = [0, 2.*np.pi, 0.5]*u.au/u.yr
    ms = 3E-6*u.M_sun
    pc = TestParticle(r=rs,
                  v=vs)
    t,r,v = pc.integrate(potential, t=t)
    
    alpha = 0.75
    color = '#29BFFF'
    s = 20
    trails = False
    
    grid = np.linspace(-5,5,100)*u.au
    fig,axes = potential.plot(grid,grid,grid)
    for ii in range(len(t)):
        if trails:
            cxy = axes[0,0].scatter(r[ii,:,0], r[ii,:,1], c=color, alpha=alpha, s=s, edgecolor="none")
            cxz = axes[1,0].scatter(r[ii,:,0], r[ii,:,2], c=color, alpha=alpha, s=s, edgecolor="none")
            cyz = axes[1,1].scatter(r[ii,:,1], r[ii,:,2], c=color, alpha=alpha, s=s, edgecolor="none")
        else:
            try:
                cxy.set_offsets(np.vstack((r[ii,:,0], r[ii,:,1])).T)
                cxz.set_offsets(np.vstack((r[ii,:,0], r[ii,:,2])).T)
                cyz.set_offsets(np.vstack((r[ii,:,1], r[ii,:,2])).T)
            except NameError:
                cxy = axes[0,0].scatter(r[ii,:,0], r[ii,:,1], c=color, alpha=alpha, s=s, edgecolor="none")
                cxz = axes[1,0].scatter(r[ii,:,0], r[ii,:,2], c=color, alpha=alpha, s=s, edgecolor="none")
                cyz = axes[1,1].scatter(r[ii,:,1], r[ii,:,2], c=color, alpha=alpha, s=s, edgecolor="none")
        
        fig.savefig(os.path.join(this_path, "{0:03d}.png".format(ii)))
    
    out = os.system(ffmpeg_cmd
              .format(os.path.join(this_path, "%3d.png"), 
                      os.path.join(this_path, "anim.mp4")))
        
    for png in glob.glob(os.path.join(this_path,"*.png")):
        os.remove(png)
    
def test_animate_galaxy():
    
    this_path = os.path.join(animation_path, "galaxy")
    if not os.path.exists(this_path):
        os.mkdir(this_path)
    else:
        for f in glob.glob(os.path.join(this_path, "*")):
            os.remove(f)
    
    units = [u.kpc, u.Myr, u.M_sun, u.radian]
    potential = CompositePotential(units=units, 
                                   origin=[0.,0.,0.]*u.kpc)
    potential["disk"] = MiyamotoNagaiPotential(potential.units,
                                  m=1.E11*u.M_sun, 
                                  a=6.5*u.kpc,
                                  b=0.26*u.kpc,
                                  origin=[0.,0.,0.]*u.kpc)
    
    potential["bulge"] = HernquistPotential(potential.units,
                               m=3.4E10*u.M_sun,
                               c=0.7*u.kpc,
                               origin=[0.,0.,0.]*u.kpc)
    
    potential["halo"] = LogarithmicPotentialLJ(potential.units,
                                  v_halo=(121.858*u.km/u.s),
                                  q1=1.38,
                                  q2=1.0,
                                  qz=1.36,
                                  phi=1.692969*u.radian,
                                  r_halo=12.*u.kpc,
                                  origin=[0.,0.,0.]*u.kpc)
                                  
    t = np.arange(0., 1000., 5.)*u.Myr
    
    N = 10000
    trails = False
    
    r = np.sqrt(np.random.uniform(0.02,1.,size=(N,1))*10.**2)
    phi = np.random.uniform(size=(N,1))*2.*np.pi
    
    x = r*np.cos(phi)
    y = r*np.sin(phi)
    z = np.random.normal(0.,0.2,size=(N,1)) #np.random.uniform(-0.2, 0.2, size=(N,1))
    rs = np.hstack((x,y,z))*u.kpc
    
    mag_v = np.random.normal(200., 7, size=(N,1))
    vx = mag_v * np.sin(phi)# + (Math.random.uniform(N)-0.5)*(dispersion/100.0/1.41),
    vy = -mag_v * np.cos(phi)# + (np.random.uniform(N)-0.5)*(dispersion/100.0/1.41);   
    vz = np.random.uniform(-10., 10., size=(N,1))*0.0
    vs = np.hstack((vx,vy,vz))*u.km/u.s
    vs = vs.to(u.kpc/u.Myr)# * 0.0
    ms = np.ones(N)*u.M_sun
    
    pc = TestParticle(r=rs,
                  v=vs)
    t,r,v = pc.integrate(potential, t=t)
    
    alpha = 0.5
    s = 3
    
    grid = np.linspace(-10,10,50)*u.kpc
    fig,axes = potential.plot(grid,grid,grid)
    for ii in range(len(t)):       
        if trails:
            cxy = axes[0,0].scatter(r[ii,:,0], r[ii,:,1], c='y', alpha=alpha, s=s, edgecolor="none")
            cxz = axes[1,0].scatter(r[ii,:,0], r[ii,:,2], c='y', alpha=alpha, s=s, edgecolor="none")
            cyz = axes[1,1].scatter(r[ii,:,1], r[ii,:,2], c='y', alpha=alpha, s=s, edgecolor="none")
        else:
            try:
                cxy.set_offsets(np.vstack((r[ii,:,0], r[ii,:,1])).T)
                cxz.set_offsets(np.vstack((r[ii,:,0], r[ii,:,2])).T)
                cyz.set_offsets(np.vstack((r[ii,:,1], r[ii,:,2])).T)
            except NameError:
                cxy = axes[0,0].scatter(r[ii,:,0], r[ii,:,1], c='y', alpha=alpha, s=s, edgecolor="none")
                cxz = axes[1,0].scatter(r[ii,:,0], r[ii,:,2], c='y', alpha=alpha, s=s, edgecolor="none")
                cyz = axes[1,1].scatter(r[ii,:,1], r[ii,:,2], c='y', alpha=alpha, s=s, edgecolor="none")
        
        axes[0,0].set_xlim(-10,10)
        axes[0,0].set_ylim(-10,10)
        axes[1,0].set_xlim(-10,10)
        axes[1,0].set_ylim(-10,10)
        axes[1,1].set_xlim(-10,10)
        axes[1,1].set_ylim(-10,10)
        fig.savefig(os.path.join(this_path, "{0:04d}.png".format(ii)))
    
    os.system(ffmpeg_cmd
              .format(os.path.join(this_path, "%4d.png"), 
                      os.path.join(this_path, "anim.mp4")))
    
    
    
    
    
    
# ==================================================================
    
    
    
    
    
    
    
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
    return
    
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

    sat_potential = MiyamotoNagaiPotential(unit_bases=[u.M_sun, u.kpc, u.Myr], 
                                           origin=[60.,0.,0.]*u.kpc,
                                           m=2.5E8*u.M_sun, 
                                           a=6.5*u.kpc, 
                                           b=0.26*u.kpc)
                                           
    #sat_potential = HernquistPotential(unit_bases=[u.M_sun, u.kpc, u.Myr],
    #                                   m=2.5E9*u.M_sun, 
    #                                   c=1.*u.kpc,
    #                                   origin=[40.,0.,0.]*u.kpc)
    
    N = 1000
    
    r = np.sqrt(np.random.uniform(size=(N,1))*5.**2)
    phi = np.random.uniform(size=(N,1))*2.*np.pi
    
    x = r*np.cos(phi) + 60.
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
