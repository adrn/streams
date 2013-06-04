# coding: utf-8
""" """

from __future__ import absolute_import, unicode_literals, division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys
import time

# Third-party
import numpy as np
import pytest
import astropy.units as u
import matplotlib.pyplot as plt
from matplotlib import cm

from streams.inference.lm10 import ln_likelihood
from streams.nbody import Particle, ParticleCollection
from streams.integrate import leapfrog
from streams.potential.lm10 import LawMajewski2010
from streams.data.sgr import lm10_particles, lm10_satellite, lm10_time

plot_path = "plots/tests/inference"
if not os.path.exists(plot_path):
    os.makedirs(plot_path)

potential = LawMajewski2010()

def test_cprofile_time():
    for ii in range(10):
        test_time_likelihood()

def test_time_likelihood():
    
    p = [1.2, 1.2, 0.121, 1.6912]
    param_names = ["q1", "qz", "v_halo", "phi"]
    particles = ParticleCollection(r=np.random.uniform(size=(100,3))*u.kpc,
                                   v=np.random.uniform(size=(100,3))*u.kpc/u.Myr,
                                   m=np.zeros(100)*u.M_sun,
                                   units=[u.kpc,u.Myr,u.M_sun])
    
    satellite = ParticleCollection(r=np.random.uniform(size=(1,3))*u.kpc,
                                   v=np.random.uniform(size=(1,3))*u.kpc/u.Myr,
                                   m=2.5E8*u.M_sun,
                                   units=[u.kpc,u.Myr,u.M_sun])
    
    resolution = 3.
    t1,t2 = lm10_time()
    
    print(ln_likelihood(p, param_names, particles, satellite, t1, t2, resolution))

def test_energy_conserve():
    N = 100
    r = np.tile([5.,0.,0.], N).reshape(N,3) \
            * np.random.uniform(0.5, 1.5, size=(N,1)) \
            * u.kpc
    
    v = np.tile([0.,200.,0.], N).reshape(N,3) \
            * np.random.uniform(0.5, 1.5, size=(N,1)) \
            * u.km/u.s
    
    particles = ParticleCollection(r=r,
                                   v=v,
                                   m=np.zeros(N)*u.M_sun,
                                   units=[u.kpc,u.Myr,u.M_sun])
    
    t = np.arange(0., 6000., 1.)*u.Myr
    
    tt,rr,vv = leapfrog(potential._acceleration_at, 
                        particles._r, particles._v, t=t.value)
    
    fig,ax = plt.subplots(1,1,figsize=(12,12))
    ax.axis('off')
    #ax.scatter(r[:,0], r[:,1], marker='.', color='k', edgecolor='none')
    
    R = np.sum(rr[0]**2, axis=-1)
    scale = lambda R2: (R2 - R.min()) / (R.max()-R.min())
    
    cmap = cm.get_cmap('RdYlBu')
    for ii in range(N):
        this_R = rr[0,ii,0]**2 + rr[0,ii,2]**2
        ax.plot(rr[:,ii,0], rr[:,ii,1], color=cmap(scale(this_R)), alpha=0.07, lw=2.)
        
    ax.set_xlim(-15,15)
    ax.set_ylim(-15,15)
    fig.savefig(os.path.join(plot_path, "particles_one_orbit.png"),
                facecolor="#444444")

def test_compare_likelihood():
    satellite = lm10_satellite()
    particles = lm10_particles(N=100)
    t1,t2 = lm10_time()
    
    p = [1.2, 1.2, 0.121, 1.6912]
    param_names = ["q1", "qz", "v_halo", "phi"]
    a = time.time()
    l1 = ln_likelihood(p, param_names, particles, satellite, t1=t1, t2=t2, resolution=3.)
    print("l1: {0} ({1} seconds)".format(l1, time.time()-a))
    
    p = [1.38, 1.36, (121.858*u.km/u.s), 1.692969*u.radian]
    param_names = ["q1", "qz", "v_halo", "phi"]
    a = time.time()
    l2 = ln_likelihood(p, param_names, particles, satellite, t1=t1, t2=t2, resolution=3.)
    print("l2: {0} ({1} seconds)".format(l2, time.time()-a))
    
    assert l2 > l1

def test_timestep_energy():
    satellite = lm10_satellite()
    particles = lm10_particles(N=1000)
    t1,t2 = lm10_time()
    
    plt.figure(figsize=(12,12))
    c = 'krgb'
    for ii,dt in enumerate([1., 5., 10., 20.]):
        t = np.arange(t1, t2, -dt)
        tt,rr,vv = leapfrog(potential._acceleration_at, 
                            satellite._r, satellite._v, t=t)
        
        pot_energy = []
        for r in rr:
            pot_energy.append(potential._value_at(r))
        pot_energy = np.array(pot_energy)
        kin_energy = 0.5*np.sum(vv**2, axis=-1)
        
        tot_E = pot_energy+kin_energy
        delta_E = (tot_E[1:] - tot_E[0]) / tot_E[0]
        
        plt.subplot(211)
        plt.plot(tt[1:], delta_E, c=c[ii], label="dt={0}".format(dt))
        
        R = np.sqrt(np.sum(rr**2, axis=-1))
        plt.subplot(212)
        plt.plot(tt, R, c=c[ii])
    
    plt.subplot(211)
    plt.legend()
    plt.ylabel("$\Delta E/E$")
    
    plt.subplot(212)
    plt.xlabel("Time [Myr]")
    plt.ylabel("Sgr Orbital Radius $R$")
    
    plt.tight_layout()
    plt.savefig(os.path.join(plot_path,"timestep_energy_conserv.png"))
    
if __name__ == "__main__":
    import cProfile
    import pstats
    
    cProfile.run("test_cprofile_time()", "/tmp/cprof")
    
    p = pstats.Stats("/tmp/cprof")
    p.sort_stats('cumulative').print_stats(50)