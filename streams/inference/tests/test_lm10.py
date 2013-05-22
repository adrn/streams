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
    
    t = np.arange(0., 6000., 5.)*u.Myr
    
    print(ln_likelihood(p, param_names, particles, satellite, t))

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

def test_likelihood():
    # TODO: make sure likelihood for correct parameters is higher
    
    pass
    
if __name__ == "__main__":
    import cProfile
    import pstats
    
    cProfile.run("test_cprofile_time()", "/tmp/cprof")
    
    p = pstats.Stats("/tmp/cprof")
    p.sort_stats('cumulative').print_stats(50)