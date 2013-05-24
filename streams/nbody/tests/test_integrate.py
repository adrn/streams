# coding: utf-8

from __future__ import absolute_import, unicode_literals, division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys

# Third-party
import numpy as np
import pytest
import astropy.units as u
from astropy.constants import G
import matplotlib.pyplot as plt

from ..core import *
from ..integrate import *
from ..potential.lm10 import LawMajewski2010

def test_acceleration():
    r = np.array([[1.,0.],
                  [0, 1.],
                  [-1., 0.],
                  [0., -1.]])
    m = np.array([1.,1.,1.,1.])
    _G = G.decompose(bases=[u.kpc,u.Myr,u.M_sun]).value
    
    a = _nbody_acceleration(_G, r, m)
    assert a.shape == r.shape
    
    print(a)

def test_api():
    
    # Create particles
    particles = []
    earth = Particle(r=np.array([1., 0., 0.])*u.au,
                     v=np.array([0., 2.*np.pi, 0.])*u.au/u.yr,
                     m=1.E-6*u.M_sun)
    particles.append(earth)
    
    sun = Particle(r=np.array([0.,0.,0.])*u.au,
                   v=np.array([0.,0.,0.])*u.au/u.yr,
                     m=1.*u.M_sun)
    particles.append(sun)
    
    pc = ParticleCollection(particles=particles, units=[u.au, u.yr, u.M_sun])
    
    # Create time grid to integrate on
    t = np.arange(0., 10., 0.02) * u.yr
    r,v = nbody_integrate(pc, time_steps=t, e=0.1)
    
    plt.figure(figsize=(8,8))
    plt.plot(r[:,0,0], r[:,0,1], 'b-')
    plt.plot(r[:,1,0], r[:,1,1], 'b-')
    plt.show()

def test_collection():
    ffmpeg_cmd = "ffmpeg -i {0} -r 12 -b 5000 -vcodec libx264 -vpre medium -b 3000k {1}"
    this_path = "plots/tests/nbody/disk"
    if not os.path.exists(this_path):
        os.makedirs(this_path)
        
    N = 10
    
    potential = LawMajewski2010()
    
    # Create particles
    particles = []
    for ii in range(N):
        R = np.sqrt(np.random.uniform())*9. + 1.
        theta = np.random.uniform(0., 2*np.pi)
        r = R*np.array([np.cos(theta), np.sin(theta)])*u.kpc
        
        V = 200.
        v = V * np.array([-np.sin(theta), np.cos(theta)])*u.km/u.s
        
        p = Particle(r=r,
                     v=v,
                     m=np.random.uniform(0.05, 10.)*u.M_sun)
        particles.append(p)
    
    pc = ParticleCollection(particles=particles, units=[u.kpc, u.Myr, u.M_sun])
    
    # Create time grid to integrate on
    t = np.arange(0., 6000., 10.) * u.Myr
    r,v = nbody_integrate(pc, time_steps=t, e=0.1, 
                          external_acceleration=potential.acceleration_at)
    
    plt.figure(figsize=(10,10))
    for jj in range(r.shape[0]):
        plt.clf()
        plt.scatter(r[jj,:,0], r[jj,:,1], c='k', edgecolor='none', 
                    s=pc.m.M_sun+5., alpha=0.4)
        plt.xlim(-15, 15)
        plt.ylim(-15, 15)
        plt.savefig(os.path.join(this_path,"{0:04d}.png".format(jj)))
 
    os.system(ffmpeg_cmd
              .format(os.path.join(this_path, "%4d.png"), 
                      os.path.join(this_path, "anim.mp4")))
    