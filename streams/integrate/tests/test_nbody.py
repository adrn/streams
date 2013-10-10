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

from .. import *
from ..nbody import *
from ...potential.lm10 import LawMajewski2010, true_params
from ...dynamics import Particle, Orbit

plot_path = "plots/tests/integrate/nbody"
if not os.path.exists(plot_path):
    os.makedirs(plot_path)

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
    
    sun = Particle(r=np.array([0.,0.,0.])*u.au,
                   v=np.array([0.,0.,0.])*u.au/u.yr,
                     m=1.*u.M_sun)
    pc = sun.merge(earth)
    
    # Create time grid to integrate on
    t = np.arange(0., 100., 0.02) * u.yr
    r,v = nbody_integrate(pc, time_steps=t, e=0.1)
    
    plt.figure(figsize=(8,8))
    plt.plot(r[:,0,0], r[:,0,1], 'b-')
    plt.plot(r[:,1,0], r[:,1,1], 'b-')
    plt.savefig(os.path.join(plot_path, "earth_sun.png"))

def test_collection():
    ffmpeg_cmd = "ffmpeg -i {0} -r 12 -b 5000 -vcodec libx264 -vpre medium -b 3000k {1}"
        
    N = 10000
    usys = (u.kpc, u.Myr, u.M_sun, u.radian)
    
    # Create particles
    r = np.zeros((N,3))
    v = np.zeros((N,3))
    for ii in range(N):
        R = np.sqrt(np.random.uniform())*9. + 1.
        theta = np.random.uniform(0., 2*np.pi)
        r[ii] = R*np.array([np.cos(theta), np.sin(theta), 0.])
        
        V = 220.
        v[ii] = V * np.array([-np.sin(theta), np.cos(theta), 0.])
    
    pc = Particle(r=r*u.kpc, v=v*u.km/u.s, 
                  m=1E5*np.random.uniform(0.05, 10.)*u.M_sun,
                  units=usys)
    
    # Create time grid to integrate on
    t = np.arange(0., 1000., 1.) * u.Myr

    import time
    a = time.time()
    r,v = nbody_integrate(pc, time_steps=t, e=0.01, 
                          external_acceleration=None)
    print("Took {0} seconds for integration".format(time.time()-a))
    
    plt.figure(figsize=(10,10))
    for jj in range(r.shape[0]):
        plt.clf()
        plt.scatter(r[jj,:,0], r[jj,:,1], c='k', edgecolor='none', 
                    s=pc.m.M_sun+5., alpha=0.4)
        plt.xlim(-15, 15)
        plt.ylim(-15, 15)
        plt.savefig(os.path.join(this_path,"{0:04d}.png".format(jj)))
 
    print(ffmpeg_cmd.format(os.path.join(this_path, "%4d.png"), 
                            os.path.join(this_path, "anim.mp4")))
