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
    
    this_path = os.path.join(plot_path, "disk")
    if not os.path.exists(this_path):
        os.mkdir(this_path)

    N = 10000
    mass_norm = 1E11/N
    usys = (u.kpc, u.Myr, u.M_sun, u.radian)
    
    # Create particles
    v = np.zeros((N,3))
    
    #R = np.sqrt(np.random.uniform(size=N))*9. + 1.
    R = (2.71/np.exp(np.random.uniform(size=N)) - 1.)*9. + 1
    thetas = np.random.uniform(0., 2*np.pi, size=N)

    r = np.zeros((N,3))
    r[:,0] = R*np.cos(thetas)
    r[:,1] = R*np.sin(thetas)
    
    #RR = np.repeat(R[:,np.newaxis], len(R), axis=1) <
    V = np.zeros_like(R)
    for ii in range(N):
        M = np.sum(R < R[ii])*mass_norm
        a = np.sqrt(G*M*u.M_sun/(R[ii]*u.kpc)).to(u.km/u.s).value * 0.9
        V[ii] = a

    v = np.zeros_like(r)
    v[:,0] = V * -np.sin(thetas)
    v[:,1] = V * np.cos(thetas)
    
    pc = Particle(r=r*u.kpc, v=v*u.km/u.s, 
                  m=mass_norm*np.random.uniform(0.05, 10.)*u.M_sun,
                  units=usys)

    # Create time grid to integrate on
    t = np.arange(0., 1000., 1.) * u.Myr
    #t = np.arange(0., 100., 1.) * u.Myr

    import time
    a = time.time()
    r,v = nbody_integrate(pc, time_steps=t, e=0.01, 
                          external_acceleration=None)
    print("Took {0} seconds for integration".format(time.time()-a))
    
    plt.figure(figsize=(10,10))
    for jj in range(r.shape[0]):
        plt.clf()
        plt.scatter(r[jj,:,0], r[jj,:,1], c='k', edgecolor='none', 
                    s=(pc.m.M_sun/mass_norm)+5., alpha=0.4)
        plt.xlim(-15, 15)
        plt.ylim(-15, 15)
        plt.savefig(os.path.join(this_path,"{0:04d}.png".format(jj)))
 
    print(ffmpeg_cmd.format(os.path.join(this_path, "%4d.png"), 
                            os.path.join(this_path, "anim.mp4")))
