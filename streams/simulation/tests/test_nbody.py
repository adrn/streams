# coding: utf-8
"""
    Test the nbody capabilities of the Particles
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
import matplotlib
matplotlib.use("WxAgg")
import matplotlib.pyplot as plt

from ..core import Particle
from ...potential import *

plot_path = "plots/tests/simulation"
animation_path = os.path.join(plot_path, "animation")
ffmpeg_cmd = "ffmpeg -i {0} -r 12 -b 5000 -vcodec libx264 -vpre medium -b 3000k {1}"

if not os.path.exists(plot_path):
    os.mkdir(plot_path)

def test_setup_particles():
    this_path = os.path.join(plot_path, "few_particles")
    if not os.path.exists(this_path):
        os.mkdir(this_path)
    else:
        for f in glob.glob(os.path.join(this_path, "*")):
            os.remove(f)
    
    units = {"length" : u.au,
             "time" : u.yr, 
             "mass" : u.M_sun}
    
    # number of particles
    N = 4
    
    #r = np.random.random(size=(N,3))
    r = np.array([[0.25,0.25,0.],
                  [0.25,0.75,0.],
                  [0.75,0.25,0.],
                  [0.75,0.75,0.]])
    v = np.random.random(size=(N,3))
    r[:,2] = np.zeros(N)
    v[:,2] = np.zeros(N)
    pc = Particle(r=r*units["length"],
                  v=v*units["length"]/units["time"],
                  m=np.ones(N)*units["mass"])
    
    r = [0.5,0.5,0.]*units["length"]
    acc_x,acc_y,acc_z = pc.acceleration_at(r)
    
    fig,axes = pc.plot_positions()
    axes[1,0].quiver(r.value[0], r.value[1], acc_x, acc_y)
    axes[1,0].axvline(r.value[0])
    axes[1,0].axhline(r.value[1])
    fig.savefig(os.path.join(this_path,"test.png"))

def test_basic_nbody():
    
    def acceleration(positions, potential=None):
        for particle_idx in np.arange(particles.nparticles):
            idx = np.delete(np.arange(particles.nparticles), particle_idx)
            acc = particles[idx].acceleration_at(particles[particle_idx].r)
            
            #if potential != None:
            #    acc += potential.acceleration_at(particles[particle_idx].r)
    
    this_path = os.path.join(plot_path, "few_particles")
    if not os.path.exists(this_path):
        os.mkdir(this_path)
    else:
        for f in glob.glob(os.path.join(this_path, "*")):
            os.remove(f)
    
    units = {"length" : u.au,
             "time" : u.yr, 
             "mass" : u.M_sun}
    
    # number of particles
    N = 4
    
    #r = np.random.random(size=(N,3))
    r = np.array([[0.25,0.25,0.],
                  [0.25,0.75,0.],
                  [0.75,0.25,0.],
                  [0.75,0.75,0.]])
    v = np.random.random(size=(N,3))
    r[:,2] = np.zeros(N)
    v[:,2] = np.zeros(N)
    pc = Particle(r=r*units["length"],
                  v=v*units["length"]/units["time"],
                  m=np.ones(N)*units["mass"])
    
    r = [0.5,0.5,0.]*units["length"]
    acc_x,acc_y,acc_z = pc.acceleration_at(r)
    
    fig,axes = pc.plot_positions()
    axes[1,0].quiver(r.value[0], r.value[1], acc_x, acc_y)
    axes[1,0].axvline(r.value[0])
    axes[1,0].axhline(r.value[1])
    fig.savefig(os.path.join(this_path,"test.png"))