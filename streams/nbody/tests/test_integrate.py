# coding: utf-8

from __future__ import absolute_import, unicode_literals, division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys

# Third-party
import numpy as np
import pytest

from ..core import *
from ..integrate import *

def test_api():
    
    # Create particles
    N = 100
    particles = []
    for ii in range(N):
        p = Particle(r=np.array([0.08*ii+1.,0.,0.])*u.kpc,
                     v=np.array([0., 220., 0.])*u.km/u.s,
                     m=1.*u.M_sun)
        particles.append(p)
    pc = ParticleCollection(particles=particles)
    
    # Create time grid to integrate on
    t = np.arange(0., 6000, 1.) * u.Myr
    
    orbits = nbody_integrate(pc, time_steps=t, merge_length=1.*u.pc)