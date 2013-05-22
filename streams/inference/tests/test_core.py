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

from streams.inference.lm10 import ln_likelihood
from streams.nbody import Particle, ParticleCollection

def cprofile_time():
    for ii in range(3):
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

if __name__ == "__main__":
    import cProfile
    import pstats
    
    cProfile.run("cprofile_time()", "/tmp/cprof")
    
    p = pstats.Stats("/tmp/cprof")
    p.sort_stats('cumulative').print_stats(50)