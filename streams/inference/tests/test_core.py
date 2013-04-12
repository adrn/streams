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

try:
    from ..core import ln_likelihood
    from ...simulation import TestParticle, TestParticleOrbit
except ValueError:
    from streams.inference.core import ln_likelihood
    from streams.simulation import TestParticle, TestParticleOrbit
    
def cprofile_time():
    for ii in range(3):
        test_time_likelihood()

def test_time_likelihood():
    
    p = [1.2, 1.2]
    param_names = ["q1", "qz"]
    particles = TestParticle(r=np.random.uniform(size=(100,3))*u.kpc,
                             v=np.random.uniform(size=(100,3))*u.kpc/u.Myr)
    
    t = np.linspace(0., 10., 500)*u.Myr
    sat_orbit = TestParticleOrbit(t=t,
                                  r=np.random.uniform(size=(500,3))*u.kpc,
                                  v=np.random.uniform(size=(500,3))*u.kpc/u.Myr)
    
    print(ln_likelihood(p, param_names, particles, sat_orbit))
    

if __name__ == "__main__":
    import cProfile
    import pstats
    
    cProfile.run("cprofile_time()", "/tmp/cprof")
    
    p = pstats.Stats("/tmp/cprof")
    p.sort_stats('cumulative').print_stats(50)