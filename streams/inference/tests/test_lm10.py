# coding: utf-8
""" """

from __future__ import absolute_import, unicode_literals, division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys
import time as pytime
import cProfile
import pstats

# Third-party
import numpy as np
import pytest
import astropy.units as u
import matplotlib.pyplot as plt
from matplotlib import cm

from streams.inference.lm10 import ln_likelihood
from streams.dynamics import ParticleCollection
from streams.potential.lm10 import LawMajewski2010, true_params
from streams.io.lm10 import particles_today, satellite_today, time

plot_path = "plots/tests/inference"
if not os.path.exists(plot_path):
    os.makedirs(plot_path)

potential = LawMajewski2010()

def test_cprofile_time():
    for ii in range(10):
        test_time_likelihood()

def time_likelihood_func():
    a = pytime.time()
    for ii in range(10):
        test_time_likelihood()
    print((pytime.time()-a) / 10., "seconds per call")

np.random.seed(42)
t1,t2 = time()
satellite = satellite_today()
particles = particles_today(N=100, expr="(Pcol > -1) & (abs(Lmflag)==1) & (dist<75)")

def test_time_likelihood():
    
    param_names = ["q1", "qz", "v_halo", "phi"]  
    
    resolution = 2.
    
    p = [1.2, 1.2, 0.121, 1.6912]
    ln_likelihood(p, param_names, particles, satellite, t1, t2, resolution)

if __name__ == "__main__":
    cProfile.run("test_cprofile_time()", os.path.join(plot_path, "cprofiled"))
    
    p = pstats.Stats(os.path.join(plot_path, "cprofiled"))
    p.sort_stats('cumulative').print_stats(50)
    
    time_likelihood_func()
