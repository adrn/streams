# coding: utf-8
""" Test the speed of the Pal5 objective function """

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

from streams.inference.pal5 import ln_likelihood, ln_posterior
from streams.dynamics import ParticleCollection
from streams.potential.pal5 import Palomar5, true_params
from streams.io.pal5 import particles_today, satellite_today, time

plot_path = "plots/tests/inference"
if not os.path.exists(plot_path):
    os.makedirs(plot_path)

potential = Palomar5()

def test_cprofile_time():
    for ii in range(10):
        time_posterior_func()

def time_posterior_func():
    a = pytime.time()
    for ii in range(10):
        test_time_posterior()
    print((pytime.time()-a) / 10., "seconds per call")

np.random.seed(42)
t1,t2 = time()
satellite = satellite_today()
particles = particles_today(N=100)

def test_time_posterior():
    resolution = 4.
    ln_posterior([], [], particles, satellite, t1, t2, resolution)

if __name__ == "__main__":
    cProfile.run("test_cprofile_time()", os.path.join(plot_path, "cprofiled_pal5"))
    
    p = pstats.Stats(os.path.join(plot_path, "cprofiled_pal5"))
    p.sort_stats('cumulative').print_stats(50)
    
    test_time_posterior()
