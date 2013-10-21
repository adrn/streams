# coding: utf-8

from __future__ import absolute_import, unicode_literals, division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys
import time as pytime

# Third-party
import numpy as np
import pytest
import astropy.units as u
import matplotlib.pyplot as plt

from ..core import StreamModel

plot_path = "plots/tests/inference"
if not os.path.exists(plot_path):
    os.makedirs(plot_path)

'''
def test_statistical_model():
    """ Test with example from main Emcee docs """

    def lnprob(x, ivar=1.7):
        return -0.5 * np.sum(ivar * x ** 2)

    parameters = ['x']
    ndim, nwalkers = 1, 100
    ivar = 1. / np.random.rand(ndim)
    p0 = [np.random.rand(ndim) for i in range(nwalkers)]

    sm = StatisticalModel(parameters,
                          ln_likelihood=lnprob,
                          likelihood_args=(1.7,),
                          parameter_bounds={'x':(-1,100)})

    sampler = sm.run(p0, 100, 0)

    plt.hist(sampler.flatchain, bins=100)
    plt.savefig(os.path.join(plot_path, "emcee_test.png"))
'''

from streams.potential.lm10 import LawMajewski2010
from streams.io.sgr import mass_selector
from streams.observation.gaia import RRLyraeErrorModel

np.random.seed(552)
Nparticles = 100

particles_today, satellite_today, time = mass_selector("2.5e7")
satellite = satellite_today()
true_particles = particles_today(N=Nparticles, expr="(tub!=0)")
t1,t2 = time()

data_particles = true_particles.observe(GaiaErrorModel)

def test_statistical_model():
    stream = StreamModel(LawMajewski2010, satellite, data_particles)