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

from ... import usys
from ..core import *

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


class TestParameter(object):

    def test_repr(self):
        q1 = Parameter(1.38, name="q1", range=(0.8,2.))
        repr(q1) == "<Parameter q1=1.38>"

    def test_prior(self):
        q1 = Parameter(1.38, range=(0.8,2.))
        assert q1.ln_prior() == 0.

        q1 = Parameter(2.5, range=(0.8,2.))
        assert np.isinf(q1.ln_prior())

        test_ln_prior = lambda v: 0. if 5.<v<10. else -np.inf
        q1 = Parameter(2.5, ln_prior=test_ln_prior)
        assert np.isinf(q1.ln_prior())

        q1 = Parameter(8.5, ln_prior=test_ln_prior)
        assert q1.ln_prior() == 0.

        q1 = Parameter(1.38, range=(0.8,2.))
        assert q1.ln_prior() == 0.
        q1.value = 99.
        assert np.isinf(q1.ln_prior())

    def test_sample(self):
        q1 = Parameter(1.38, range=(0.8,2.))
        assert 0.8 < q1.sample() < 2.

        s = q1.sample(size=1000)
        assert np.all((0.8 < s) & (s < 2))

        s = q1.sample(size=(1000,4))
        assert np.all((0.8 < s) & (s < 2))
        assert s.shape == (1000,4)

from streams.potential.lm10 import LawMajewski2010
from streams.io.sgr import mass_selector
from streams.observation.gaia import RRLyraeErrorModel

np.random.seed(552)
Nparticles = 100

particles_today, satellite_today, time = mass_selector("2.5e7")
satellite = satellite_today()
true_particles = particles_today(N=Nparticles, expr="(tub!=0)")
t1,t2 = time()

error_model = RRLyraeErrorModel(units=usys)
data_particles = true_particles.observe(error_model)

def test_statistical_model():
    stream = StreamModel(LawMajewski2010, satellite, data_particles)