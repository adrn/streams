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

class TestStreamModel(object):

    def setup(self):
        from streams.potential.lm10 import LawMajewski2010
        from streams.io.sgr import mass_selector
        from streams.observation.gaia import RRLyraeErrorModel

        np.random.seed(552)
        self.Nparticles = 3

        particles_today, satellite_today, time = mass_selector("2.5e7")
        satellite = satellite_today()
        self.t1,self.t2 = time()

        self._particles = particles_today(N=self.Nparticles, expr="tub!=0")
        error_model = RRLyraeErrorModel(units=usys)
        self.obs_data, self.obs_error = self._particles.observe(error_model)

        self.potential = LawMajewski2010()
        self.satellite = satellite_today()

    def test_call(self):

        params = []
        params.append(Parameter(target=self.potential.q1,
                                attr="_value",
                                ln_prior=LogUniformPrior(*self.potential.q1._range)))

        model = StreamModel(self.potential, self.satellite, self._particles,
                            self.obs_data, self.obs_error, parameters=params)

        model([1.4])
        assert model.vector[0] == self.potential.q1.value

    def test_vector(self):

        params = []
        params.append(Parameter(target=self.potential.q1,
                                attr="_value",
                                ln_prior=LogUniformPrior(*self.potential.q1._range)))
        params.append(Parameter(target=self._particles,
                                attr="flat_X"))
        #params.append(Parameter(target=self._particles,
        #                        attr="tub"))
        model = StreamModel(self.potential, self.satellite, self._particles,
                            self.obs_data, self.obs_error, parameters=params)

        model([1.4] + list(np.random.random(size=self.Nparticles*6)) + \
              list(np.random.randint(6266, size=self.Nparticles)),
              self.t1, self.t2, -1.)

        assert model.vector[0] == self.potential.q1.value
        assert self.potential.q1.value == self.potential.parameters["q1"].value