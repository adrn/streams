# coding: utf-8

from __future__ import absolute_import, unicode_literals, division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys
import time as pytime
import copy

# Third-party
import numpy as np
import pytest
import astropy.units as u
import matplotlib.pyplot as plt

from ... import usys
from ..core import *
from ...potential.lm10 import LawMajewski2010
from ...coordinates import _gc_to_hel
from ..parameter import *
from ..prior import *

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

class TestStreamModel(object):

    def setup(self):

        self.plot_path = os.path.join(plot_path, "StreamModel")
        if not os.path.exists(self.plot_path):
            os.mkdir(self.plot_path)

        from streams.io import SgrSimulation

        np.random.seed(52)
        self.Nparticles = 10
        self.simulation = SgrSimulation(mass="2.5e8")

        self.particles = self.simulation.particles(N=self.Nparticles,
                                                   expr="tub!=0")

        self.potential = LawMajewski2010()
        self.satellite = self.simulation.satellite()

    def test_call(self):

        params = []
        params.append(ModelParameter(target=self.potential.q1,
                        attr="_value",
                        ln_prior=LogUniformPrior(*self.potential.q1._range)))

        model = StreamModel(self.potential, self.simulation, self.satellite,
                            self.particles.copy(), parameters=params)

        model([1.4])
        assert model.vector[0] == self.potential.q1.value

    def test_vector(self):
        particles = self.particles.copy()

        params = []
        params.append(ModelParameter(target=self.potential.q1,
                                attr="_value",
                                ln_prior=LogUniformPrior(*self.potential.q1._range)))
        params.append(ModelParameter(target=particles,
                                attr="flat_X"))
        params.append(ModelParameter(target=particles,
                                attr="tub"))

        model = StreamModel(self.potential, self.satellite, particles,
                            parameters=params)

        model([1.4] + list(np.random.random(size=self.Nparticles*6)) + \
              list(np.random.randint(6266, size=self.Nparticles)),
              self.simulation.t1, self.simulation.t2, -1.)

        assert model.vector[0] == self.potential.q1.value
        assert self.potential.q1.value == self.potential.parameters["q1"].value

    def test_potential_likelihood(self):

        fig = plt.figure()
        ax = fig.add_subplot(111)

        for p_name in self.potential.parameters.keys():
            _particles = copy.deepcopy(self._particles)
            potential = LawMajewski2010()
            p = potential.parameters[p_name]

            params = []
            params.append(ModelParameter(target=p,
                                    attr="_value",
                                    ln_prior=LogUniformPrior(*p._range)))
            params.append(ModelParameter(target=_particles,
                                    attr="flat_X"))
            params.append(ModelParameter(target=_particles,
                                    attr="tub"))

            model = StreamModel(potential, self.satellite, _particles,
                                self.obs_data, self.obs_error,
                                parameters=params)

            sampled_X = np.ravel(_particles._X)
            tub = _particles.tub # true

            Ls = []
            vals = np.linspace(0.8, 1.2, 31)*p._truth
            for q in vals:
                Ls.append(model([q] + list(sampled_X) + list(tub), \
                          self.t1, self.t2, -1.))

            ax.cla()
            ax.plot(vals, Ls)
            ax.axvline(p._truth, color='r')
            ax.set_ylabel("ln likelihood", fontsize=24)
            ax.set_xlabel(p.latex, fontsize=24)
            fig.savefig(os.path.join(self.plot_path,
                                     "{0}.png".format(p_name)))

    def test_particle_position_likelihood(self):
        fig = plt.figure()
        ax = fig.add_subplot(111)

        for ii in range(self.Nparticles*6):
            _particles = copy.deepcopy(self._particles)
            potential = LawMajewski2010()

            params = []
            params.append(ModelParameter(target=_particles,
                                    attr="flat_X"))
            params.append(ModelParameter(target=_particles,
                                    attr="tub"))

            model = StreamModel(potential, self.satellite, _particles,
                                self.obs_data, self.obs_error,
                                parameters=params)

            true_sampled_X = np.ravel(self._particles._X)
            sampled_X = np.ravel(_particles._X)
            tub = _particles.tub # true

            Ls = []
            vals = np.linspace(0.8, 1.2, 31)*true_sampled_X[ii]
            for v in vals:
                sampled_X[ii] = v
                Ls.append(model(list(sampled_X) + list(tub), \
                          self.t1, self.t2, -1.))

            dist = np.sqrt(np.sum(self._particles._X[int(ii / 6),:3]**2))

            ax.cla()
            ax.plot(vals, Ls)
            ax.axvline(true_sampled_X[ii], color='r')
            ax.set_title("particle {0}, dist={1}".format(int(ii / 6), dist))
            ax.set_ylabel("ln likelihood", fontsize=24)
            ax.set_xlabel("dim {0}".format(ii%6), fontsize=24)
            fig.savefig(os.path.join(self.plot_path,
                      "particle_{0}_dim{1}.png".format(int(ii / 6), ii % 6)))

    def test_particle_tub_likelihood(self):
        fig = plt.figure()
        ax = fig.add_subplot(111)

        for ii in range(self.Nparticles):
            _particles = copy.deepcopy(self._particles)
            potential = LawMajewski2010()

            params = []
            params.append(ModelParameter(target=_particles,
                                    attr="flat_X"))
            params.append(ModelParameter(target=_particles,
                                    attr="tub"))

            model = StreamModel(potential, self.satellite, _particles,
                                self.obs_data, self.obs_error,
                                parameters=params)

            sampled_X = np.ravel(_particles._X)

            true_tub = np.array(_particles.tub)
            tub = _particles.tub

            Ls = []
            vals = np.linspace(0.8, 1.2, 31)*true_tub[ii]
            for v in vals:
                tub[ii] = v
                Ls.append(model(list(sampled_X) + list(tub),
                                self.t1, self.t2, -1.))

            dist = np.sqrt(np.sum(self._particles._X[ii,:3]**2))

            ax.cla()
            ax.plot(vals, Ls)
            ax.axvline(true_tub[ii], color='r')
            ax.set_title("particle {0}, dist={1}".format(ii, dist))
            ax.set_ylabel("ln likelihood", fontsize=24)
            ax.set_xlabel("particle {0}, tub".format(ii), fontsize=24)
            fig.savefig(os.path.join(self.plot_path,
                        "tub_particle_{0}.png".format(ii)))