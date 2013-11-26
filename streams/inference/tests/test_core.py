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
from astropy.io.misc import fnpickle
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
                                attr="_X"))
        params.append(ModelParameter(target=particles,
                                attr="tub"))

        model = StreamModel(self.potential, self.simulation,
                            self.satellite, particles,
                            parameters=params)

        model([1.4] + list(np.random.random(size=self.Nparticles*6)) + \
              list(np.random.randint(6266, size=self.Nparticles)))

        assert model.vector[0] == self.potential.q1.value
        assert self.potential.q1.value == self.potential.parameters["q1"].value

    def test_potential_likelihood(self):

        fig = plt.figure()
        ax = fig.add_subplot(111)

        for p_name in self.potential.parameters.keys():
            potential = LawMajewski2010()

            particles = self.particles.to_frame("heliocentric")
            satellite = self.satellite.to_frame("heliocentric")

            p = potential.parameters[p_name]

            params = []
            params.append(ModelParameter(target=p,
                                    attr="_value",
                                    ln_prior=LogUniformPrior(*p._range)))

            model = StreamModel(potential, self.simulation,
                                satellite, particles,
                                parameters=params)

            Ls = []
            vals = np.linspace(0.8, 1.2, 31)*p._truth
            for q in vals:
                Ls.append(model([q]))

            ax.cla()
            ax.plot(vals, Ls)
            ax.axvline(p._truth, color='r')
            ax.set_ylabel("ln likelihood", fontsize=24)
            ax.set_xlabel(p.latex, fontsize=24)
            fig.savefig(os.path.join(self.plot_path,
                                     "{0}.png".format(p_name)))

    def test_potential_likelihood_w_particle_X(self):

        fig = plt.figure()
        ax = fig.add_subplot(111)

        for p_name in self.potential.parameters.keys():
            potential = LawMajewski2010()

            particles = self.particles.to_frame("heliocentric")
            satellite = self.satellite.to_frame("heliocentric")

            p = potential.parameters[p_name]

            params = []
            params.append(ModelParameter(target=p,
                                    attr="_value",
                                    ln_prior=LogUniformPrior(*p._range)))
            params.append(ModelParameter(target=particles,
                                         attr="_X",
                                         ln_prior=LogPrior()))

            model = StreamModel(potential, self.simulation,
                                satellite, particles,
                                parameters=params)

            Ls = []
            vals = np.linspace(0.8, 1.2, 31)*p._truth
            for q in vals:
                Ls.append(model([q] + list(np.ravel(particles._X))))

            ax.cla()
            ax.plot(vals, Ls)
            ax.axvline(p._truth, color='r')
            ax.set_ylabel("ln likelihood", fontsize=24)
            ax.set_xlabel(p.latex, fontsize=24)
            fig.savefig(os.path.join(self.plot_path,
                                     "{0}_X.png".format(p_name)))

    def test_particle_position_likelihood(self):
        fig = plt.figure()
        ax = fig.add_subplot(111)

        potential = LawMajewski2010()

        _particles = self.particles.to_frame("heliocentric")
        particles = self.particles.to_frame("heliocentric")
        satellite = self.satellite.to_frame("heliocentric")

        params = []
        params.append(ModelParameter(target=particles,
                                     attr="_X",
                                     ln_prior=LogPrior()))

        model = StreamModel(potential, self.simulation,
                            satellite, particles,
                            parameters=params)

        Ls = []
        vals = np.linspace(0.5, 1.5, 21)
        for v in vals:
            sampled_X = np.ravel(_particles._X)*v
            Ls.append(model(list(sampled_X)))

        ax.cla()
        ax.plot(vals, Ls)
        ax.axvline(1., color='r')
        ax.set_ylabel("ln likelihood", fontsize=24)
        fig.savefig(os.path.join(self.plot_path, "particles.png"))

    def test_particle_tub_likelihood(self):
        fig = plt.figure()
        ax = fig.add_subplot(111)

        _particles = self.particles.to_frame("heliocentric")
        for ii in range(self.Nparticles):
            potential = LawMajewski2010()

            particles = self.particles.to_frame("heliocentric")
            satellite = self.satellite.to_frame("heliocentric")

            params = []
            params.append(ModelParameter(target=particles,
                                         attr="tub",
                                         ln_prior=LogPrior()))

            model = StreamModel(potential, self.simulation,
                                satellite, particles,
                                parameters=params)

            true_tub = np.array(_particles.tub)
            tub = _particles.tub

            Ls = []
            vals = np.linspace(0.8, 1.2, 31)*true_tub[ii]
            for v in vals:
                tub[ii] = v
                Ls.append(model(list(tub)))

            ax.cla()
            ax.plot(vals, Ls)
            ax.axvline(true_tub[ii], color='r')
            ax.set_title("particle {0}".format(ii))
            ax.set_ylabel("ln likelihood", fontsize=24)
            ax.set_xlabel("particle {0}, tub".format(ii), fontsize=24)
            fig.savefig(os.path.join(self.plot_path,
                        "tub_particle_{0}.png".format(ii)))

    def test_pickle(self):
        from ...observation.gaia import gaia_spitzer_errors

        potential = LawMajewski2010()

        particles = self.particles.to_frame("heliocentric")
        satellite = self.satellite.to_frame("heliocentric")

        particle_errors = gaia_spitzer_errors(particles)
        particles = particles.observe(particle_errors)

        params = []
        for p_name in self.potential.parameters.keys():
            p = potential.parameters[p_name]
            params.append(ModelParameter(target=p,
                                    attr="_value",
                                    ln_prior=LogUniformPrior(*p._range)))

        params.append(ModelParameter(target=particles,
                                     attr="tub",
                                     ln_prior=LogPrior()))

        sigmas = np.array([particles.errors[n].decompose(usys).value \
                    for n in particles.names]).T
        covs = [np.diag(s**2) for s in sigmas]

        prior = LogNormalPrior(np.array(particles._X),
                               cov=np.array(covs))
        params.append(ModelParameter(target=particles,
                                     attr="_X",
                                     ln_prior=prior))

        params.append(ModelParameter(target=satellite,
                                     attr="_X",
                                     ln_prior=LogPrior()))

        model = StreamModel(potential, self.simulation,
                            satellite, particles,
                            parameters=params)

        fnpickle(model, os.path.join(self.plot_path, "test.pickle"))