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
from ...coordinates.frame import galactocentric, heliocentric
from ..parameter import *
from ..prior import *
from ...io import SgrSimulation

plot_path = "plots/tests/inference"
if not os.path.exists(plot_path):
    os.makedirs(plot_path)

class TestStreamModel(object):

    def setup(self):

        self.plot_path = os.path.join(plot_path, "StreamModel")
        if not os.path.exists(self.plot_path):
            os.mkdir(self.plot_path)

        np.random.seed(52)
        self.Nparticles = 25
        self.simulation = SgrSimulation(mass="2.5e8")
        self.args = (self.simulation.t1,self.simulation.t2,-1.)

        self.particles = self.simulation.particles(N=self.Nparticles,
                                                   expr="tub!=0")
        self.particles = self.particles.to_frame(heliocentric)

        self.potential = LawMajewski2010()
        self.satellite = self.simulation.satellite()
        self.satellite = self.satellite.to_frame(heliocentric)

    def test_call(self):

        params = []
        params.append(ModelParameter(target=self.potential.q1,
                        attr="_value",
                        ln_prior=LogUniformPrior(*self.potential.q1._range)))

        model = StreamModel(self.potential, self.satellite, self.particles.copy(), parameters=params)

        model([1.4], *self.args)
        assert model.vector[0] == self.potential.q1.value

    def test_vector(self):
        particles = self.particles.copy()
        satellite = self.satellite.copy()

        params = []
        params.append(ModelParameter(target=self.potential.q1,
                                attr="_value",
                                ln_prior=LogUniformPrior(*self.potential.q1._range)))
        params.append(ModelParameter(target=particles,
                                attr="_X"))
        params.append(ModelParameter(target=particles,
                                attr="tub"))

        model = StreamModel(self.potential, satellite, particles,
                            parameters=params)

        model([1.4] + list(np.random.random(size=self.Nparticles*6)) + \
              list(np.random.randint(6266, size=self.Nparticles)), *self.args)

        assert model.vector[0] == self.potential.q1.value
        assert self.potential.q1.value == self.potential.parameters["q1"].value

    def test_potential_likelihood(self):

        fig = plt.figure()
        ax = fig.add_subplot(111)

        for p_name in self.potential.parameters.keys():
            potential = LawMajewski2010()

            particles = self.particles.copy()
            satellite = self.satellite.copy()

            p = potential.parameters[p_name]

            params = []
            params.append(ModelParameter(target=p,
                                    attr="_value",
                                    ln_prior=LogUniformPrior(*p._range)))

            model = StreamModel(potential, satellite, particles,
                                parameters=params)

            Ls = []
            vals = np.linspace(0.9, 1.1, 31)*p._truth
            for q in vals:
                Ls.append(model([q], *self.args))

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

            particles = self.particles.copy()
            satellite = self.satellite.copy()

            p = potential.parameters[p_name]

            params = []
            params.append(ModelParameter(target=p,
                                    attr="_value",
                                    ln_prior=LogUniformPrior(*p._range)))
            params.append(ModelParameter(target=particles,
                                         attr="_X",
                                         ln_prior=LogPrior()))

            model = StreamModel(potential, satellite, particles,
                                parameters=params)

            Ls = []
            vals = np.linspace(0.8, 1.2, 31)*p._truth
            for q in vals:
                Ls.append(model([q] + list(np.ravel(particles._X)), *self.args))

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

        _particles = self.particles.copy()
        particles = self.particles.copy()
        satellite = self.satellite.copy()

        params = []
        params.append(ModelParameter(target=particles,
                                     attr="_X",
                                     ln_prior=LogPrior()))

        model = StreamModel(potential, satellite, particles,
                            parameters=params)

        Ls = []
        vals = np.linspace(0.5, 1.5, 21)
        for v in vals:
            sampled_X = np.ravel(_particles._X)*v
            Ls.append(model(list(sampled_X), *self.args))

        ax.cla()
        ax.plot(vals, Ls)
        ax.axvline(1., color='r')
        ax.set_ylabel("ln likelihood", fontsize=24)
        fig.savefig(os.path.join(self.plot_path, "particles.png"))

    def test_particle_tub_likelihood(self):
        fig = plt.figure()
        ax = fig.add_subplot(111)

        _particles = self.particles.copy()
        for ii in range(self.Nparticles):
            potential = LawMajewski2010()

            particles = self.particles.copy()
            satellite = self.satellite.copy()

            params = []
            params.append(ModelParameter(target=particles,
                                         attr="tub",
                                         ln_prior=LogPrior()))

            model = StreamModel(potential, satellite, particles,
                                parameters=params)

            true_tub = np.array(self.particles.tub)
            tub = np.array(self.particles.tub)

            Ls = []
            #vals = np.linspace(0.8, 1.2, 31)*true_tub[ii]
            vals = np.linspace(self.simulation.t2, self.simulation.t1, 21)
            for v in vals:
                tub[ii] = v
                Ls.append(model(list(tub), *self.args))

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

        particles = self.particles.copy()
        satellite = self.satellite.copy()

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
                    for n in particles.frame.coord_names]).T
        covs = [np.diag(s**2) for s in sigmas]

        prior = LogNormalPrior(np.array(particles._X),
                               cov=np.array(covs))
        params.append(ModelParameter(target=particles,
                                     attr="_X",
                                     ln_prior=prior))

        params.append(ModelParameter(target=satellite,
                                     attr="_X",
                                     ln_prior=LogPrior()))

        model = StreamModel(potential, satellite, particles,
                            parameters=params)

        fnpickle(model, os.path.join(self.plot_path, "test.pickle"))

    def test_time_likelihood(self):
        potential = LawMajewski2010()

        particles = self.particles.copy()
        satellite = self.satellite.copy()

        p = potential.q1

        params = []
        params.append(ModelParameter(target=p,
                                attr="_value",
                                ln_prior=LogUniformPrior(*p._range)))

        model = StreamModel(potential, satellite, particles,
                            parameters=params)

        N = 10
        a = pytime.time()
        for ii in range(N):
            model([1.5], *self.args)

        print("{} sec. per likelihood call".format((pytime.time()-a)/float(N)))

    def test_time_sampler(self):
        import emcee
        potential = LawMajewski2010()

        particles = self.particles.copy()
        satellite = self.satellite.copy()

        p = potential.q1

        params = []
        params.append(ModelParameter(target=p,
                                attr="_value",
                                ln_prior=LogUniformPrior(*p._range)))

        model = StreamModel(potential, satellite, particles,
                            parameters=params)

        Nwalkers = 8
        Nsteps = 10
        p0 = np.random.random(size=Nwalkers)*2
        p0 = p0.reshape(Nwalkers,1)

        a = pytime.time()
        sampler = emcee.EnsembleSampler(Nwalkers, model.ndim, model, args=self.args)
        pos, xx, yy = sampler.run_mcmc(p0, Nsteps)
        b = pytime.time()

        print("{} sec. for sampler test".format(b-a))
        print("{} sec. per call".format( (b-a)/Nsteps/float(Nwalkers)))


    def test_likelihood_shape(self):

        this_plot_path = os.path.join(plot_path, "likelihood_shape")
        if not os.path.exists(this_plot_path):
            os.makedirs(this_plot_path)

        for p_name in self.potential.parameters.keys():
            particles = self.particles.copy()
            satellite = self.satellite.copy()
            potential = LawMajewski2010()

            p = potential.parameters[p_name]
            vals = np.linspace(0.75,1.25,51)*p._truth

            params = []
            params.append(ModelParameter(target=p, attr="_value",
                                         ln_prior=LogUniformPrior(*p._range)))

            model = StreamModel(potential, satellite, particles,
                                parameters=params)

            Ls = []
            for val in vals:
                Ls.append(model([val], *self.args))

            plt.clf()
            plt.plot(vals,Ls)
            plt.axvline(p._truth)
            plt.savefig(os.path.join(this_plot_path, "L_vs_{}.png".format(p_name)))

def test_likelihood_observed_particles():
    from streams.observation.gaia import gaia_spitzer_errors

    np.random.seed(52)
    simulation = SgrSimulation(mass="2.5e8")

    particles = simulation.particles(N=1, expr="tub!=0")
    particles = particles.to_frame(heliocentric)

    potential = LawMajewski2010()
    satellite = simulation.satellite()
    satellite = satellite.to_frame(heliocentric)

    fig = plt.figure()
    ax = fig.add_subplot(111)

    particle_errors = gaia_spitzer_errors(particles)
    o_particles = particles.observe(particle_errors)
    sigmas = np.array([o_particles.errors[n].decompose(usys).value \
                for n in o_particles.frame.coord_names]).T
    covs = [np.diag(s**2) for s in sigmas]

    prior = LogNormalPrior(np.array(o_particles._X),
                           cov=np.array(covs))

    potential = LawMajewski2010()

    o_particles = o_particles.copy()
    satellite = satellite.copy()

    p = ModelParameter(target=o_particles,
                       attr="_X",
                       ln_prior=prior)
    params = [p]
    model = StreamModel(potential, satellite, particles, parameters=params)

    assert p.ln_prior() == (-0.5*6*np.log(2*np.pi) - 0.5*np.array([np.linalg.slogdet(c)[1]
                                                                    for c in covs]))[0]

    facs = np.linspace(0.5,1.5,11)
    Ps = []

    _X = o_particles._X
    arr = np.ones_like(o_particles._X)
    for fac in facs:
        arr[:,2:] = fac
        X = arr*_X
        print("posterior", model(np.ravel(X), *self.args))
        print("prior", model.ln_prior())
        print("likelihood", model.ln_likelihood())
        print()
        Ps.append(model(np.ravel(X), *self.args))

    ax.plot(facs, Ps)
    ax.set_xlabel("Err. Factor")
    ax.set_ylabel("Posterior")
    fig.savefig(os.path.join(plot_path, "observed_particle_prior.png"))