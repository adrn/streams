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
from ...io import read_hdf5

# TODO: MAJOR TODO HERE!

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

# # read in the particles
# d_path = os.path.join(os.environ["STREAMSPATH"], "data/observed_particles/")
# data_file = "N128.hdf5"
# d = io.read_hdf5(os.path.join(d_path, data_file))

# # integration stuff
# t1 = float(d["t1"])
# t2 = float(d["t2"])
# dt = -1.

# # potential
# q1 = ModelParameter('q1', prior=LogUniformPrior(1.,1.75), truth=1.38)
# assert q1.prior(1.5) == 0.
# assert q1.prior(2.5) == -np.inf

# qz = ModelParameter('qz', prior=LogUniformPrior(1.,1.75), truth=1.36)
# assert qz.prior(1.5) == 0.
# assert qz.prior(2.5) == -np.inf

# v_halo = ModelParameter('v_halo', prior=LogUniformPrior(0.1, 0.15),
#                         truth=0.124625659009114)
# assert v_halo.prior(0.121) == 0.
# assert v_halo.prior(0.221) == -np.inf

# phi = ModelParameter('phi', prior=LogUniformPrior(1.5, 1.9),
#                      truth=1.692969)
# assert phi.prior(1.69) == 0.

# # particles
# nparticles = 5
# particles = ObservedParticle(d['particles']._X[:nparticles].T,
#                              d['particles']._error_X[:nparticles].T,
#                              frame=d['particles'].frame,
#                              units=d['particles']._internal_units)

# true_particles = Particle(d['true_particles']._X[:nparticles].T,
#                           frame=d['true_particles'].frame,
#                           units=d['true_particles']._internal_units)
# true_particles.tub = d['true_particles'].tub[:nparticles]

# #X = ModelParameter('_X', truth=true_particles._X)
# priors = [LogNormalPrior(particles._X[ii],particles._error_X[ii]) for ii in range(nparticles)]
# X = ModelParameter('_X', value=particles._X, prior=priors,
#                    truth=true_particles._X)

# test_X = np.random.normal(particles._X, particles._error_X)
# test_X2 = np.random.normal(particles._X, particles._error_X*10.)
# assert X.prior(test_X).shape == (nparticles,)
# assert np.sum(X.prior(test_X2)) < np.sum(X.prior(test_X))

# #truth=true_particles.tub,
# priors = [LogUniformPrior(t2, t1) for ii in range(nparticles)]
# tub = ModelParameter('tub', value=np.zeros(nparticles), prior=priors,
#                      truth=true_particles.tub)

# test_tubs = np.zeros(nparticles)
# assert np.sum(tub.prior(test_tubs - 100.)) == -np.inf
# assert np.sum(tub.prior(test_tubs + 100.)) == 0.

# # satellite
# satellite = d['satellite']
# priors = [LogNormalPrior(satellite._X[0],satellite._error_X[0])]
# s_X = ModelParameter('_X', value=satellite._X, prior=priors,
#                      truth=d['true_satellite']._X)

# test_X = np.random.normal(satellite._X, satellite._error_X)
# test_X2 = np.random.normal(satellite._X, satellite._error_X*10.)
# assert s_X.prior(test_X).shape == (1,)
# assert np.sum(s_X.prior(test_X2)) < np.sum(s_X.prior(test_X))

# # MODEL
# potential = sp.LawMajewski2010()
# model = StreamModel(potential)

# assert len(model.sample_priors()) == 0

# model.add_parameter('potential', q1)
# model.add_parameter('potential', qz)
# model.add_parameter('potential', v_halo)
# model.add_parameter('potential', phi)

# assert len(model.sample_priors()) == 4

# model.add_parameter('particles', X)
# assert len(model.sample_priors()) == nparticles*6 + 4

# model.add_parameter('particles', tub)
# assert len(model.sample_priors()) == nparticles*7 + 4

# model.add_parameter('satellite', s_X)
# assert len(model.sample_priors()) == nparticles*7 + 4 + 6

# true_ln_p = model.ln_posterior(model.truths, t1, t2, dt)
# for ii in range(1):
#     p0 = model.sample_priors()
#     ln_p = model.ln_posterior(p0, t1, t2, dt)
#     assert true_ln_p > ln_p

# import time
# import emcee
# import multiprocessing

# a = time.time()
# t = model.truths
# for ii in range(10):
#     ln_p = model.ln_posterior(t, t1, t2, dt)
# print((time.time() - a)/10., "seconds per posterior call")

# nwalkers = model.nparameters*2 + 2
# print(nwalkers, "walkers for", model.nparameters, "parameters")

# a = time.time()
# p0 = model.sample_priors(size=nwalkers)
# print(time.time() - a, "seconds for sampling p0")

# assert p0.shape == (nwalkers, model.nparameters)

# #pickle.dumps(model)
# #sys.exit(0)

# pool = multiprocessing.Pool(2)
# sampler = emcee.EnsembleSampler(nwalkers, model.nparameters,
#                                 model,
#                                 args=(t1,t2,dt),
#                                 pool=pool)

# print("running")
# sampler.run_mcmc(p0, 2)
# #model.ln_posterior)

# # model.run()

# # model.chain["potential"]["q1"]
# # model.chain["potential"]["qz"]
# # model.chain["potential"]["v_halo"]
# # model.chain["potential"]["phi"]

# # model.chain["particles"]["_X"]
# # model.chain["particles"]["tub"]

# # model.chain["satellite"]["_X"]