# coding: utf-8

from __future__ import absolute_import, unicode_literals, division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys
import time as pytime
import copy

# Third-party
import emcee
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

plot_path = "plots/tests/inference/sampler"
if not os.path.exists(plot_path):
    os.makedirs(plot_path)

class TestStreamModel(object):

    def setup(self):
        from streams.io import SgrSimulation

        np.random.seed(52)
        self.Nparticles = 25
        self.simulation = SgrSimulation(mass="2.5e8")

        self.particles = self.simulation.particles(N=self.Nparticles,
                                                   expr="tub!=0")
        self.particles = self.particles.to_frame(heliocentric)

        self.potential = LawMajewski2010()
        self.satellite = self.simulation.satellite()
        self.satellite = self.satellite.to_frame(heliocentric)

    def test_time_sampler(self):

        p_name = "q1"

        potential = LawMajewski2010()

        particles = self.particles.copy()
        satellite = self.satellite.copy()

        p = potential.parameters[p_name]

        params = []
        params.append(ModelParameter(target=p,
                                attr="_value",
                                ln_prior=LogUniformPrior(*p._range)))

        model = StreamModel(potential, self.simulation,
                            satellite, particles,
                            parameters=params)

        Nwalkers = 4
        Nsteps = 100
        p0 = np.random.random(size=Nwalkers)*2
        p0 = p0.reshape(Nwalkers,1)

        a = time.time()
        sampler = emcee.EnsembleSampler(Nwalkers, model.ndim, model)
        pos, xx, yy = sampler.run_mcmc(p0, Nsteps)
        b = time.time()

        print("{} sec. for sampler test".format(b-a))
        print("{} sec. per call".format( (b-a)/Nsteps/float(Nwalkers)))

    def test_sample_potential_params(self):

        Nwalkers = 4
        Nsteps = 50
        for p_name in self.potential.parameters.keys():
            potential = LawMajewski2010()

            particles = self.particles.copy()
            satellite = self.satellite.copy()

            p = potential.parameters[p_name]

            params = []
            params.append(ModelParameter(target=p,
                                    attr="_value",
                                    ln_prior=LogUniformPrior(*p._range)))

            model = StreamModel(potential, self.simulation,
                                satellite, particles,
                                parameters=params)

            p0 = np.random.uniform(0.75, 1.25, size=Nwalkers).reshape(Nwalkers,1)*p._truth

            sampler = emcee.EnsembleSampler(Nwalkers, model.ndim, model)
            pos, nuh, uh = sampler.run_mcmc(p0, Nsteps)

            plt.clf()
            plt.axhline(p._truth, linestyle='--', color='#fff000', lw=5.)
            for ii in range(Nwalkers):
                plt.plot(sampler.chain[ii], drawstyle='steps', alpha=0.5, lw=2.)
            plt.savefig(os.path.join(plot_path, "sampled_{}.png".format(p_name)))