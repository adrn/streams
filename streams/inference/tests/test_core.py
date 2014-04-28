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
from ..parameter import *
from ..prior import *

plot_path = "plots/tests/inference"
if not os.path.exists(plot_path):
    os.makedirs(plot_path)

class TestModelParameter(object):

    def test_string(self):
        p = ModelParameter("m", value=np.nan, truth=1.5,
                           prior=LogUniformPrior(1.,2.))
        assert repr(p) == "<ModelParameter 'm' truth=1.5>"
        assert str(p) == "m"

        p = ModelParameter("m", value=np.nan,
                           prior=LogUniformPrior(1.,2.))
        assert repr(p) == "<ModelParameter 'm'>"
        assert str(p) == "m"

class TestModel(object):

    def setup(self):
        np.random.seed(42)

        self.flat_model = Model()
        for name in "abcdefg":
            p = ModelParameter(name, value=np.nan, truth=np.random.random())
            self.flat_model.add_parameter(p)

        self.group_model = Model()
        for group in ["herp","derp"]:
            for name in "abcd":
                p = ModelParameter(name, value=np.nan, truth=np.random.random())
                self.group_model.add_parameter(p, parameter_group=group)

        self.vec_model = Model()
        for name in "abcd":
            length = np.random.randint(2,10)
            p = ModelParameter(name, value=[np.nan]*length, truth=np.random.random(size=length))
            self.vec_model.add_parameter(p)

        self.models = [self.group_model, self.flat_model, self.vec_model]

    def test_init(self):
        m = ModelParameter("m", value=np.nan, truth=1.5,
                           prior=LogUniformPrior(1.,2.))
        b = ModelParameter("b", value=np.nan, truth=6.7,
                           prior=LogUniformPrior(0.,10.))

        model = Model()
        model.add_parameter(m)
        model.add_parameter(b)
        model.parameters['main']['m']
        model.parameters['main']['b']

        assert np.all(model.truths == np.array([1.5,6.7]))

    def test_walk_parameters(self):
        for model in self.models:
            for group,name,p in model._walk(model.parameters):
                assert name == str(p)

    def test_decompose_compose(self):
        for model in self.models:
            vec = np.random.random(size=model.nparameters)
            decom = model.vector_to_parameters(vec)
            for group,name,p in model._walk(model.parameters):
                decom[group][name]

            com = model.parameters_to_vector(decom)
            assert np.all((vec-com) == 0.)

    def test_flatchain(self):
        nsteps = 1024
        for model in self.models:
            vec = np.random.random(size=(model.nparameters,nsteps))
            decom = model.vector_to_parameters(vec)
            for group,name,p in model._walk(model.parameters):
                print(decom[group][name].shape)

    def test_prior(self):
        for model in self.models:
            vec = np.random.random(size=model.nparameters)
            decom = model.vector_to_parameters(vec)
            print(model.ln_prior(decom))

    def test_likelihood(self):
        for model in self.models:
            vec = np.random.random(size=model.nparameters)
            decom = model.vector_to_parameters(vec)
            print(model.ln_likelihood(decom))


# class TestStreamModel(object):

#     def setup(self):

#         self.plot_path = os.path.join(plot_path, "StreamModel")
#         if not os.path.exists(self.plot_path):
#             os.mkdir(self.plot_path)

#         np.random.seed(52)
#         self.Nparticles = 25
#         self.simulation = SgrSimulation(mass="2.5e8")
#         self.args = (self.simulation.t1,self.simulation.t2,-1.)

#         self.particles = self.simulation.particles(N=self.Nparticles,
#                                                    expr="tub!=0")
#         self.particles = self.particles.to_frame(heliocentric)

#         self.potential = LawMajewski2010()
#         self.satellite = self.simulation.satellite()
#         self.satellite = self.satellite.to_frame(heliocentric)
