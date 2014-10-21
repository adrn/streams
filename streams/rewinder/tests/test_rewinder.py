# coding: utf-8

""" Test the Rewinder model subclass """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys
import logging
import time

# Third-party
from astropy import log as logger
import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import streamteam.potential as sp
from streamteam.units import galactic

# Project
from .. import Rewinder, RewinderSampler
from ..likelihood import rewinder_likelihood
from ...util import streamspath

this_path = os.path.dirname(__file__)
output_path = os.path.join(streamspath, "output/tests/rewinder")
if not os.path.exists(output_path):
    os.makedirs(output_path)

logger.setLevel(logging.DEBUG)

class TestSimple(object):

    def setup(self):
        # read stuff from text file
        stardata = np.genfromtxt(os.path.join(this_path, "true_stars.txt"), names=True)
        progdata = np.genfromtxt(os.path.join(this_path, "true_prog.txt"), names=True)

        self.stars = np.vstack((stardata['x'],stardata['y'],stardata['z'],
                                stardata['vx'],stardata['vy'],stardata['vz'])).T.copy()[:16]

        self.prog = np.vstack((progdata['x'],progdata['y'],progdata['z'],
                               progdata['vx'],progdata['vy'],progdata['vz'])).T.copy()

        self.betas = stardata['tail'].copy()[:16]

    def test_call(self):
        potential = sp.LeeSutoNFWPotential(v_h=0.5, r_h=20., a=1., b=1., c=1., units=galactic)
        ll = rewinder_likelihood(-1., 6000.,
                                 potential.c_instance,
                                 self.prog, self.stars,
                                 2.5E6, 0.,
                                 1., self.betas, -0.3)
        # print(ll.shape)

    def test_time(self):
        tmp = np.zeros((6000,64))
        print("Spherical:")
        potential = sp.LeeSutoNFWPotential(v_h=0.5, r_h=20.0,
                                           a=1., b=1., c=1., units=galactic)

        nrepeat = 100
        t1 = time.time()
        for i in range(nrepeat):  # ~10 ms per call
            ll = rewinder_likelihood(-1., 6000,
                                     potential.c_instance,
                                     self.prog, self.stars,
                                     2.5E6, 0.,
                                     1., self.betas, -0.3)

        t = (time.time() - t1) / float(nrepeat)

        print("{} ms per call".format(t*1000.))

        print("\n")
        print("Triaxial, rotated:")
        potential = sp.LeeSutoNFWPotential(v_h=0.5, r_h=20.0, phi=0.2,
                                           a=1.3, b=1., c=0.8, units=galactic)

        t1 = time.time()
        for i in range(nrepeat):  # ~10 ms per call
            ll = rewinder_likelihood(-1., 6000,
                                     potential.c_instance,
                                     self.prog, self.stars,
                                     2.5E6, 0.,
                                     1., self.betas, -0.3)

        t = (time.time() - t1) / float(nrepeat)

        print("{} ms per call".format(t*1000.))

class TestConfig(object):

    def do_the_mcmc(self, sampler, p0):
        n = 100

        # burn in
        sampler.run_inference(p0, n)
        best_pos = sampler.flatchain[sampler.flatlnprobability.argmax()]
        sampler.reset()

        # restart walkers from best position, burn again
        new_pos = np.random.normal(best_pos, best_pos*0.02,
                                   size=(sampler.nwalkers, p0.shape[1]))
        sampler.run_inference(new_pos, n)
        pos = sampler.chain[:,-1]
        sampler.reset()

        # run for inference steps
        sampler.run_inference(pos, n)

        return sampler

    def make_plots(self, sampler, p0, path):
        for i in range(sampler.chain.shape[-1]):
            plt.clf()
            for chain in sampler.chain[...,i]:
                plt.plot(chain, marker=None, drawstyle='steps', alpha=0.2, color='k')

            for pp in p0[:,i]:
                plt.axhline(pp, alpha=0.2, color='r')

            plt.savefig(os.path.join(path, "param_{}.png".format(i)))

    def test1(self):
        """ Test 1:
                No uncertainties, fix alpha, fix theta, only sample over v_h, r_h
        """

        path = os.path.abspath(os.path.join(this_path, "test1.yml"))
        model = Rewinder.from_config(path)
        sampler = RewinderSampler(model, nwalkers=64)

        truth = np.array([0.5, 20.])
        p0_sigma = np.array([0.1, 1.])
        p0 = np.random.normal(truth, p0_sigma, size=(sampler.nwalkers, len(truth)))

        print("Model value at truth: {}".format(model(truth)))
        for pp in p0:
            if np.any(np.isnan(model(pp))):
                raise ValueError("Model returned -inf for initial position!")

        plot_path = os.path.join(output_path, "test1")
        if not os.path.exists(plot_path):
            os.mkdir(plot_path)

        sampler = self.do_the_mcmc(sampler, p0)
        self.make_plots(sampler, p0, plot_path)

    def test2(self):
        """ Test 2:
                No uncertainties, sample over v_h, r_h, alpha, theta, mass
        """

        path = os.path.abspath(os.path.join(this_path, "test1.yml"))
        model = Rewinder.from_config(path)
        sampler = RewinderSampler(model, nwalkers=64)

        truth = np.array([0.5, 20., 1.25, -0.3, 2.5E6])
        p0_sigma = np.array([0.1, 1., 0.1, 0.02, 1E5])
        p0 = np.random.normal(truth, p0_sigma, size=(sampler.nwalkers, len(truth)))

        print("Model value at truth: {}".format(model(truth)))
        for pp in p0:
            if np.any(np.isnan(model(pp))):
                raise ValueError("Model returned -inf for initial position!")

        plot_path = os.path.join(output_path, "test2")
        if not os.path.exists(plot_path):
            os.mkdir(plot_path)

        sampler = self.do_the_mcmc(sampler, p0)
        self.make_plots(sampler, p0, plot_path)
