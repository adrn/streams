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
        # burn in
        sampler.run_inference(p0, 100)
        best_pos = sampler.flatchain[sampler.flatlnprobability.argmax()]
        sampler.reset()

        # restart walkers from best position, burn again
        new_pos = np.random.normal(best_pos, best_pos*0.02,
                                   size=(sampler.nwalkers, len(truth)))
        sampler.run_inference(new_pos, 100)
        pos = sampler.chain[:,-1]
        sampler.reset()

        # run for inference steps
        sampler.run_inference(pos, 100)

        return sampler

    def make_plots(self, sampler, p0, path):
        for i in range(sampler.shape[-1]):
            plt.clf()
            for chain in sampler.chain[...,i]:
                plt.plot(chain, marker=None, drawstyle='steps', alpha=0.2, color='k')

            for pp in p0[:,i]:
                plt.axhline(pp, alpha=0.2, color='k')

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


def test_build_model():
    from ..starsprogenitor import Stars, Progenitor
    from ..rewinder import Rewinder
    from streamteam.inference import ModelParameter, LogUniformPrior, LogPrior

    # read data from files
    stardata = np.genfromtxt(os.path.join(this_path, "true_stars.txt"), names=True)
    progdata = np.genfromtxt(os.path.join(this_path, "true_prog.txt"), names=True)

    # -----------------------------------------------------------------------------------
    # Progenitor
    #
    prog_hel = np.vstack((progdata["l"],progdata["b"],progdata["d"],
                          progdata["mul"],progdata["mub"],progdata["vr"])).T.copy()
    progenitor = Progenitor(data=np.zeros_like(prog_hel),
                            errors=np.zeros_like(prog_hel),
                            truths=prog_hel)

    progenitor.parameters['m0'] = ModelParameter('m0',
                                                 truth=progdata['mass'][0],
                                                 prior=LogPrior()) # TODO: logspace?

    progenitor.parameters['mdot'] = ModelParameter('mdot', truth=0., prior=LogPrior())
    progenitor.parameters['mdot'].frozen = 0.

    progenitor.parameters['alpha'] = ModelParameter('alpha', shape=(1,),
                                                    prior=LogUniformPrior(1., 3.))

    # -----------------------------------------------------------------------------------
    # Stars
    #
    # star_hel = np.vstack((stardata["l"],stardata["b"],stardata["d"],
    #                       stardata["mul"],stardata["mub"],stardata["vr"])).T.copy()
    stars = Stars(data=np.zeros_like(star_hel),
                  errors=np.zeros_like(star_hel),
                  truths=star_hel)

    stars.parameters['tail'] = ModelParameter('tail', truth=stardata['tail'])
    stars.parameters['tail'].frozen = stars.parameters['tail'].truth

    stars.parameters['tub'] = ModelParameter('tub', truth=stars_data['tub'])

    model = Rewinder(potential, progenitor, stars, t1, t2, dt, K=config["K"])

    # -----------------------------------------------------------------------------------
    # Potential
    #
    potential = sp.LeeSutoNFWPotential(v_h=ppars['v_h'], r_h=ppars['r_h'],
                                       a=1., b=1., c=1., units=galactic)

    pparams = []
    pparams.append(ModelParameter('v_h', truth=0.5, prior=LogUniformPrior(0.3,0.8)))
    pparams.append(ModelParameter('r_h', truth=20, prior=LogUniformPrior(10,40.)))
    pparams.append(ModelParameter('a', truth=1., prior=LogPrior()))
    pparams[-1].fixed = 1.
    pparams.append(ModelParameter('b', truth=1., prior=LogPrior()))
    pparams[-1].fixed = 1.
    pparams.append(ModelParameter('c', truth=1., prior=LogPrior()))
    pparams[-1].fixed = 1.
