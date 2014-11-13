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
import pytest
import gary.dynamics as sd
import gary.potential as sp
from gary.units import galactic

# Project
from .. import Rewinder, RewinderSampler
from ..likelihood import rewinder_likelihood
from ..py_likelihood import rewinder_likelihood as py_rewinder_likelihood
from ...util import streamspath
from ... import heliocentric_names

logger.setLevel(logging.DEBUG)

this_path = os.path.dirname(__file__)
output_path = os.path.join(streamspath, "output/tests/rewinder")
if not os.path.exists(output_path):
    logger.debug("Creating directory {}".format(output_path))
    os.makedirs(output_path)

class TestTrueSimple(object):

    def setup(self):
        # read stuff from text file
        stardata = np.genfromtxt(os.path.join(this_path, "true_stars.txt"), names=True)
        progdata = np.genfromtxt(os.path.join(this_path, "true_prog.txt"), names=True)

        self.nstars = 16
        self.stars = np.vstack((stardata['x'],stardata['y'],stardata['z'],
                                stardata['vx'],stardata['vy'],stardata['vz'])).T.copy()[:self.nstars]

        self.prog = np.vstack((progdata['x'],progdata['y'],progdata['z'],
                               progdata['vx'],progdata['vy'],progdata['vz'])).T.copy()

        self.betas = stardata['tail'].copy()[:self.nstars]

    @pytest.mark.skipif(True, reason="derp.")
    def test_plot(self):
        # Have to enable debug stuff in Cython file.
        potential = sp.LeeSutoTriaxialNFWPotential(v_h=0.5, r_h=20., a=1., b=1., c=1., units=galactic)

        ll = np.zeros((6000,self.nstars), dtype=float)
        x,v = rewinder_likelihood(ll, -1., 6000, potential.c_instance,
                                  self.prog, self.stars, 2.5E6, 0.,
                                  1.25, self.betas, -0.3, True)

        w = np.vstack((x.T,v.T)).T
        fig = sd.plot_orbits(w, ix=0, marker=None, alpha=0.5)
        plt.show()

    def test_call(self):
        # potential = sp.SphericalNFWPotential(v_c=0.5*(np.log(2)-0.5), r_s=20., units=galactic)
        potential = sp.LeeSutoTriaxialNFWPotential(v_h=0.5, r_h=20., a=1., b=1., c=1., units=galactic)
        ll = np.zeros((6000,self.nstars), dtype=float) - 9999.
        rewinder_likelihood(ll,
                            -1., 6000,
                            potential.c_instance,
                            self.prog, self.stars,
                            2.5E6, 0.,
                            1.25, self.betas, -0.3,
                            False)

        plt.plot(ll[:,0])
        return

        from scipy.misc import logsumexp
        b = np.ones((6000,1))
        b[0] = b[-1] = 0.5
        L = logsumexp(ll, axis=0, b=b).sum()
        print(L)

    def test_call_py(self):
        # potential = sp.SphericalNFWPotential(v_c=0.5*(np.log(2)-0.5), r_s=20., units=galactic)
        potential = sp.LeeSutoTriaxialNFWPotential(v_h=0.5, r_h=20., a=1., b=1., c=1., units=galactic)
        ll = py_rewinder_likelihood(-1., 6000, potential, self.prog, self.stars,
                                    2.5E6, 0., 1.25, self.betas, -0.3, True)

        plt.plot(ll[:,0])
        plt.show()
        return

        from scipy.misc import logsumexp
        b = np.ones((6000,1))
        b[0] = b[-1] = 0.5
        L = logsumexp(ll, axis=0, b=b).sum()
        print(L)

    def test_time(self):
        ll = np.zeros((6000,self.nstars), dtype=float)

        print("Spherical w/ {} stars:".format(self.nstars))
        potential = sp.SphericalNFWPotential(v_c=0.5*np.sqrt(np.log(2)-0.5), r_s=20.0, units=galactic)

        nrepeat = 100
        t1 = time.time()
        for i in range(nrepeat):  # ~10 ms per call
            rewinder_likelihood(ll, -1., 6000,
                                potential.c_instance,
                                self.prog, self.stars,
                                2.5E6, 0.,
                                1., self.betas, -0.3,
                                True)

        t = (time.time() - t1) / float(nrepeat)

        print("{} ms per call".format(t*1000.))

        # print("\n")
        # print("Triaxial, rotated:")
        # potential = sp.LeeSutoTriaxialNFWPotential(v_h=0.5, r_h=20.0, phi=0.2,
        #                                    a=1.3, b=1., c=0.8, units=galactic)

        # t1 = time.time()
        # for i in range(nrepeat):  # ~10 ms per call
        #     rewinder_likelihood(ll, -1., 6000,
        #                         potential.c_instance,
        #                         self.prog, self.stars,
        #                         2.5E6, 0.,
        #                         1., self.betas, -0.3,
        #                         True)

        # t = (time.time() - t1) / float(nrepeat)

        # print("{} ms per call".format(t*1000.))

    #@pytest.mark.skipif(True, reason="derp.")
    def test_profile(self):
        # Have to turn on cython profiling for this to work
        import pstats, cProfile

        ll = np.zeros((6000,self.nstars), dtype=float)
        potential = sp.SphericalNFWPotential(v_c=0.5*np.sqrt(np.log(2)-0.5), r_s=20.0, units=galactic)

        the_str = """for i in range(100): rewinder_likelihood(ll, -1., 6000,
                                            potential.c_instance,
                                            self.prog, self.stars,
                                            2.5E6, 0.,
                                            1., self.betas, -0.3,
                                            True)"""

        cProfile.runctx(the_str, globals(), locals(), "pro.prof")

        s = pstats.Stats("pro.prof")
        s.strip_dirs().sort_stats("cumulative").print_stats(50)
        # s.strip_dirs().sort_stats("time").print_stats(10)

    def test_time_many_stars(self):

        stars = np.repeat(self.stars, 64, axis=0)
        nstars = len(stars)
        betas = np.repeat(self.betas, 64)

        ll = np.zeros((6000,nstars), dtype=float)

        print("Spherical w/ {} stars:".format(nstars))
        potential = sp.LeeSutoTriaxialNFWPotential(v_h=0.5, r_h=20.0,
                                           a=1., b=1., c=1., units=galactic)

        t1 = time.time()
        rewinder_likelihood(ll, -1., 6000,
                            potential.c_instance,
                            self.prog, stars,
                            2.5E6, 0.,
                            1., betas, -0.3,
                            True)
        t = (time.time() - t1)
        print("{} sec per call".format(t))

class TestObsSimple(object):

    def setup(self):
        # read stuff from text file
        stardata = np.genfromtxt(os.path.join(this_path, "obs_stars.txt"), names=True)
        progdata = np.genfromtxt(os.path.join(this_path, "obs_prog.txt"), names=True)

        self.nstars = 4
        self.stars = np.vstack((stardata['x'],stardata['y'],stardata['z'],
                                stardata['vx'],stardata['vy'],stardata['vz'])).T.copy()[:self.nstars]

        self.prog = np.vstack((progdata['x'],progdata['y'],progdata['z'],
                               progdata['vx'],progdata['vy'],progdata['vz'])).T.copy()

        self.betas = stardata['tail'].copy()[:self.nstars]

        path = os.path.abspath(os.path.join(this_path, "test6.yml"))
        self.model = Rewinder.from_config(path)

    def test_call(self):
        true_progdata = np.genfromtxt(os.path.join(this_path, "true_prog.txt"), names=True)
        true_prog_pos = np.array([true_progdata[name] for name in heliocentric_names])
        parameter_values = dict(potential=dict(v_c=0.5*np.sqrt(np.log(2)-0.5), r_s=20.),
                                progenitor=dict(m0=2.5E6, **dict(zip(heliocentric_names,true_prog_pos))))

        # for n in [128,256,512,1024,2048,4096]:
        for n in [512,1024,2048,4096]:
            print("running {}...".format(n))
            self.model.nsamples = n
            self.model._ln_likelihood_tmp = np.zeros((self.model.nsteps, self.model.nsamples))  # HACK!!

            vcs = np.linspace(0.45,0.55,25)*np.sqrt(np.log(2)-0.5)
            ls = np.zeros_like(vcs)
            for i,v_c in enumerate(vcs):
                pv = parameter_values.copy()
                pv['potential']['v_c'] = v_c
                p = self.model.vectorize(parameter_values)
                ls[i] = self.model(p)

            plt.clf()
            plt.title("Num. of samples: {}".format(n))
            plt.plot(vcs, ls)
            plt.savefig(os.path.join(output_path, "{}.png".format(n)))

class TestConfig(object):

    def setup(self):
        true_progdata = np.genfromtxt(os.path.join(this_path, "true_prog.txt"), names=True)
        true_prog_pos = np.array([true_progdata[name] for name in heliocentric_names])

        self.parameter_values = dict(potential=dict(v_c=0.5*np.sqrt(np.log(2.)-0.5), r_s=20.),
                                     progenitor=dict(m0=2.5E6, **dict(zip(heliocentric_names,true_prog_pos))),
                                     hyper=dict(alpha=1.25, theta=-0.3))
        self.parameter_sigmas = dict(potential=dict(v_c=0.01, r_s=1.),
                                     progenitor=dict(m0=1E5,l=1E-8,b=1E-8,d=1E-1,mul=1E-4,mub=1E-4,vr=1E-3),
                                     hyper=dict(alpha=0.1, theta=0.01))

    def do_the_mcmc(self, sampler, p0, p0_sigma, truth):
        n = 25

        print("Model value at truth: {}".format(sampler.lnprobfn(truth)))
        for pp in p0:
            if np.any(~np.isfinite(sampler.lnprobfn(pp))):
                raise ValueError("Model returned -inf for initial position! {}".format(pp))

        # burn in
        sampler.run_inference(p0, n)
        best_pos = sampler.flatchain[sampler.flatlnprobability.argmax()]
        sampler.reset()

        # restart walkers from best position, burn again
        new_pos = np.random.normal(best_pos, p0_sigma/2.,
                                   size=(sampler.nwalkers, p0.shape[1]))
        sampler.run_inference(new_pos, n)
        pos = sampler.chain[:,-1]
        sampler.reset()

        # run for inference steps
        sampler.run_inference(pos, n)

        return sampler

    def make_plots(self, sampler, p0, truth, test_name):
        plot_path = os.path.join(output_path, test_name)
        if not os.path.exists(plot_path):
            os.mkdir(plot_path)

        for i in range(len(truth)):
            print("Plotting param {}".format(i))
            plt.clf()
            for chain in sampler.chain[...,i]:
                plt.plot(chain, marker=None, drawstyle='steps', alpha=0.2, color='k')

            for pp in p0[:,i]:
                plt.axhline(pp, alpha=0.2, color='r')

            plt.axhline(truth[i], alpha=0.7, color='g')
            plt.savefig(os.path.join(plot_path, "param_{}.png".format(i)))

    # ---------------------------------------------------------------------------------------------

    def _main_test(self, test_name, initialize_far=False):
        path = os.path.abspath(os.path.join(this_path, "{}.yml".format(test_name)))
        model = Rewinder.from_config(path)
        sampler = RewinderSampler(model, nwalkers=64)

        truth = model.vectorize(self.parameter_values)
        p0_sigma = model.vectorize(self.parameter_sigmas)

        if initialize_far:
            p0 = np.random.normal(truth + 2*p0_sigma, p0_sigma, size=(sampler.nwalkers, len(truth)))
        else:
            p0 = np.random.normal(truth, p0_sigma, size=(sampler.nwalkers, len(truth)))
        sampler = self.do_the_mcmc(sampler, p0, p0_sigma, truth)
        self.make_plots(sampler, p0, truth, test_name)

    def test1(self):
        """ Test 1:
                No uncertainties, fix alpha, fix theta, only sample over v_c, r_s
        """
        self._main_test('test1')

    def test1pt5(self):
        """ Test 1.5:
                Same as Test 1 but start from far from truth initial conditions
        """
        self._main_test('test1.5', initialize_far=True)

    def test2(self):
        """ Test 2:
                No uncertainties, sample over v_c, r_s, alpha, theta, mass
        """
        self._main_test('test2')

    def test3(self):
        """ Test 3:
                Uncertainties on progenitor, no uncertainties in stars
                sample over v_c, r_s
        """
        self._main_test('test3')

    def test4(self):
        """ Test 4:
                Uncertainties on progenitor, no uncertainties in stars
                sample over v_c, r_s, alpha, theta
        """
        self._main_test('test4')

    def test5(self):
        """ Test 5:
                Uncertainties on progenitor, no uncertainties in stars
                sample over v_c, r_s, missing proper motions for progenitor
        """
        self._main_test('test5')

    def test6(self):
        """ Test 6:
                Uncertainties in stars, no uncertainties in progenitor,
                sample over v_c, r_s
        """
        self._main_test('test6')

