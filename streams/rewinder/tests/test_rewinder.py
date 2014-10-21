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
from .. import Rewinder
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
                                stardata['vx'],stardata['vy'],stardata['vz'])).T.copy()

        self.prog = np.vstack((progdata['x'],progdata['y'],progdata['z'],
                               progdata['vx'],progdata['vy'],progdata['vz'])).T.copy()

        self.betas = stardata['tail'].copy()

    def test_call(self):
        potential = sp.LeeSutoNFWPotential(v_h=0.5, r_h=20., a=1., b=1., c=1., units=galactic)
        ll = rewinder_likelihood(-1., 6000.,
                                 potential.c_instance,
                                 self.prog, self.stars,
                                 2.5E6, 0.,
                                 1., self.betas, -0.3)
        print(ll.shape)

    def test_time(self):
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

        print("\n\n\n")
        print("Triaxial, rotated:")
        potential = sp.LeeSutoNFWPotential(v_h=0.5, r_h=20.0, phi=0.2,
                                           a=1.3, b=1., c=1., units=galactic)

        t1 = time.time()
        for i in range(nrepeat):  # ~10 ms per call
            ll = rewinder_likelihood(-1., 6000,
                                     potential.c_instance,
                                     self.prog, self.stars,
                                     2.5E6, 0.,
                                     1., self.betas, -0.3)

        t = (time.time() - t1) / float(nrepeat)

        print("{} ms per call".format(t*1000.))

def test_from_config():
    path = os.path.abspath(os.path.join(this_path, "../../../config/test.yml"))
    rw = Rewinder.from_config(path)

    rw([0.5, 20., 2.5E6, 0., 1.2, -0.3])

    vhs = np.linspace(0.3,0.7,55)
    lls = np.zeros_like(vhs)
    for i,vh in enumerate(vhs):
        lls[i] = rw([vh, 20, 2.5E6, 0., 1.2, -0.3])

    fig,axes = plt.subplots(1,2,figsize=(10,5))
    axes[0].plot(vhs, lls, marker='o')
    axes[1].plot(vhs, np.exp(lls - lls.max()), marker='o')
    fig.savefig(os.path.join(plot_path, "vary_vh.png"))

    # --------------------------------------------------------------

    rhs = np.linspace(10.,40.,55)
    lls = np.zeros_like(rhs)
    for i,rh in enumerate(rhs):
        lls[i] = rw([0.5, rh, 2.5E6, 0., 1.2, -0.3])

    fig,axes = plt.subplots(1,2,figsize=(10,5))
    axes[0].plot(rhs, lls, marker='o')
    axes[1].plot(rhs, np.exp(lls - lls.max()), marker='o')
    fig.savefig(os.path.join(plot_path, "vary_rh.png"))

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








# -----------------------------------------------------------------------------
# Old
# def test_config():
#     rw = Rewinder.from_config("/Users/adrian/projects/streams/config/test.yml")
#     for p in rw._walk():
#         print(p)

#     p0 = rw.sample_p0(1000)
#     p = p0[15]
#     print(p)
#     # print(rw(p))

#     vals = np.linspace(1.28, 1.48, 25)
#     #vals = np.linspace(1., 1.72, 7)
#     ll = []
#     for val in vals:
#         try:
#             ll.append(rw(np.array([val] + list(p[1:]))))
#         except ValueError:
#             ll.append(np.nan)

#     ll = np.array(ll)

#     fig,axes = plt.subplots(2,1)
#     axes[0].plot(vals, ll)
#     axes[0].axvline(1.36)
#     axes[1].plot(vals, np.exp(ll-np.max(ll)))
#     axes[1].axvline(1.36)
#     axes[1].set_xlim(0.8,1.9)
#     fig.savefig("/Users/adrian/Downloads/derp_{}.png".format(rw.K))
