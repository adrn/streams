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
from ..rewinder import Rewinder
from ..rewinder_likelihood import rewinder_likelihood

logger.setLevel(logging.DEBUG)

def test_config():
    rw = Rewinder.from_config("/Users/adrian/projects/streams/config/test.yml")
    for p in rw._walk():
        print(p)

    p0 = rw.sample_p0(1000)
    p = p0[15]
    print(p)
    # print(rw(p))

    vals = np.linspace(1.28, 1.48, 25)
    #vals = np.linspace(1., 1.72, 7)
    ll = []
    for val in vals:
        try:
            ll.append(rw(np.array([val] + list(p[1:]))))
        except ValueError:
            ll.append(np.nan)

    ll = np.array(ll)

    fig,axes = plt.subplots(2,1)
    axes[0].plot(vals, ll)
    axes[0].axvline(1.36)
    axes[1].plot(vals, np.exp(ll-np.max(ll)))
    axes[1].axvline(1.36)
    axes[1].set_xlim(0.8,1.9)
    fig.savefig("/Users/adrian/Downloads/derp_{}.png".format(rw.K))

# -----------------------------------------------------------------------------
# BELOW HERE IS FOR TIMING
#
stars = np.array([[ -4.52517380e+01,   3.52057856e-02,   3.85155128e-01,
         -3.60541472e-03,  -1.60517822e-01,   3.75658297e-04],
       [ -3.35736131e+01,  -1.66575894e+01,  -3.31192536e-01,
          1.54695658e-01,  -1.24574615e-01,   1.17627889e-04],
       [ -4.42072680e+00,  -2.26940756e+01,   4.47393070e-02,
          2.98548759e-01,   2.03551061e-02,   1.11108105e-03],
       [ -4.20131543e+01,   1.01839437e+01,  -1.54012154e-01,
         -9.16548063e-02,  -1.50766539e-01,   3.43455729e-03],
       [ -4.39711120e+01,   5.20381700e+00,  -2.69321040e-01,
         -5.03253059e-02,  -1.59999139e-01,   1.51061691e-03],
       [ -2.31483448e+01,  -2.15197972e+01,  -2.55629410e-01,
          2.24449936e-01,  -8.46415383e-02,   8.27580694e-04],
       [ -2.66711327e+01,  -1.99333079e+01,   1.35061194e-01,
          2.05619452e-01,  -9.81571549e-02,  -1.91566099e-03],
       [ -4.29474067e+01,  -8.73160580e+00,   7.30469380e-02,
          6.94081440e-02,  -1.55045771e-01,   2.54281489e-04],
       [ -4.43624120e+01,   3.58385865e+00,  -5.03670180e-01,
         -3.65884063e-02,  -1.60609418e-01,  -1.43082952e-03],
       [ -1.60539339e+00,  -2.29184066e+01,   2.47143274e-02,
          2.97304133e-01,   4.43258216e-02,  -1.92689255e-04],
       [ -5.82452200e+00,  -2.33314173e+01,   9.79736510e-02,
          2.93081463e-01,   1.45058632e-02,   1.31818370e-03],
       [ -4.48759180e+01,   2.76823981e+00,  -2.48013508e-01,
         -2.84520476e-02,  -1.59659443e-01,   2.39328835e-04],
       [ -4.42869900e+01,   2.66930111e+00,  -1.99090989e-01,
         -3.65329440e-02,  -1.62287229e-01,  -8.19253266e-04],
       [ -4.19371948e+01,   1.33556409e+01,  -3.68318951e-01,
         -1.02015301e-01,  -1.41594173e-01,  -2.11643635e-03],
       [  6.80858130e+00,  -2.03811013e+01,  -9.97448640e-02,
          2.94886372e-01,   9.79882773e-02,   2.85553863e-03],
       [ -4.48442700e+01,  -5.03297370e+00,  -5.96755720e-01,
          3.95622450e-02,  -1.57395081e-01,   6.73886154e-04]])

betas = np.array([-1.,  1.,  1., -1., -1.,  1.,  1., -1.,
                  -1.,  1.,  1., -1., -1., -1.,  1., -1.])

prog = np.array([[ -3.66844266e+01,  -1.51100753e+01,  -7.44185090e-03,
          1.34803336e-01,  -1.35268320e-01,   4.46802528e-05]])

potential = sp.LeeSutoNFWPotential(v_h=0.5, r_h=20.0,
                                   a=1., b=1., c=1., units=galactic)

def test_time():
    nrepeat = 1000
    t1 = time.time()
    for i in range(nrepeat):  # ~10 ms per call
        ll = rewinder_likelihood(6000., 0., -1.,
                                 potential.c_instance,
                                 prog, stars,
                                 2.5E6, 0.,
                                 1., betas, -0.3)

    t = (time.time() - t1) / float(nrepeat)

    print("{} ms per call".format(t*1000.))
