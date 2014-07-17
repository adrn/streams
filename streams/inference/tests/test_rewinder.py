# coding: utf-8

""" Test the Rewinder model subclass """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys
import logging

# Third-party
from astropy import log as logger
import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np

# Project
from ..rewinder import Rewinder

logger.setLevel(logging.DEBUG)

def test_config():
    rw = Rewinder.from_config("/Users/adrian/projects/streams/config/test.yml")
    for p in rw._walk():
        print(p)

    p0 = rw.sample_priors(24)[0]
    print(rw(np.array([1.36, 1.5])))

    vals = np.linspace(1.26, 1.46, 25)
    #vals = np.linspace(1., 1.72, 7)
    ll = []
    for val in vals:
        try:
            ll.append(rw(np.array([val,1.6])))
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