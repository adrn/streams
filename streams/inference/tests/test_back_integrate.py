# coding: utf-8
""" Test the likelihood optimization for LM10 and Pal5 """

from __future__ import absolute_import, unicode_literals, division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys
import copy
import time as pytime

# Third-party
import numpy as np
import pytest
import astropy.units as u
import matplotlib.pyplot as plt

from ...potential.lm10 import LawMajewski2010
from ...io.sgr import mass_selector

particles_today, satellite_today, time = mass_selector("2.5e7")
satellite = satellite_today()
true_particles = particles_today(N=N, expr="(tub!=0)")
t1,t2 = time()

plot_path = "plots/tests/inference/back_integrate"
if not os.path.exists(plot_path):
    os.makedirs(plot_path)

def test_likelihood():
    p = [1.38, 1.36] + list(np.ravel(true_particles._X)) + [300, 100, 1000]
    p_names = ['q1', 'qz']

    data_errors =

    back_integrate_likelihood(p, potential_params, satellite,
                              data, data_errors,
                              Potential, t1, t2)