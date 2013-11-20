# coding: utf-8
"""
    Test the Palomar5 potential
"""

from __future__ import absolute_import, unicode_literals, division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

import os
import pytest
import numpy as np
from astropy.constants.si import G
import astropy.units as u
import matplotlib.pyplot as plt

from ..pal5 import Palomar5

plot_path = "plots/tests/potential"
if not os.path.exists(plot_path):
    os.makedirs(plot_path)

def test_simple():
    p = Palomar5()
    p = Palomar5(m=1.21*u.M_sun)

def test_plot():
    potential = Palomar5()
    r = ([0.,0.,0.]*u.kpc).reshape(3,1)
    pot_val = potential.value_at(r)
    acc_val = potential.acceleration_at(r)

    grid = np.linspace(-20.,20, 50)*u.kpc
    fig,axes = potential.plot(grid=grid, ndim=3)
    fig.savefig(os.path.join(plot_path, "pal5.png"))