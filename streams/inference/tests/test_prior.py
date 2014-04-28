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
import matplotlib.pyplot as plt

from ..prior import *

plot_path = "plots/tests/inference"
if not os.path.exists(plot_path):
    os.makedirs(plot_path)

def test_uniform():

    prior = LogUniformPrior(0.,1.)
    assert prior.shape == (1,)
    assert prior(0.5) == 0.
    assert prior(1.5) == -np.inf
    assert prior.sample(size=10).shape == (10,1)

    prior = LogUniformPrior([0.,1],[1.,2])
    assert prior.shape == (2,)
    assert np.all(prior([0.5,1.5]) == 0.)
    assert np.all(prior([1.5,0.5]) == -np.inf)
    assert prior.sample(size=10).shape == (10,2)

    prior = LogUniformPrior([0.,1]*u.kpc,[1.,2]*u.kpc)
    assert prior.shape == (2,)
    assert np.all(prior([0.5,1.5]*u.kpc) == 0.)
    assert np.all(prior([1.5,0.5]*u.kpc) == -np.inf)
    assert np.all(prior([1.5,0.5]*u.km) == -np.inf)
    assert prior.sample(size=10).shape == (10,2)

def test_normal():

    prior = LogNormalPrior(mean=0., stddev=0.5)
    print(prior(0.2))
    assert prior.shape == (1,1)
    assert prior.cov.shape == (1,1,1)
    assert prior._norm.shape == (1,)

    # 2 1D Gaussians
    prior = LogNormalPrior(mean=np.array([0.,1.]),
                           stddev=np.array([0.5,0.5]))
    print(prior([0.2, 1.2]))
    assert prior.shape == (2,1)
    assert prior.cov.shape == (2,1,1)
    assert prior._norm.shape == (2,)

    # 1 2D Gaussian
    prior = LogNormalPrior(mean=np.array([[0.,1.]]),
                           stddev=np.array([[0.5,0.5]]))
    print(prior([[0.2, 1.2]]))
    assert prior.shape == (1,2)
    assert prior.cov.shape == (1,2,2)
    assert prior._norm.shape == (1,)

    # 1 2D Gaussian
    prior = LogNormalPrior(mean=np.array([[0.,1.]]),
                           cov=np.array([[0.25,0.],[0.,0.25]]))
    print(prior([[0.2, 1.2]]))
    assert prior.shape == (1,2)
    assert prior.cov.shape == (1,2,2)
    assert prior.stddev is None
    assert prior._norm.shape == (1,)
