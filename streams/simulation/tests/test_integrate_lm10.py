# coding: utf-8
"""
    Test the Cython integrate code
"""

from __future__ import absolute_import, unicode_literals, division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys
import glob
import time

# Third-party
import numpy as np
import pytest
import astropy.units as u
import matplotlib.pyplot as plt

from .._integrate_lm10 import lm10_acceleration, leapfrog_lm10
from ...potential.lm10 import LawMajewski2010, halo_params
from ...integrate import leapfrog

def test_cython_vs_python1():
    r = np.random.random((100,3))
    
    Ntest = 10
    
    a = time.time()
    for ii in range(Ntest):
        acc1 = lm10_acceleration(r, len(r), 1.38, 1.36, 
                                1.692969, 0.124625659009).T
    cython = (time.time() - a) / float(Ntest)
    
    lm10 = LawMajewski2010()
    
    a = time.time()
    for ii in range(Ntest):
        acc2 = lm10.acceleration_at(r)
    pure_python = (time.time() - a) / float(Ntest)
    
    print("cython: {0}".format(cython))
    print("pure python: {0}".format(pure_python))
    
    assert cython < pure_python
    
def test_cython_vs_python2():
    r = np.random.random((100,3))
    v = np.random.random((100,3))
    t = np.arange(0, 7000, 10.)
    
    Ntest = 10
    
    a = time.time()
    for ii in range(Ntest):
        tt1,rr1,vv1 = leapfrog_lm10(r, v, t, 1.38, 1.36, 1.692969, 0.124625659009)
    cython = (time.time() - a) / Ntest
    
    lm10 = LawMajewski2010()
                              
    a = time.time()
    for ii in range(Ntest):
        tt2,rr2,vv2 = leapfrog(lm10.acceleration_at, r, v, t)
    pure_python = (time.time() - a) / Ntest
    
    plt.plot(tt1, rr1[:,0,0], color='r')
    plt.plot(tt1, rr2[:,0,0], color='b')
    plt.savefig("/var/www/scratch/cython_test.png")
    
    print("cython: {0}".format(cython))
    print("pure python: {0}".format(pure_python))
    assert cython < pure_python
    