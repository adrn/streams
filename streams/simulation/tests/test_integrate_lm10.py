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
from ...potential import LawMajewski2010
from ...integrate import leapfrog

def test_cython_vs_python1():
    r = np.random.random((100,3))
    
    a = time.time()
    for ii in range(10000):
        lm10_acceleration(r, 2, 1.6, 1.6, 1.69, 0.121)        
    cython = (time.time() - a) / 10000.
    
    lm10 = LawMajewski2010()
    
    a = time.time()
    for ii in range(10000):
        lm10.acceleration_at(r)    
    pure_python = (time.time() - a) / 10000.
    
    assert cython < pure_python
    
def test_cython_vs_python2():
    r = np.random.random((100,3))
    v = np.random.random((100,3))
    t = np.arange(0, 7000, 10.)
    
    a = time.time()
    for ii in range(10):
        leapfrog_lm10(r, v, 1.6, 1.6, 1.69, 0.121, t=t)
    cython = (time.time() - a) / 10.
    
    lm10 = LawMajewski2010()
                              
    a = time.time()
    for ii in range(10):
        leapfrog(lm10.acceleration_at, r, v, t)
    pure_python = (time.time() - a) / 10.
    
    print(cython, pure_python)
    #assert cython < pure_python
    