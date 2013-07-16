# coding: utf-8
""" """

from __future__ import absolute_import, unicode_literals, division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys
import time

# Third-party
import numpy as np
import pytest
import astropy.units as u
import matplotlib.pyplot as plt
from matplotlib import cm

from ..inferpotential import infer_potential
from ..lm10 import ln_posterior

plot_path = "plots/tests/inference"
if not os.path.exists(plot_path):
    os.makedirs(plot_path)

def test_optimize_potential():
    """ Try inferring the halo parameters with a simple optimization method """
    
    from scipy.optimize import fmin
    
    param_names = ['q1', 'qz', 'v_halo', 'phi']
    p0 = [np.random.uniform(1., 2.),
          np.random.uniform(1., 2.),
          (np.random.uniform(100, 150.)*u.km/u.s).to(u.kpc/u.Myr).value,
          np.random.uniform(1., 2.)]
          
    t,satellite,particles = read_lm10(N=100, dt=10.)
    args = (param_names, particles, satellite, t)
    p_opt = fmin(lambda *args,**kwargs: -ln_posterior(*args,**kwargs), 
                     x0=p0, 
                     ftol=1E3,
                     args=args,
                     disp=True,
                     callback=lambda xk: print(ln_posterior(xk, *args)))
    
    print(p_opt)