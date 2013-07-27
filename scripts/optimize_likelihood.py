# coding: utf-8

""" Here I'll try using an optimizer to infer the parameters """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys

# Third-party
import numpy as np
from scipy.optimize import fmin_l_bfgs_b

# Project
from streams.io.lm10 import particles_today, satellite_today, time
from streams.inference import generalized_variance
from streams.inference.lm10 import ln_likelihood, ln_posterior

t1,t2 = time()
particles = particles_today(N=100, expr="(Pcol>0) & (Pcol<7) & (abs(Lmflag)==1)")
satellite = satellite_today()

'''
print("read in particles, starting posterior calculation")

for val in np.linspace(1.2,1.5,10):
    print(val, ln_posterior([val], ['qz'], particles, satellite, t1, t2, 3.))
'''

print("attempting minimization")
x0 = fmin_l_bfgs_b(lambda *args,**kwargs: -ln_posterior(*args,**kwargs),
                  x0=[1.21, 1.21],
                  args=(['q1', 'qz'], particles, satellite, t1, t2, 3.),
                  bounds=[(1.2,1.4),(1.2,1.4)],
                  maxiter=3,
                  factr=1E5,
                  approx_grad=True,
                  epsilon=1E-4)

print(x0)