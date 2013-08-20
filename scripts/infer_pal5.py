# coding: utf-8

""" In this module, I'll try to infer the Palomar 5 potential that Andreas
    used in an Nbody simulation of the stream.
"""

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys

# Third-party
import matplotlib.pyplot as plt
import numpy as np
import astropy.units as u

# project
from streams.io.pal5 import particles_today, satellite_today, time
from streams.inference.pal5 import ln_prior, ln_likelihood, ln_posterior, objective
from streams.potential.pal5 import Palomar5, true_params, _true_params
from streams.integrate.satellite_particles import SatelliteParticleIntegrator
from streams.inference import relative_normalized_coordinates
from streams.observation.gaia import add_uncertainties_to_particles

np.random.seed(44)
true_particles = particles_today(N=100) # , expr="(Pcol>-1) & (Pcol<8) & (abs(Lmflag)==1)")
satellite = satellite_today()
t1,t2 = time()
resolution = 4.
#err_particles = add_uncertainties_to_particles(true_particles)

def test_left_right():
    l = ln_likelihood([], [], true_particles, satellite, t1, t2, resolution)
    l_L = ln_likelihood([0.7], ['qz'], true_particles, satellite, t1, t2, resolution)
    l_R = ln_likelihood([0.9], ['qz'], true_particles, satellite, t1, t2, resolution)
    print(l_L, l, l_R)

def test_likelihood(fn, ps, frac_bounds=(0.8, 1.2), Nbins=21):
    frac_range = np.linspace(frac_bounds[0],frac_bounds[1],Nbins)
    
    fig = plt.figure(figsize=(12,6))
    for ii,param in enumerate(['qz', 'm']):
        vals = frac_range*true_params[param]
        ls = []
        for val in vals:
            l = fn([val], [param], ps, satellite, t1, t2, resolution)
            ls.append(l)
        
        plt.subplot(1,2,ii+1)
        plt.plot(vals, np.array(ls))
        plt.axvline(_true_params[param])
    
    return fig

fig = test_likelihood(ln_likelihood, true_particles, frac_bounds=(0.6,1.4))
plt.show()