# coding: utf-8
""" Test the likelihood optimization for LM10 and Pal5 """

from __future__ import absolute_import, unicode_literals, division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys
import time as pytime

# Third-party
import numpy as np
import pytest
import astropy.units as u
import matplotlib.pyplot as plt
from matplotlib import cm

plot_path = "plots/tests/inference"
if not os.path.exists(plot_path):
    os.makedirs(plot_path)

resolution = 4.

Nparticles = 500

class TestLM10(object):
    
    def setup_class(self):
        from ...io.lm10 import particles_today, satellite_today, time
        from ...inference.lm10 import ln_posterior
        from ...potential.lm10 import true_params, _true_params
        
        self.ln_posterior = lambda self,*args: ln_posterior(*args)
        self.true_params = true_params
        self._true_params = _true_params
        
        np.random.seed(42)
        self.t1,self.t2 = time()
        self.satellite = satellite_today()
        self.particles = particles_today(N=Nparticles, expr="(Pcol > -1) & (abs(Lmflag)==1) & (dist<75)")

    def test_time_posterior(self):
        N = 10
        a = pytime.time()
        for ii in range(N):
            self.ln_posterior([], [], self.particles, self.satellite, 
                              self.t1, self.t2, resolution)
        print("LM10: {0} seconds per ln_posterior call".format(float(pytime.time() - a) / N))
    
    def test_posterior_shape(self, frac_bounds=(0.8,1.2), Nbins=15):
        for p_name in self._true_params.keys():
            true_p = self._true_params[p_name]
            vals = np.linspace(frac_bounds[0], frac_bounds[1], Nbins) * true_p
            posterior_shape = []
            for val in vals:
                post_val = self.ln_posterior([val], [p_name], self.particles, 
                                             self.satellite, self.t1, self.t2, 
                                             resolution)
                posterior_shape.append(post_val)
            
            posterior_shape = np.array(posterior_shape)
            
            fig = plt.figure(figsize=(6,6))
            ax = fig.add_subplot(111)
            ax.plot(vals, posterior_shape, lw=2.)
            ax.axvline(self._true_params[p_name], color='b', linestyle='--')
            fig.savefig(os.path.join(plot_path, "lm10_{0}.png".format(p_name)))

class TestPal5(object):
    
    def setup_class(self):
        from ...io.pal5 import particles_today, satellite_today, time
        from ...inference.pal5 import ln_posterior
        from ...potential.pal5 import true_params, _true_params
        
        self.ln_posterior = lambda self,*args: ln_posterior(*args)
        self.true_params = true_params
        self._true_params = _true_params
        
        np.random.seed(42)
        self.t1,self.t2 = time()
        self.satellite = satellite_today()
        self.particles = particles_today(N=Nparticles)

    def test_time_posterior(self):
        N = 10
        a = pytime.time()
        for ii in range(N):
            self.ln_posterior([], [], self.particles, self.satellite, 
                              self.t1, self.t2, resolution)
        print("Pal5: {0} seconds per ln_posterior call".format(float(pytime.time() - a) / N))
    
    def test_posterior_shape(self, frac_bounds=(0.5,1.5), Nbins=15):
        for p_name in self._true_params.keys():
            true_p = self._true_params[p_name]
            vals = np.linspace(frac_bounds[0], frac_bounds[1], Nbins) * true_p
            posterior_shape = []
            for val in vals:
                post_val = self.ln_posterior([val], [p_name], self.particles, 
                                             self.satellite, self.t1, self.t2, 
                                             resolution)
                posterior_shape.append(post_val)
            
            posterior_shape = np.array(posterior_shape)
            
            fig = plt.figure(figsize=(6,6))
            ax = fig.add_subplot(111)
            ax.plot(vals, posterior_shape, lw=2.)
            ax.axvline(self._true_params[p_name], color='b', linestyle='--')
            fig.savefig(os.path.join(plot_path, "pal5_{0}.png".format(p_name)))
        