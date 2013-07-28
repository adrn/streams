# coding: utf-8

""" Here I'll try using an optimizer to infer the parameters """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys

# Third-party
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import anneal

# Project
from streams.io.lm10 import particles_today, satellite_today, time
from streams.inference import generalized_variance
from streams.inference.lm10 import ln_likelihood, ln_posterior
from streams.potential.lm10 import true_params

np.random.seed(42)
t1,t2 = time()
particles = particles_today(N=100, expr="(Pcol>0) & (Pcol<7) & (abs(Lmflag)==1)")
satellite = satellite_today()

param_initial = dict(q1=1.4,
                     qz=1.4,
                     phi=1.3)
param_ranges = dict(q1=(1.25,1.45),
                    qz=(1.25,1.45),
                    phi=(1.6,1.8))

posterior = lambda *args,**kwargs: -np.exp(ln_posterior(*args,**kwargs))

def anneal_lm10(params):
    ret = anneal(posterior,
                 x0=[param_initial[p] for p in params],
                 args=(params, particles, satellite, t1, t2, 3.))
    return ret

def minimize_lm10(params):
    ret = fmin_l_bfgs_b(posterior,
                        x0=[param_initial[p] for p in params],
                        args=(params, particles, satellite, t1, t2, 3.),
                        bounds=[param_ranges[p] for p in params],
                        approx_grad=True,
                        epsilon=1E-4,
                        factr=1E8)
    return ret

def plot_objective(params, x_min, Nbins=25, fname="objective"):
    fig,axes = plt.subplots(len(params),1,figsize=(8,5*len(params)))
    if len(params) == 1:
        axes = [axes]
        x0 = [x0]

    for ii,p in enumerate(params):
        p_vals = np.linspace(param_ranges[p][0],
                             param_ranges[p][1],
                             Nbins)
    
        post_vals = []
        for val in p_vals:
            post_vals.append(posterior([val], [p], particles, satellite, t1, t2, 3.))
    
        axes[ii].plot(p_vals, post_vals)
        axes[ii].axvline(x0[ii], color='r')
        axes[ii].axvline(true_params[p], color='g', linestyle='--')

    plt.savefig("plots/{0}.png".format(fname))

if __name__ == '__main__':
    params = ['qz']
    
    anneal_lm10(params)
