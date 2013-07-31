# coding: utf-8

""" Here I'll try using an optimizer to infer the parameters """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys

# Third-party
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from scipy.optimize import anneal, fmin_l_bfgs_b

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
param_ranges = dict(q1=(1.2,1.5),
                    qz=(1.2,1.5),
                    phi=(1.6,1.8))

#posterior = lambda *args,**kwargs: -np.exp(ln_posterior(*args,**kwargs))
posterior = lambda *args,**kwargs: -ln_posterior(*args,**kwargs)

def anneal_lm10(params):
    """ anneal() returns:
        
        xmin : ndarray
            Point giving smallest value found.
        Jmin : float
            Minimum value of function found.
        T : float
            Final temperature.
        feval : int
            Number of function evaluations.
        iters : int
            Number of cooling iterations.
        accept : int
            Number of tests accepted.
        retval : int
            Flag indicating stopping condition:
            Want 0, I think?
    """
    ret = anneal(posterior,
                 x0=[param_initial[p] for p in params],
                 args=(params, particles, satellite, t1, t2, 3.),
                 T0=1E4,
                 Tf=1.,
                 learn_rate=5.,
                 maxiter=100,
                 full_output=True,
                 lower=[param_ranges[p][0] for p in params],
                 upper=[param_ranges[p][1] for p in params])
    
    xmin = ret[0]
    retval = ret[-1]
    print(ret)
    return xmin, retval == 0

def minimize_lm10(params):
    ret = fmin_l_bfgs_b(posterior,
                        x0=[param_initial[p] for p in params],
                        args=(params, particles, satellite, t1, t2, 3.),
                        bounds=[param_ranges[p] for p in params],
                        approx_grad=True,
                        epsilon=1E-5,
                        factr=10)
    xmin = ret[0]
    retval = ret[-1]           
    return xmin, retval

def plot_objective(params, xmin, Nbins=25, fname="objective"):
    fig,axes = plt.subplots(len(params),1,figsize=(8,5*len(params)))
    if len(params) == 1:
        axes = [axes]
        xmin = [xmin]

    for ii,p in enumerate(params):
        p_vals = np.linspace(param_ranges[p][0],
                             param_ranges[p][1],
                             Nbins)
    
        post_vals = []
        for val in p_vals:
            post_vals.append(posterior([val], [p], particles, satellite, t1, t2, 3.))
    
        axes[ii].plot(p_vals, post_vals)
        axes[ii].axvline(xmin[ii], color='r')
        
        if hasattr(true_params[p], 'unit'):
            axes[ii].axvline(true_params[p].value, color='g', linestyle='--')
        else:
            axes[ii].axvline(true_params[p], color='g', linestyle='--')

    plt.savefig("plots/{0}.png".format(fname))

def vary_q1qz():
    
    Nbins = 25
    post_vals = []
    for q1 in np.linspace(1.3,1.5,Nbins):
        for qz in np.linspace(1.3,1.5,Nbins):
            post_vals.append(posterior([q1, qz], ['q1', 'qz'], particles, satellite, t1, t2, 3.))
    
    post_vals = np.array(post_vals)
    post_vals = post_vals.reshape(Nbins,Nbins)
    
    plt.clf()
    plt.imshow(post_vals, extent=[1.3,1.5,1.3,1.5], interpolation='nearest', cmap=cm.Greys)
    plt.axvline(true_params['qz'])
    plt.axhline(true_params['q1'])
    plt.savefig("plots/vary_q1qz.png")

if __name__ == '__main__':
    
    #vary_q1qz()
    #sys.exit(0)

    params = ['q1', 'qz']
    
    #print("starting annealing")
    #xmin, converged = anneal_lm10(params)
    
    print("starting minimizer")
    xmin, converged = minimize_lm10(params)
    
    print("Found minimum at: {0}".format(xmin))
    print("Converged: {0}".format(converged))

    plot_objective(params, xmin, fname="minimize")
