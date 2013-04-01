# coding: utf-8

""" Test back_integrate, and that the statistic is convex. """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys

# Third-party
import pytest
import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt
from matplotlib import cm

# Project
from streams.simulation import back_integrate, Particle, \
                               TestParticleSimulation, minimum_distance_matrix
from streams.data import SgrSnapshot, SgrCen
from streams.potential import *
from streams.potential.lm10 import halo_params as true_halo_params
from streams.potential.lm10 import param_ranges

Nsteps = 20
np.random.seed(42)
sgr_cen = SgrCen()
sgr_snap = SgrSnapshot(num=1000)

# Get timestep information from SGR_CEN
t1 = min(sgr_cen.t)
t2 = max(sgr_cen.t)
dt = sgr_cen.dt[0]*10

# Interpolate SgrCen data onto new times
ts = np.arange(t2, t1, -dt)*u.Myr
sgr_cen.interpolate(ts)

@pytest.mark.parametrize(("param_name", ), 
                         [("q1",), ("q2",), ("qz",), ("v_halo",), \
                            ("phi",), ("r_halo",)])
def test_parameter(param_name):
    halo_params = true_halo_params.copy()
    
    vals = np.linspace(param_ranges[param_name][0], 
                       param_ranges[param_name][1],
                       Nsteps)
    var_sums = []
    eig_sums = []
    for val in vals:
        # Define potential
        halo_params[param_name] = val
        potential = LawMajewski2010(**halo_params)
        ts, xs, vs = back_integrate(potential, sgr_snap, sgr_cen, dt)
        min_ps = minimum_distance_matrix(potential, xs, vs, sgr_cen)

        #var_sums.append(np.sum(np.var(min_ps, axis=0)))
        cov_matrix = np.cov(min_ps.T)
        w,v = np.linalg.eig(cov_matrix)
        
        var_sums.append(np.trace(cov_matrix))
        eig_sums.append(np.sum(w))
    
    plt.figure(figsize=(8,8))
    plt.plot(vals, var_sums, color="k", marker="o", linestyle="none", label="trace")
    plt.plot(vals, eig_sums, color="r", marker="o", linestyle="none", label="eigenvalues", alpha=0.6)
    plt.axvline(true_halo_params[param_name], color="r", linewidth=2)
    plt.xlabel(param_name)
    plt.ylabel("Sum of variances")
    plt.legend()
    plt.savefig("plots/tests/test_backintegrate_{0}.png".format(param_name))

def test_covariance():
    # Define potential
    potential = LawMajewski2010(**true_halo_params)
    ts, xs, vs = back_integrate(potential, sgr_snap, sgr_cen, dt)
    min_ps = minimum_distance_matrix(potential, xs, vs, sgr_cen)
    cov_matrix = np.cov(min_ps.T)
    
    fig,ax = plt.subplots(1,1,figsize=(8,8))
    ax.imshow(cov_matrix, interpolation="nearest", cmap=cm.Greys_r)
    plt.savefig("plots/tests/cov_matrix.png")
    
    fig,axes = plt.subplots(6,6,figsize=(16,16))
    
    for ii in range(axes.shape[0]):
        for jj in range(axes.shape[1]):
            if ii < jj:
                axes[ii,jj].set_visible(False)
                continue
            elif ii == jj:
                axes[ii,jj].hist(min_ps[:,jj], bins=25, color="k", \
                                 histtype="step", alpha=0.75, linewidth=2)
            else:
                axes[ii,jj].scatter(min_ps[:,jj], min_ps[:,ii], c="k", alpha=0.5)
    
    fig.savefig("plots/tests/6_by_6.png")

"""
n,bins,batches = plt.hist(np.sum(min_ps_wrong**2, axis=1), bins=25, color="r", histtype="step", alpha=1, linewidth=2)
n,bins,batches = plt.hist(np.sum(min_ps_true**2, axis=1), bins=bins, color="k", histtype="step", alpha=0.75, linewidth=2)
plt.ylim(0,300)
plt.savefig("plots/test.png")

def vary_one_parameter(sgr_cen, sgr_snap, dt):

    stats_for_param = dict()
    for param in ["qz", "q1", "q2", "v_halo", "phi", "r_halo"]:
        print(param)
        gen_variances = []
        vals = np.linspace(param_ranges[param][0], param_ranges[param][1], 20)
        for val in vals:
            halo_params = true_halo_params.copy()
            halo_params[param] = val
            potential = LawMajewski2010(**halo_params)
            
            var = back_integrate(potential, sgr_snap, sgr_cen, dt=dt)
            gen_variances.append(var)
        
        stats_for_param[param] = gen_variances
        
        plt.figure(figsize=(12,8))    
        plt.plot(vals, np.array(stats_for_param[param]), color="k", marker="o", linestyle="none")
        plt.axvline(true_halo_params[param])
        plt.xlabel(param)
        plt.ylabel("Median of minimum energy distance distribution")
        plt.savefig("plots/grid_search/{0}.png".format(param))     
"""