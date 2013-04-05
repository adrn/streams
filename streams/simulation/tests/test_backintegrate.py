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
sgr_snap = SgrSnapshot(N=1000)

# Get timestep information from SGR_CEN
t1 = min(sgr_cen["t"])
t2 = max(sgr_cen["t"])
dt = sgr_cen["dt"][0]*10

# Interpolate SgrCen data onto new times
ts = np.arange(t2, t1, -dt)*u.Myr
sgr_cen = sgr_cen.interpolate(ts)

def test_simple_integrate():
    potential = LawMajewski2010(**true_halo_params)
    ts, xs, vs = back_integrate(potential, 
                                sgr_snap.as_particles(), 
                                t2, t1, dt)
    
    assert len(ts) == len(sgr_cen["t"])
    assert (ts == sgr_cen["t"]).all()

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
        ts, xs, vs = back_integrate(potential, 
                                    sgr_snap.as_particles(), 
                                    t1, t2, dt)
        
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

def test_single_column_distributions():
    # Define potential
    potential = LawMajewski2010(**true_halo_params)
    ts, xs, vs = back_integrate(potential, sgr_snap, sgr_cen, dt)
    min_ps = minimum_distance_matrix(potential, xs, vs, sgr_cen)
    
    true_halo_params["qz"] = true_halo_params["qz"]*1.1
    potential = LawMajewski2010(**true_halo_params)
    ts, xs, vs = back_integrate(potential, sgr_snap, sgr_cen, dt)
    wrong_min_ps = minimum_distance_matrix(potential, xs, vs, sgr_cen)
    
    for ii in range(6):
        fig, ax = plt.subplots(1,1,figsize=(8,8))
        n,bins,patches = ax.hist(wrong_min_ps[:,ii], bins=25, color="r", \
                                    histtype="step", alpha=0.75, linewidth=2)
        n,bins,patches = ax.hist(min_ps[:,ii], bins=bins, color="k", \
                                    histtype="step", alpha=0.75, linewidth=2)
    
        fig.savefig("plots/tests/single_column_distribution_{0}.png".format(ii))

@pytest.mark.parametrize(("param_name", ), 
                         [("q1",), ("q2",), ("qz",), ("v_halo",), \
                            ("phi",), ("r_halo",)])
def test_covariance_change(param_name):
    print(param_name)
    Nsteps = 9
    halo_params = true_halo_params.copy()
    
    vals = np.linspace(param_ranges[param_name][0], 
                       param_ranges[param_name][1],
                       Nsteps)
    
    # First run with the True parameter value, and then plot the *difference*
    #   from the true covariance matrix
    potential = LawMajewski2010(**halo_params)
    ts, xs, vs = back_integrate(potential, sgr_snap, sgr_cen, dt)
    min_ps = minimum_distance_matrix(potential, xs, vs, sgr_cen)
    true_cov_matrix = np.cov(min_ps.T)
    
    fig,axes = plt.subplots(int(np.sqrt(Nsteps)),int(np.sqrt(Nsteps)),figsize=(8,8), sharex=True, sharey=True)
    flat_axes = axes.flat
    cov_matrices = []
    for ii,val in enumerate(vals):
        print("\t {0}".format(val))
        
        # Define potential
        halo_params[param_name] = val
        potential = LawMajewski2010(**halo_params)
        ts, xs, vs = back_integrate(potential, sgr_snap, sgr_cen, dt)
        min_ps = minimum_distance_matrix(potential, xs, vs, sgr_cen)
        cov_matrix = np.cov(min_ps.T)
        cov_matrices.append(cov_matrix)
    
    vmin = min([cov_matrix.min() for cov_matrix in cov_matrices])
    vmax = max([cov_matrix.max() for cov_matrix in cov_matrices])
    
    for ii,val in enumerate(vals):
        flat_axes[ii].imshow(cov_matrices[ii], interpolation="nearest", \
                         cmap=cm.Greys, vmin=vmin, \
                         vmax=vmax)
        flat_axes[ii].text(3,1,"{0:.2f}".format(val),fontsize=18, color='red')
    
    for ii in range(int(np.sqrt(Nsteps))):
        for jj in range(int(np.sqrt(Nsteps))):
            if jj == 0:
                axes[ii,jj].set_yticklabels(["", "x","y","z","vx","vy","vz"])
            
            if ii == int(np.sqrt(Nsteps))-1:
                axes[ii,jj].set_xticklabels(["", "x","y","z","vx","vy","vz"])
    
    fig.suptitle("{0}, true value: {1:.3f}".format(param_name, true_halo_params[param_name]))
    fig.subplots_adjust(hspace=0.0, wspace=0.0, left=0.08, bottom=0.08, top=0.9, right=0.9 )
    fig.savefig("plots/tests/cov_matrix_vary{0}.png".format(param_name))