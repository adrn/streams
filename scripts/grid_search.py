# coding: utf-8

""" Perform a grid search over N parameters in the logarithmic halo potential
    to see if the MMEDD (median of the minimum energy distance distribution) is
    convex over all parameters.
"""

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys

# Third-party
import numpy as np

from streams.simulation.back_integrate import run_back_integration

true_halo_params = dict(v_halo=(115.7651*u.km/u.s).to(u.kpc/u.Myr).value,
                        q1=1.38,
                        q2=1.0,
                        qz=1.36,
                        phi=1.692969,
                        c=12.)

param_ranges = dict(qz=(0.1,3),
                    q1=(0.1,3),
                    q2=(0.1,3),
                    v_halo=((100.*u.km/u.s).to(u.kpc/u.Myr).value,
                            (150.*u.km/u.s).to(u.kpc/u.Myr).value),
                    phi=(1,2.5),
                    c=(8,20))

def vary_one_parameter():

    stats_for_param = dict()
    for param in ["qz", "q1", "q2", "v_halo", "phi", "c"]:
        stats = dict(mean=[], std=[], median=[])
        vals = np.linspace(param_ranges[param][0], param_ranges[param][1], 20)
        for val in vals:
            halo_params = true_halo_params.copy()
            halo_params[param] = val
            halo_potential = LogarithmicPotentialLJ(**halo_params)
        
            dist = run_back_integration(halo_potential, sgr_snap, sgr_cen)
        
            stats["mean"].append(np.mean(dist))
            stats["std"].append(np.std(dist))
            stats["median"].append(np.median(dist))
        
        stats_for_param[param] = stats
    
    for param in ["qz", "q1", "q2", "v_halo", "phi", "c"]:
        vals = np.linspace(param_ranges[param][0], param_ranges[param][1], 20)
        
        plt.figure(figsize=(12,8))
        plt.plot(vals, np.array(stats_for_param[param]["median"]), color="k", marker="o", linestyle="none")
        plt.axvline(true_halo_params[param])
        plt.xlabel(param)
        plt.ylabel("Median of minimum energy distance distribution")
        
def vary_two_parameters(param_pair, grid_size=10):

    param1,param2 = param_pair
    
    stats_for_param_pair = dict()
    p1_vals = np.linspace(param_ranges[param1][0], param_ranges[param1][1], grid_size)
    p2_vals = np.linspace(param_ranges[param2][0], param_ranges[param2][1], grid_size)
    
    for p1_val in p1_vals:
        for p2_val in p2_vals:
            halo_params = true_halo_params.copy()
            halo_params[param1] = p1_val
            halo_params[param2] = p2_val
            halo_potential = LogarithmicPotentialLJ(**halo_params)
        
            dist = run_back_integration(halo_potential, sgr_snap, sgr_cen)
            
            stats = dict()
            stats["mean"] = np.mean(dist)
            stats["std"] = np.std(dist)
            stats["median"] = np.median(dist)
            stats_for_param_pair[(p1_val,p2_val)] = stats
    
    return stats_for_param_pair

def plotstuff():    
    for param in ["qz", "q1", "q2", "v_halo", "phi", "c"]:
        vals = np.linspace(param_ranges[param][0], param_ranges[param][1], 20)
        
        plt.figure(figsize=(12,8))
        plt.plot(vals, np.array(stats_for_param[param]["median"]), color="k", marker="o", linestyle="none")
        plt.axvline(true_halo_params[param])
        plt.xlabel(param)
        plt.ylabel("Median of minimum energy distance distribution")