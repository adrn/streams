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
import astropy.units as u
from astropy.io.misc import fnpickle, fnunpickle
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from streams.potential import *
from streams.simulation import run_back_integration
from streams.data import SgrSnapshot, SgrCen

true_halo_params = dict(v_halo=(121.858*u.km/u.s).to(u.kpc/u.Myr).value,
                        q1=1.38,
                        q2=1.0,
                        qz=1.36,
                        phi=1.692969,
                        c=12.)

param_ranges = dict(qz=(1.,2.),
                    q1=(1.,2.),
                    q2=(1.,2.),
                    v_halo=((100.*u.km/u.s).to(u.kpc/u.Myr).value,
                            (150.*u.km/u.s).to(u.kpc/u.Myr).value),
                    phi=(np.pi/4, 3*np.pi/4),
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
        
def vary_two_parameters(param_pair, sgr_snap, sgr_cen, dt, grid_size=10):
    param1,param2 = param_pair
    
    stats_for_param_pair = dict()
    p1_vals = np.linspace(param_ranges[param1][0], param_ranges[param1][1], grid_size)
    p2_vals = np.linspace(param_ranges[param2][0], param_ranges[param2][1], grid_size)
    
    X,Y = np.meshgrid(p1_vals, p2_vals)
    Z = np.zeros_like(X)
    
    for ii,p1_val in enumerate(p1_vals):
        for jj,p2_val in enumerate(p2_vals):
            halo_params = true_halo_params.copy()
            halo_params[param1] = p1_val
            halo_params[param2] = p2_val
            halo_potential = LogarithmicPotentialLJ(**halo_params)
        
            dist = run_back_integration(halo_potential, sgr_snap, sgr_cen, dt)
            Z[ii,jj] = dist
    
    return X,Y,Z

def main():
    sgr_snap = SgrSnapshot(num=100, no_bound=True)
    sgr_cen = SgrCen()
    
    # Get timestep information from SGR_CEN
    t1 = min(sgr_cen.data["t"])
    t2 = max(sgr_cen.data["t"])
    dt = sgr_cen.data["dt"][0]*10
        
    # Interpolate SgrCen data onto new times
    ts = np.arange(t2, t1, -dt)
    sgr_cen.interpolate(ts)

    # q2 qz phi v_halo c
    for params in [("qz", "phi"), ("qz","q1"), ("q1","phi"), 
                   ("v_halo", "q1"), ("v_halo", "qz"), ("v_halo", "phi")]:
        print(params)
        pickle_filename = "data/grid_search/{0}_{1}.pickle".format(*params)
        
        if not os.path.exists(pickle_filename):
            X,Y,Z = vary_two_parameters(params, sgr_snap, sgr_cen, dt, grid_size=10)
            fnpickle((X,Y,Z), pickle_filename)
        
        (X,Y,Z) = fnunpickle(pickle_filename)
        
        plt.figure(figsize=(12,12))
        plt.pcolor(X, Y, Z, cmap=cm.Greys)
        plt.axvline(true_halo_params[params[0]])
        plt.axhline(true_halo_params[params[1]])
        plt.xlabel(params[0])
        plt.ylabel(params[1])
        plt.savefig("plots/grid_search/{0}_{1}.png".format(*params))

if __name__ == "__main__":
    main()
