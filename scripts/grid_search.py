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
from streams.simulation import back_integrate
from streams.data import SgrSnapshot, SgrCen
from streams.potential.lm10 import halo_params as true_halo_params

param_ranges = dict(qz=(1.,2.),
                    q1=(1.,2.),
                    q2=(1.,2.),
                    v_halo=((100.*u.km/u.s).to(u.kpc/u.Myr).value,
                            (150.*u.km/u.s).to(u.kpc/u.Myr).value),
                    phi=(np.pi/4, 3*np.pi/4),
                    r_halo=(8,20))

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
            potential = LawMajewski2010(**halo_params)
        
            gen_variance = back_integrate(potential, sgr_snap, sgr_cen, dt)
            Z[ii,jj] = gen_variance
    
    return X,Y,Z

def main_two():
    sgr_snap = SgrSnapshot(num=100)
    sgr_cen = SgrCen()
    
    # Get timestep information from SGR_CEN
    t1 = min(sgr_cen.t)
    t2 = max(sgr_cen.t)
    dt = sgr_cen.dt[0]*10
    
    # Interpolate SgrCen data onto new times
    ts = np.arange(t2, t1, -dt)*u.Myr
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
    
def main_one():
    sgr_snap = SgrSnapshot(num=100)
    sgr_cen = SgrCen()
    
    # Get timestep information from SGR_CEN
    t1 = min(sgr_cen.t)
    t2 = max(sgr_cen.t)
    dt = sgr_cen.dt[0]*10
    
    # Interpolate SgrCen data onto new times
    ts = np.arange(t2, t1, -dt)*u.Myr
    sgr_cen.interpolate(ts)
    
    vary_one_parameter(sgr_cen, sgr_snap, dt)

if __name__ == "__main__":
    #main_two()
    main_one()
    
