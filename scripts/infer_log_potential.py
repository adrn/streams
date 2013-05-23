#!/hpc/astro/users/amp2217/yt-x86_64/bin/python
# coding: utf-8

""" In this module, I'll test how the distribution of 'energy distance' changes 
    as I tweak various galaxy potential parameters. Ultimately, I want to come 
    up with a way to evaluate the 'best' potential.
"""

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys
import copy
import logging
from datetime import datetime
import multiprocessing

# Third-party
import matplotlib.pyplot as plt
import numpy as np
np.seterr(all="ignore")
import scipy
scipy.seterr(all="ignore")
import astropy.units as u
from emcee.utils import MPIPool
from astropy.io.misc import fnpickle, fnunpickle

# Project
from streams.simulation.config import read
from streams.data import SgrSnapshot, SgrCen, read_lm10
from streams.data.gaia import add_uncertainties_to_particles
from streams.inference import infer_potential, max_likelihood_parameters
from streams.plot import plot_sampler_pickle
from streams.inference.lm10 import ln_posterior
from streams.potential.lm10 import halo_params, param_ranges

global pool
pool = None

# Create logger
logger = logging.getLogger(__name__)

def main(config_file):
    
    # Read in simulation parameters from config file
    config = read(config_file)
    
    # Expression for selecting particles from the simulation data snapshot
    if len(config["expr"]) > 0:
        if isinstance(config["expr"], list):
            expr = " & ".join(["({0})".format(x) for x in expr])
        else:
            expr = config["expr"]
    else:
        expr = None
    
    if config["mpi"]:
        # Initialize the MPI pool
        pool = MPIPool()

        # Make sure the thread we're running on is the master
        if not pool.is_master():
            pool.wait()
            sys.exit(0)
    else:
        if config.has_key("threads") and config["threads"] > 1:
            pool = multiprocessing.Pool(config["threads"])
        else:
            pool = None
    
    np.random.seed(config["seed"])
    
    # Read in Sagittarius simulation data
    if config["particle_source"] == "kvj":
        satellite_orbit = SgrCen().as_orbit()
        satellite = satellite_orbit[-1]
        
        sgr_snap = SgrSnapshot(N=config["particles"],
                               expr=expr)
        
        # Define new time grid
        time_grid = np.arange(max(satellite_orbit.t.value), 
                              min(satellite_orbit.t.value),
                              -config["dt"].to(satellite_orbit.t.unit).value)
        time_grid *= satellite_orbit.t.unit
        
        particles = sgr_snap.as_particles()
        
    elif config["particle_source"] == "lm10":
        time_grid, satellite, particles = read_lm10(N=config["particles"], 
                                                    expr=expr,
                                                    dt=config["dt"])
    else:
        raise ValueError("Invalid particle source {0}"
                         .format(config["particle_source"]))
    
    # Create a new path for the output
    if config["make_plots"]:
        iso_now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        dirname = "{0}{1}".format(config.get("name", ""), iso_now)
        path = os.path.join(config["output_path"], dirname)
        os.mkdir(path)
    
    # Get the number of bootstrap reamples. if not specified, it's just 1
    B = config.get("bootstrap_resamples", 1)
    
    if config["observational_errors"]:
        rv_error = config.get("radial_velocity_error", None)
        d_error = config.get("distance_error_percent", None)
        pre_error_particles = copy.copy(particles)
        particles = add_uncertainties_to_particles(particles, 
                                                radial_velocity_error=rv_error,
                                                distance_error_percent=d_error)
    
    # Create initial position array for walkers
    for p_name in config["model_parameters"]:
        # sample initial parameter values from uniform distributions over 
        #   the ranges specified in lm10.py
        this_p = np.random.uniform(param_ranges[p_name][0], 
                                   param_ranges[p_name][1],
                                   size=config["walkers"])
        try:
            p0 = np.vstack((p0, this_p))
        except NameError:
            p0 = this_p
    p0 = p0.T
    
    all_best_parameters = []
    for bb in range(B):        
        try:
            sampler = infer_potential(ln_posterior, p0, steps=config["steps"],
                                      burn_in=config["burn_in"], pool=pool,
                                      args=(config["model_parameters"], 
                                            particles, 
                                            satellite, 
                                            time_grid))
        except:
            if config["mpi"]: pool.close()
            raise
        
        try:
            best_parameters = max_likelihood_parameters(sampler)
        except:
            continue
            
        all_best_parameters.append(best_parameters)
        
        # Create a new path for the output
        if config["make_plots"]:
            # Plot the positions of the particles in galactic XYZ coordinates
            fig,axes = pre_error_particles.plot_r(
                                        subplots_kwargs=dict(figsize=(16,16)),
                                        scatter_kwargs={"c":"k"})
            particles.plot_r(axes=axes, scatter_kwargs={"c":"r"})
            fig.savefig(os.path.join(path, "positions.png"))
            
            fig,axes = pre_error_particles.plot_v(
                                        subplots_kwargs=dict(figsize=(16,16)),
                                        scatter_kwargs={"c":"k"})
            particles.plot_v(axes=axes, scatter_kwargs={"c":"r"})
            fig.savefig(os.path.join(path, "velocities.png"))
            
            # write the sampler to a pickle file
            data_file = os.path.join(path, "sampler_data.pickle")
            sampler.lnprobfn = None
            sampler.pool = None
            fnpickle(sampler, data_file)
            
            # make sexy plots from the sampler data
            fig = plot_sampler_pickle(os.path.join(path,data_file), 
                                      params=config["model_parameters"], 
                                      acceptance_fraction_bounds=(0.15,0.6),
                                      show_true=True)
            
            # add the max likelihood estimates to the plots                           
            for ii,param_name in enumerate(config["model_parameters"]):
                fig.axes[int(2*ii+1)].axhline(best_parameters[ii], 
                                              color="#CA0020",
                                              linestyle="--",
                                              linewidth=2)
            
            fig.savefig(os.path.join(path, "emcee_sampler_{0}.png".format(bb)))
    
    # if we're running with MPI, we have to close the processor pool, otherwise
    #   the script will never finish running until the end of timmmmeeeee (echo)
    if config["mpi"]: pool.close()
    
    best_p = dict()
    for ii,name in enumerate(config["model_parameters"]):
        best_p[name] = [x[ii] for x in all_best_parameters]
    
    # finally, make plots showing the bootstrap resamples
    if B > 1 and config["make_plots"]:
        fnpickle(best_p, os.path.join(path, "all_best_parameters.pickle"))
        
        fig,axes = plt.subplots(4,1,figsize=(14,12))
        fig.subplots_adjust(left=0.075, right=0.95)
        for ii,name in enumerate(config["model_parameters"]):
            try:
                p = halo_params[name].value
            except AttributeError:
                p = halo_params[name]
                
            axes[ii].hist(best_p[name], bins=25, histtype="step", color="k", linewidth=2)
            axes[ii].axvline(p, linestyle="--", color="#EF8A62", linewidth=3)
            axes[ii].set_ylabel(name)
            axes[ii].set_ylim(0,20)
        
        fig.savefig(os.path.join(path,"bootstrap_1d.png"))
        
        # Now make 2D plots of the bootstrap results
        #fig,axes = scatter_plot_matrix()
    
    if config["make_plots"]:
        with open(config_file) as f:
            g = open(os.path.join(path,"config.txt"), "w")
            g.write(f.read())
            g.close()
        
if __name__ == "__main__":
    from argparse import ArgumentParser
    
    parser = ArgumentParser(description="")
    parser.add_argument("-v", "--verbose", action="store_true", dest="verbose", 
                    default=False, help="Be chatty! (default = False)")
    parser.add_argument("-q", "--quiet", action="store_true", dest="quiet", 
                    default=False, help="Be quiet! (default = False)")
    parser.add_argument("-f", "--file", dest="file", default="streams.cfg", 
                    help="Path to the configuration file to run with.")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    elif args.quiet:
        logging.basicConfig(level=logging.ERROR)
    else:
        logging.basicConfig(level=logging.INFO)
    
    try:
        main(args.file)
    except:
        if pool is not None:
            pool.close()
        raise
    sys.exit(0)
