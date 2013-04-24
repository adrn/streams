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
from streams.simulation import TestParticle
from streams.data import SgrSnapshot, SgrCen, read_lm10
from streams.data.gaia import add_uncertainties_to_particles
from streams.inference import infer_potential, max_likelihood_parameters
from streams.plot import plot_sampler_pickle
from streams.potential.lm10 import halo_params

# Create logger
logger = logging.getLogger(__name__)

def main(config_file):
    
    # Read in simulation parameters from config file
    config = read(config_file)
    
    # Expression for selecting particles from the simulation data snapshot
    expr = ""
    if len(config["expr"]) > 0:
        expr += " & " + " & ".join(["({0})".format(x) for x in config["expr"]])
    else:
        expr += "(tub > 0.)"
    
    if config["mpi"]:
        # Initialize the MPI pool
        pool = MPIPool()

        # Make sure the thread we're running on is the master
        if not pool.is_master():
            pool.wait()
            sys.exit(0)
    else:
        if config.has_key("threads"):
            pool = multiprocessing.Pool(config["threads"])
    
    if isinstance(config["expr"], list):
        expr = " & ".join(["({0})".format(x) for x in config["expr"]])
    else:
        expr = config["expr"]
    
    np.random.seed(config["seed"])
    
    # Read in Sagittarius simulation data
    if config["particle_source"] == "kvj":
        satellite_orbit = SgrCen().as_orbit()
        satellite_ic = satellite_orbit[-1]
        
        sgr_snap = SgrSnapshot(N=config["particles"],
                               expr=config["expr"])
        
        # Define new time grid
        time_grid = np.arange(max(satellite_orbit.t.value), 
                              min(satellite_orbit.t.value),
                              -config["dt"].to(satellite_orbit.t.unit).value)
        time_grid *= satellite_orbit.t.unit
        
        particles = sgr_snap.as_particles()
        
    elif config["particle_source"] == "lm10":
        satellite_ic, particles = read_lm10(N=config["particles"], expr=expr)
                        
        # Define new time grid -here
        time_grid = np.arange(satellite_ic.t1,
                              satellite_ic.t2,
                              -config["dt"].to(u.Myr).value)
        time_grid *= u.Myr
        
    else:
        raise ValueError("Invalid particle source {0}"
                         .format(config["particle_source"]))
    
    # Interpolate satellite_orbit onto new time grid
    #satellite_orbit = satellite_orbit.interpolate(time_grid)
    
    # Create a new path for the output
    if config["make_plots"]:
        path = os.path.join(config["output_path"], 
                            datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
        
        if config.has_key("name"):
            path = path + "-" + config["name"]
            
        os.mkdir(path)
    
    # Get the number of bootstrap reamples. if not specified, it's just 1
    B = config.get("bootstrap_resamples", 1)
    
    if config["observational_errors"]:
        particles = add_uncertainties_to_particles(particles)
    
    all_best_parameters = []
    for bb in range(B):    
        try:
            sampler = infer_potential(particles, satellite_ic,
                                      t=time_grid,
                                      model_parameters=config["model_parameters"],
                                      walkers=config["walkers"],
                                      steps=config["steps"],
                                      burn_in=config["burn_in"],
                                      pool=pool)
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
            fig,axes = particles.plot_positions()
            fig.savefig(os.path.join(path, "particles.png"))
            
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
    
    main(args.file)
    sys.exit(0)
