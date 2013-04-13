#!/hpc/astro/users/amp2217/yt-x86_64/bin/python
# coding: utf-8

""" In this module, I'll test how the distribution of 'energy distance' changes as I tweak
    various galaxy potential parameters. Ultimately, I want to come up with a way to evaluate
    the 'best' potential.
"""

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys
import datetime
import logging
import multiprocessing
import cPickle as pickle

# Third-party
import numpy as np
np.seterr(all="ignore")
import scipy
scipy.seterr(all="ignore")
from scipy import interpolate
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
import astropy.units as u
from astropy.io.misc import fnpickle, fnunpickle
import emcee
from emcee.utils import MPIPool

# Project
from streams.potential.lm10 import param_ranges, param_to_latex, param_units
from streams.simulation.setup import simulation_setup
from streams.simulation import config
from streams.data import SgrSnapshot, SgrCen
from streams.data.gaia import add_uncertainties_to_particles
from streams.inference import ln_posterior
from streams.plot import plot_sampler_pickle

def write_defaults(filename=None):
    """ Write a default configuration file for running infer_potential to
        the supplied path. If no path supplied, will return a string version.
        
        Parameters
        ----------
        path : str (optional)
            Path to write the file to or return string version.
    """
    config_file = []
    config_file.append("(I) particles: 100")
    config_file.append("(U) dt: 1. Myr # timestep")
    config_file.append("(B) observational_errors: yes # add observational errors")
    config_file.append("(S) expr: tub > 0.")
    config_file.append("(L,S) model_parameters: q1 qz v_halo phi")
    config_file.append("(I) seed: 42")
    config_file.append("(I) walkers: 128")
    config_file.append("(I) burn_in: 100")
    config_file.append("(I) steps: 100")
    config_file.append("(B) mpi: no")
    config_file.append("(B) make_plots: no")
    config_file.append("(S) output_path: /tmp/")
    
    if filename == None:
        filename = "streams.cfg"
        
    f = open(filename, "w")
    f.write("\n".join(config_file))
    f.close()

def main(config_file):
        
    # Make sure input configuration file exists
    if not os.path.exists(config_file):
        raise ValueError("Configuration file '{0}' doesn't exist!"
                         .format(config_file))
    
    simulation_params = config.read(config_file)
    
    # Expression for selecting particles from the simulation data snapshot
    expr = "(tub > 10.)"
    if len(simulation_params["expr"]) > 0:
        expr = ""
        expr += " & " + " & ".join(["({0})".format(x) for x in simulation_params["expr"]])
    
    # Read in Sagittarius simulation data
    np.random.seed(simulation_params["seed"])
    sgr_cen = SgrCen()
    satellite_orbit = sgr_cen.as_orbit()
    
    sgr_snap = SgrSnapshot(N=simulation_params["particles"],
                           expr=simulation_params["expr"])
    particles = sgr_snap.as_particles()
    
    # Define new time grid
    time_grid = np.arange(max(satellite_orbit.t.value), 
                          min(satellite_orbit.t.value),
                          -simulation_params["dt"].to(satellite_orbit.t.unit).value)
    time_grid *= satellite_orbit.t.unit
    
    # Interpolate satellite_orbit onto new time grid
    satellite_orbit = satellite_orbit.interpolate(time_grid)
    
    if simulation_params["observational_errors"]:
        particles = add_uncertainties_to_particles(particles)
    
    best_parameters = infer_potential(particles, satellite_orbit, simulation_params)
    
def infer_potential(particles, satellite_orbit, simulation_params):
    """ """
    
    # Shorthand!
    sp = simulation_params
    
    # Create the starting points for all walkers
    p0 = []
    for ii in range(sp["walkers"]):
        p0.append([np.random.uniform(param_ranges[p_name][0], param_ranges[p_name][1])
                    for p_name in sp["model_parameters"]])
    p0 = np.array(p0)
    ndim = len(sp["model_parameters"])
    
    # Construct the log posterior probability function to pass in to emcee
    args = sp["model_parameters"], particles, satellite_orbit
    
    if sp["mpi"]:
        # Initialize the MPI pool
        pool = MPIPool()

        # Make sure the thread we're running on is the master
        if not pool.is_master():
            pool.wait()
            sys.exit(0)

        sampler = emcee.EnsembleSampler(sp["walkers"], ndim, ln_posterior, 
                                        pool=pool, args=args)
    else:
        if "threads" in sp.keys():
            threads = sp["threads"]
        else:
            threads = 1
            
        sampler = emcee.EnsembleSampler(sp["walkers"], ndim, 
                                        ln_posterior, args=args,
                                        threads=threads)
    
    logger.info("About to start simulation with parameters: \n{0}"
                .format("\n\t".join(["{0}: {1}".format(k,v) for k,v in sp.items()])))
    
    # Create a new path for the output
    path = os.path.join(sp["output_path"], 
                        datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    
    logger.debug("Output path: {0}".format(path))
    
    if sp["make_plots"]:
        if not os.path.exists(path):
            os.mkdir(path)
    
        # Plot the initial positions of the particles in galactic XYZ coordinates
        fig,axes = particles.plot_positions()
        fig.savefig(os.path.join(path, "particles.png"))
    
    # If a burn-in period is requested, run the sampler for nburn_in steps then
    #   reset the walkers and use the end positions as new initial conditions
    if sp["burn_in"] > 0:
        pos, prob, state = sampler.run_mcmc(p0, sp["burn_in"])
        sampler.reset()
    else:
        pos = p0
    
    sampler.run_mcmc(pos, sp["steps"])
    #run_parameters.append("median acceptance fraction: {0:.3f}".\
    #                        format(np.median(sampler.acceptance_fraction)))
    
    # if we're running with MPI, we have to close the processor pool, otherwise
    #   the script will never finish running until the end of timmmmeeeee (echo)
    if sp["mpi"]: pool.close()
    
    data_file = os.path.join(path, "sampler_data.pickle")
    
    sampler.lnprobfn = None
    sampler.pickle(data_file)

    idx = (sampler.acceptance_fraction > 0.1) & \
            (sampler.acceptance_fraction < 0.6) # rule of thumb, bitches
    logger.info("{0} walkers ({1:.1f}%) converged"
                .format(sum(idx), sum(idx)/sp["walkers"]*100))
    
    # Pluck out good chains, make a new flatchain from those...
    good_flatchain = []
    good_chains = sampler.chain[idx]
    for chain in good_chains:
        good_flatchain += list(chain)
    good_flatchain = np.array(good_flatchain)
    
    if sp["make_plots"]:
        fig = plot_sampler_pickle(data_file, params=sp["model_parameters"])
        fig.savefig(os.path.join(path, "emcee_sampler.png"), format="png")
    
    # Get "best" (mean) potential parameters:
    return dict(zip(sp["model_parameters"],np.mean(good_flatchain,axis=0)))

if __name__ == "__main__":
    from argparse import ArgumentParser
    
    parser = ArgumentParser(description="")
    parser.add_argument("-v", "--verbose", action="store_true", dest="verbose", 
                    default=False, help="Be chatty! (default = False)")
    parser.add_argument("-q", "--quiet", action="store_true", dest="quiet", 
                    default=False, help="Be quiet! (default = False)")
    parser.add_argument("-f", "--file", dest="file", default="streams.cfg", 
                    help="Path to the configuration file to run with.")
    
    # Dump defaults
    parser.add_argument("-d", "--dump-defaults", action="store_true", 
                    dest="dump", default=False, help="Don't do anything "
                    "except write out the default config file.")
    parser.add_argument("-o", "--output", dest="output", default=None, 
                    help="File to write the default config settings to.")
    
    args = parser.parse_args()
    
    # Create logger
    logger = logging.getLogger(__name__)
    ch = logging.StreamHandler()
    formatter = logging.Formatter("%(name)s / %(levelname)s / %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    elif args.quiet:
        logger.setLevel(logging.ERROR)
    else:
        logger.setLevel(logging.INFO)
    
    if args.dump:
        if args.output == None:
            raise ValueError("If you want to dump a default config file, you "
                             "must also throw the --output flag and give it "
                             "the full path to the file you want to write to.")
        
        write_defaults(args.output)
        sys.exit(0)
    
    main(args.file)
    sys.exit(0)
