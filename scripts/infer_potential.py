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
from dateteim import datetime
import multiprocessing

# Third-party
import numpy as np
np.seterr(all="ignore")
import scipy
scipy.seterr(all="ignore")
import astropy.units as u
from emcee.utils import MPIPool
from astropy.io.misc import fnpickle, fnunpickle

# Project
from streams.simulatio.config import read
from streams.data import SgrSnapshot, SgrCen
from streams.data.gaia import add_uncertainties_to_particles
from streams.inference import infer_potential, max_likelihood_parameters

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
        
    # Read in Sagittarius simulation data
    np.random.seed(config["seed"])
    sgr_cen = SgrCen()
    satellite_orbit = sgr_cen.as_orbit()
    
    sgr_snap = SgrSnapshot(N=config["particles"],
                           expr=config["expr"])
    particles = sgr_snap.as_particles()
    
    # Define new time grid
    time_grid = np.arange(max(satellite_orbit.t.value), 
                          min(satellite_orbit.t.value),
                          -config["dt"].to(satellite_orbit.t.unit).value)
    time_grid *= satellite_orbit.t.unit
    
    # Interpolate satellite_orbit onto new time grid
    satellite_orbit = satellite_orbit.interpolate(time_grid)
    
    # Create a new path for the output
    if config["make_plots"]:
        path = os.path.join(config["output_path"], 
                            datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
        os.mkdir(path)
    
    # TODO: this is where i can turn this in to a bootstrap loop?
    
    if config["observational_errors"]:
        particles = add_uncertainties_to_particles(particles)
    
    try:
        sampler = infer_potential(particles, satellite_orbit, path, 
                                    config, pool=pool)
    except:
        if config["mpi"]: pool.close()
        raise
    
    # if we're running with MPI, we have to close the processor pool, otherwise
    #   the script will never finish running until the end of timmmmeeeee (echo)
    if config["mpi"]: pool.close()
    
    best_parameters = max_likelihood_parameters(sampler)
    
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
