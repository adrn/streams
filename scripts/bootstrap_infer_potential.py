#!/hpc/astro/users/amp2217/yt-x86_64/bin/python
# coding: utf-8

""" In this module, I'll:
        - Sample some stars from Sgr
        - Draw B bootstrap sub-samples from this set
        - For each b sub-sample, do the inference and get max. likelihood
            parameters
        - Plot up projections of the derived parameter distribution
"""

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys
import logging
import datetime

# Third-party
import numpy as np
np.seterr(all="ignore")
import scipy
scipy.seterr(all="ignore")
import astropy.units as u
from astropy.io.misc import fnpickle, fnunpickle
from emcee.utils import MPIPool
import matplotlib.pyplot as plt

# Project
from streams.simulation import config
from streams.data import SgrSnapshot, SgrCen
from streams.data.gaia import add_uncertainties_to_particles
from streams.inference import infer_potential

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
    config_file.append("(I) bootstrap_resamples: 100")
    config_file.append("(U) dt: 5. Myr # timestep")
    config_file.append("(B) observational_errors: yes # add observational errors")
    config_file.append("(S) expr: tub > 0.")
    config_file.append("(L,S) model_parameters: q1 qz v_halo phi")
    config_file.append("(I) seed: 42")
    config_file.append("(I) walkers: 32")
    config_file.append("(I) burn_in: 50")
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
    
    if simulation_params["mpi"]:
        # Initialize the MPI pool
        pool = MPIPool()

        # Make sure the thread we're running on is the master
        if not pool.is_master():
            pool.wait()
            sys.exit(0)
    
    # Expression for selecting particles from the simulation data snapshot
    expr = "(tub > 10.)"
    if len(simulation_params["expr"]) > 0:
        expr = ""
        expr += " & " + " & ".join(["({0})".format(x) for x in simulation_params["expr"]])
    
    # Read in Sagittarius simulation data
    np.random.seed(simulation_params["seed"])
    sgr_cen = SgrCen()
    satellite_orbit = sgr_cen.as_orbit()
    
    sgr_snap = SgrSnapshot(N=simulation_params["particles"]*10,
                           expr=simulation_params["expr"])
    particles = sgr_snap.as_particles()
    
    # Define new time grid
    time_grid = np.arange(max(satellite_orbit.t.value), 
                          min(satellite_orbit.t.value),
                          -simulation_params["dt"].to(satellite_orbit.t.unit).value)
    time_grid *= satellite_orbit.t.unit
    
    # Interpolate satellite_orbit onto new time grid
    satellite_orbit = satellite_orbit.interpolate(time_grid)
    
    # Create a new path for the output
    path = os.path.join(simulation_params["output_path"], 
                        datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    os.mkdir(path)
    
    all_best_parameters = []
    for bb in range(simulation_params["bootstrap_resamples"]):
        resample_idx = np.random.randint(len(particles), size=simulation_params["particles"])
        bootstrap_particles = particles[resample_idx]
        
        if simulation_params["observational_errors"]:
            bootstrap_particles = add_uncertainties_to_particles(bootstrap_particles)
    
        best_parameters = infer_potential(particles, satellite_orbit, 
                                          path, simulation_params, pool=pool)
        all_best_parameters.append(best_parameters)
    
    # if we're running with MPI, we have to close the processor pool, otherwise
    #   the script will never finish running until the end of timmmmeeeee (echo)
    if simulation_params["mpi"]: pool.close()
    
    fnpickle(all_best_parameters, os.path.join(path,"all_best_parameters.pickle"))
    
    all_q1 = [x["q1"] for x in all_best_parameters]
    all_qz = [x["qz"] for x in all_best_parameters]
    all_v_halo = [x["v_halo"] for x in all_best_parameters]
    all_phi = [x["phi"] for x in all_best_parameters]    
    
    fig,axes = plt.subplots(4,1)
    axes[0].hist(all_q1)
    axes[0].hist(all_qz)
    axes[0].hist(all_v_halo)
    axes[0].hist(all_phi)
    fig.savefig(os.path.join(path,"bootstrap_test.png"))
    
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
