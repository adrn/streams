#!/hpc/astro/users/amp2217/yt-x86_64/bin/python
# coding: utf-8

""" Script for using the Rewinder to infer the Galactic host potential """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys
import shutil
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
from astropy.io.misc import fnpickle, fnunpickle
from astropy.utils.console import color_print
import triangle

try:
    from emcee.utils import MPIPool
except ImportError:
    color_print("Failed to import MPIPool from emcee! MPI functionality "
                "won't work.", "yellow")

# Project
from streams.simulation.config import read
from streams.observation.gaia import add_uncertainties_to_particles
from streams.inference import infer_potential, max_likelihood_parameters
from streams.plot import plot_sampler_pickle, bootstrap_scatter_plot

global pool
pool = None

# Create logger
logger = logging.getLogger(__name__)

def main(config_file, job_name=None):
    
    # Read in simulation parameters from config file
    config = read(config_file)
    
    # Read particles and posterior from whatever simulation
    if config['particle_source'] == 'lm10':
        from streams.io.lm10 import particles_today, satellite_today, time
        from streams.inference.lm10 import ln_posterior, param_ranges
        from streams.potential.lm10 import true_params, _true_params
    elif config['particle_source'] == 'pal5':
        from streams.io.pal5 import particles_today, satellite_today, time
        from streams.inference.pal5 import ln_likelihood, ln_posterior, param_ranges
        from streams.potential.pal5 import true_params, _true_params
    else:
        raise ValueError("Invalid particle source {0}"
                         .format(config["particle_source"]))
                         
    # Expression for selecting particles from the simulation data snapshot
    if len(config["expr"]) > 0:
        expr = config["expr"]
    else:
        expr = None
    
    np.random.seed(config["seed"])
    
    # This needs to go here so I don't read in the particle file 128 times!!
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
    
    # Get the number of bootstrap reamples. if not specified, it's just 1
    B = config.get("bootstrap_resamples", 1)
    
    # Actually get simulation data
    satellite = satellite_today()
    t1,t2 = time()
    if isinstance(expr, list):
        if not isinstance(config["particles"], list):
            raise ValueError("If multiple expr's provided, multiple "
                             "particle numbers must be provided!")
        elif len(config["particles"]) != len(expr):
            raise ValueError("Must supply a particle count for each expr")
        
        for N_i,expr_i in zip(config["particles"], expr):
            these_p = particles_today(N=N_i*B, expr=expr_i)
            
            try:
                particles = particles.merge(these_p)
            except NameError:
                particles = these_p
        Nparticles = len(particles._r)
    else:
        Nparticles = config["particles"]
        particles = particles_today(N=Nparticles*B, expr=expr)
    
    if config["observational_errors"]:
        rv_error = config.get("radial_velocity_error", None)
        d_error = config.get("distance_error_percent", None)
        mu_error = config.get("proper_motion_error", None)
        pre_error_particles = copy.copy(particles)
        particles = add_uncertainties_to_particles(particles, 
                                                radial_velocity_error=rv_error,
                                                distance_error_percent=d_error,
                                                proper_motion_error=mu_error)
    
    # Create initial position array for walkers
    for p_name in config["model_parameters"]:
        # Dan F-M says emcee is better at expanding than contracting...
        this_p = np.random.uniform(param_ranges[p_name][0], 
                                   param_ranges[p_name][1],
                                   size=config["walkers"])
        
        try:
            p0 = np.vstack((p0, this_p))
        except NameError:
            p0 = this_p
    
    resolution = config.get("resolution", 4.)
    
    p0 = p0.T
    if p0.ndim == 1:
        p0 = p0[np.newaxis].T
    
    # Create a new path for the output
    if config["make_plots"]:
        if job_name is not None:
            path = os.path.join(config["output_path"], job_name)
        else:
            if config.has_key("name"):
                path = os.path.join(config["output_path"], config["name"])
            else:    
                iso_now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                path = os.path.join(config["output_path"], iso_now)
        
        if os.path.exists(path):
            if config.get("overwrite", False):
                shutil.rmtree(path)
            else:
                raise IOError("Path {0} already exists!".format(path))
            
        os.mkdir(path)
    
    try:
        all_best_parameters = []
        for bb in range(B):
            # bootstrap resample particles
            if B > 1:
                p_idx = np.random.randint(Nparticles*B, size=Nparticles)
                b_particles = particles[p_idx]
            else:
                b_particles = particles
            
            try:
                sampler = infer_potential(ln_posterior, p0, steps=config["steps"],
                                          burn_in=config["burn_in"], pool=pool,
                                          args=(config["model_parameters"], 
                                                b_particles, 
                                                satellite, 
                                                t1, t2, resolution))
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
                if config["observational_errors"]:
                    fig,axes = pre_error_particles.plot_r("xyz",
                                            subplots_kwargs=dict(figsize=(12,12)),
                                            scatter_kwargs={"alpha":0.5,"c":"k"})
                    b_particles.plot_r("xyz", axes=axes, scatter_kwargs={"alpha":1.,
                                                                         "c":"#CA0020"})
                else:
                    fig,axes = b_particles.plot_r("xyz", scatter_kwargs={"alpha":0.75,
                                                                         "c":"k"})
    
                fig.savefig(os.path.join(path, "positions_{0}.png".format(bb)))
                
                if config["observational_errors"]:
                    fig,axes = pre_error_particles.plot_v(['vx','vy','vz'],
                                            subplots_kwargs=dict(figsize=(12,12)),
                                            scatter_kwargs={"alpha":0.5,"c":"k"})
                    b_particles.plot_v(['vx','vy','vz'], axes=axes, 
                                     scatter_kwargs={"alpha":1., "c":"#CA0020"})
                else:
                    fig,axes = b_particles.plot_v(['vx','vy','vz'], 
                                                scatter_kwargs={"alpha":0.75, "c":"k"})
    
                fig.savefig(os.path.join(path, "velocities_{0}.png".format(bb)))
                
                # write the sampler to a pickle file
                data_file = os.path.join(path, "sampler_data.pickle")
                sampler.lnprobfn = None
                sampler.pool = None
                fnpickle(sampler, data_file)
                
                # make sexy plots from the sampler data
                fig = plot_sampler_pickle(os.path.join(path,data_file), 
                                          params=config["model_parameters"], 
                                          acceptance_fraction_bounds=(0.15,0.6),
                                          show_true=True, 
                                          param_ranges=param_ranges)
                
                # add the max likelihood estimates to the plots                           
                for ii,param_name in enumerate(config["model_parameters"]):
                    fig.axes[int(2*ii+1)].axhline(best_parameters[ii], 
                                                  color="#CA0020",
                                                  linestyle="--",
                                                  linewidth=2)
                
                fig.savefig(os.path.join(path, "emcee_sampler_{0}.png".format(bb)))
                
                # print MAP values
                idx = sampler.flatlnprobability.argmax()
                best_p = sampler.flatchain[idx]
                print("MAP values: {0}".format(best_p))
                
                # make triangle plot with 5-sigma ranges
                extents = []
                truths = []
                for ii,param in enumerate(config["model_parameters"]):
                    mu = np.median(sampler.flatchain[:,ii])
                    sigma = np.std(sampler.flatchain[:,ii])
                    extents.append((mu-5*sigma,mu+5*sigma))
                    truths.append(_true_params[param])
                    
                fig = triangle.corner(sampler.flatchain,
                                      labels=config["model_parameters"],
                                      extents=extents,
                                      truths=truths,
                                      quantiles=[0.16,0.5,0.84])
                fig.savefig(os.path.join(path, "triangle_{0}.png".format(bb)))
                
    except:
        if config["mpi"]: pool.close()
        raise
    
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
                p = true_params[name].value
            except AttributeError:
                p = true_params[name]
                
            axes[ii].hist(best_p[name], bins=25, histtype="step", color="k", linewidth=2)
            axes[ii].axvline(p, linestyle="--", color="#EF8A62", linewidth=3)
            axes[ii].set_ylabel(name)
            axes[ii].set_ylim(0,20)
        
        fig.savefig(os.path.join(path,"bootstrap_1d.png"))
        
        # Now make 2D plots of the bootstrap results
        if config["observational_errors"]:
            subtitle = r"$\sigma_{{RV}}={0}$ km/s; $\sigma_D={1}\%D$; {2} particles"
            subtitle = subtitle.format(rv_error, d_error, Nparticles)
        else:
            subtitle = "{0} particles".format(Nparticles)
        fig = bootstrap_scatter_plot(best_p, subtitle=subtitle)
        fig.savefig(os.path.join(path,"bootstrap_2d.png"))
    
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
    parser.add_argument("-n", "--name", dest="job_name", default=None, 
                    help="Name of the output.")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    elif args.quiet:
        logging.basicConfig(level=logging.ERROR)
    else:
        logging.basicConfig(level=logging.INFO)
    
    try:
        main(args.file, args.job_name)
        
    except:
        raise
        sys.exit(1)
    
    sys.exit(0)