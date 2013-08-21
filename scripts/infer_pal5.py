#!/hpc/astro/users/amp2217/yt-x86_64/bin/python
# coding: utf-8

""" In this module, I'll try to infer the Palomar 5 potential that Andreas
    used in an Nbody simulation of the stream.
"""

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys
import logging
import shutil
import logging
from datetime import datetime
import multiprocessing

# Third-party
try:
    from emcee.utils import MPIPool
except ImportError:
    print("Failed to import MPIPool from emcee! MPI functionality won't work.")
    
import matplotlib.pyplot as plt
import numpy as np
import astropy.units as u
from astropy.io.misc import fnpickle, fnunpickle
import triangle

# project
from streams.simulation.config import read
from streams.io.pal5 import particles_today, satellite_today, time
from streams.inference.pal5 import ln_likelihood, ln_posterior, param_ranges
from streams.potential.pal5 import true_params, _true_params
from streams.integrate.satellite_particles import SatelliteParticleIntegrator
from streams.inference import infer_potential, max_likelihood_parameters
from streams.plot import plot_sampler_pickle, bootstrap_scatter_plot

#from streams.observation.gaia import add_uncertainties_to_particles

def test_left_right():
    l = ln_likelihood([], [], true_particles, satellite, t1, t2, resolution)
    l_L = ln_likelihood([0.7], ['qz'], true_particles, satellite, t1, t2, resolution)
    l_R = ln_likelihood([0.9], ['qz'], true_particles, satellite, t1, t2, resolution)
    print(l_L, l, l_R)

def test_likelihood(fn, ps, frac_bounds=(0.8, 1.2), Nbins=21):
    frac_range = np.linspace(frac_bounds[0],frac_bounds[1],Nbins)
    
    fig = plt.figure(figsize=(12,6))
    for ii,param in enumerate(['qz', 'm']):
        vals = frac_range*true_params[param]
        ls = []
        for val in vals:
            l = fn([val], [param], ps, satellite, t1, t2, resolution)
            ls.append(l)
        
        plt.subplot(1,2,ii+1)
        plt.plot(vals, np.array(ls))
        plt.axvline(_true_params[param])
    
    return fig

global pool
pool = None

# Create logger
logger = logging.getLogger(__name__)

def main(config_file, job_name=None):
    
    # Read in simulation parameters from config file
    config = read(config_file)
    
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
    
    # Read in simulation data
    satellite = satellite_today()
    t1,t2 = time()
    Nparticles = config["particles"]
    particles = particles_today(N=Nparticles*B)
    
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
            p_idx = np.random.randint(Nparticles*B, size=Nparticles)
            b_particles = particles[p_idx]
            
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
                fig,axes = b_particles.plot_r("xyz", scatter_kwargs={"alpha":0.75,
                                                                     "c":"k"})
    
                fig.savefig(os.path.join(path, "positions_{0}.png".format(bb)))
                fig,axes = b_particles.plot_v(['vx','vy','vz'], 
                                            scatter_kwargs={"alpha":0.75, "c":"k"})
    
                fig.savefig(os.path.join(path, "velocities_{0}.png".format(bb)))
                
                # write the sampler to a pickle file
                data_file = os.path.join(path, "sampler_data.pickle")
                sampler.lnprobfn = None
                sampler.pool = None
                fnpickle(sampler, data_file)
                
                fig = triangle.corner(sampler.flatchain, 
                                      labels=config["model_parameters"], 
                                      truths=[_true_params[p] for p in config["model_parameters"]], 
                                      quantiles=[0.16, 0.5, 0.84],
                                      plot_datapoints=False)
                
                fig.savefig(os.path.join(path, "emcee_sampler_{0}.png".format(bb)))
                
                # print MAP values
                idx = sampler.flatlnprobability.argmax()
                best_p = sampler.flatchain[idx]
                print("MAP values: ".format(",".join(best_p)))
                
                # now plot the walker traces
                fig = plot_sampler_pickle(os.path.join(path,data_file), 
                                          params=config["model_parameters"], 
                                          acceptance_fraction_bounds=(0.15,0.6))
                
                # add the max likelihood estimates to the plots                           
                for ii,param_name in enumerate(config["model_parameters"]):
                    fig.axes[int(2*ii+1)].axhline(best_parameters[ii], 
                                                  color="#CA0020",
                                                  linestyle="--",
                                                  linewidth=2)
                
                fig.savefig(os.path.join(path, "emcee_trace_{0}.png".format(bb)))
                
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
    parser.add_argument("-t", "--test", action="store_true", dest="test", 
                    default=False, help="Run tests, exit.")
    
    args = parser.parse_args()
    
    if args.test:
        np.random.seed(44)
        true_particles = particles_today(N=100) # , expr="(Pcol>-1) & (Pcol<8) & (abs(Lmflag)==1)")
        satellite = satellite_today()
        t1,t2 = time()
        resolution = 4.

        fig = test_likelihood(ln_likelihood, true_particles, 
                              frac_bounds=(0.6,1.4), Nbins=5)
        plt.show()
        
        #test_left_right()
        sys.exit(0)
    
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
