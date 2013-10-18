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
from streams.observation.gaia import add_uncertainties_to_particles, \
                                     rr_lyrae_observational_errors
from streams.inference import StatisticalModel, back_integrate_likelihood
from streams.plot import emcee_plot, bootstrap_scatter_plot

global pool
pool = None

# Create logger
logger = logging.getLogger(__name__)

# TODO: this is A HACK
from streams.coordinates import gc_to_hel
def particles_to_heliocentric(particles):
    # Transform to heliocentric coordinates
    return gc_to_hel(particles.r[:,0], particles.r[:,1], particles.r[:,2],
                     particles.v[:,0], particles.v[:,1], particles.v[:,2])

def main(config_file, job_name=None):
    
    # Read in simulation parameters from config file
    config = read(config_file)
    
    # Read particles and posterior from whatever simulation
    # TODO: this could be much cleaner...
    if config['particle_source'] == 'lm10':
        from streams.io.lm10 import particles_today, satellite_today, time
        from streams.potential.lm10 import true_params, _true_params, param_to_latex
        from streams.potential.lm10 import LawMajewski2010 as Potential

    elif config['particle_source'] == 'pal5':
        from streams.io.pal5 import particles_today, satellite_today, time
        from streams.potential.pal5 import true_params, _true_params, param_to_latex
        from streams.potential.pal5 import Palomar5 as Potential
    elif 'sgr' in config['particle_source']:
        # one of Kathryn's Sgr simulations, in the form of Sgr2.5e8
        m = config['particle_source'][3:]
        from streams.io.sgr import mass_selector
        particles_today, satellite_today, time = mass_selector(m)
        from streams.potential.lm10 import true_params, _true_params, param_to_latex
        from streams.potential.lm10 import LawMajewski2010 as Potential
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
        
        logger.debug("Running with MPI...")
    else:
        if config.has_key("threads") and config["threads"] > 1:
            pool = multiprocessing.Pool(config["threads"])
        else:
            pool = None
        
        logger.debug("Running with multiprocessing...")
    
    logger.debug("Particle source: {0}...".format(config['particle_source']))
    
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
    
    logger.debug("Read in {0} particles...".format(Nparticles))
    
    if config["observational_errors"]:
        rv_error = config.get("radial_velocity_error", None)
        d_error = config.get("distance_error_percent", None)
        mu_error = config.get("proper_motion_error", None)
        
        logger.debug("Adding observational errors...")
        
        pre_error_particles = copy.copy(particles)
        data_particles = add_uncertainties_to_particles(particles, 
                                            radial_velocity_error=rv_error,
                                            distance_error_percent=d_error,
                                            proper_motion_error=mu_error)
        data = particles_to_heliocentric(data_particles)
        data_errors = rr_lyrae_observational_errors(*data)
    
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
    logger.debug("Running with a timestep resolution of {0}...".format(resolution))
    
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
        
        logger.debug("Will save plots to {0}...".format(path))
    
    try:
        all_best_parameters = []
        for bb in range(B):
            # bootstrap resample particles
            # TODO: broken with B > 1
            # TODO: remove bootstrap?!
            if B > 1:
                p_idx = np.random.randint(Nparticles*B, size=Nparticles)
                b_particles = particles[p_idx]
            else:
                b_particles = particles
            
            largs = (config["model_parameters"], satellite, \
                     data, data_errors, Potential, t1, t2)
            stat_model = StatisticalModel(config["model_parameters"], 
                                          back_integrate_likelihood,
                                          likelihood_args=largs,
                                          parameter_bounds=parameter_bounds)

            try:
                sampler = stat_model.run(p0, nsteps=config["steps"], 
                                         nburn=config["burn_in"],
                                         pool=pool)
            except:
                if config["mpi"]: pool.close()
                raise
            
            # TODO: re-write this...
            #best_parameters = max_likelihood_parameters(sampler)
            best_parameters = []
            
            if len(best_parameters) == 0:
                best_parameters = np.array([0.]*len(config['model_parameters']))
                
            all_best_parameters.append(best_parameters)
            
            logger.debug("Making particle position / velocity plots...")

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
    
                logger.debug("Saving sampler data...")

                # write the sampler to a pickle file
                data_file = os.path.join(path, "sampler_data.pickle")
                sampler.lnprobfn = None
                sampler.pool = None
                fnpickle(sampler, data_file)
                
                logger.debug("Making emcee trace plot...")

                # make sexy plots from the sampler data
                truths = [_true_params[p] for p in config["model_parameters"]]
                labels = [param_to_latex[p] for p in config["model_parameters"]]
                fig = emcee_plot(sampler.chain, 
                                 labels=labels, 
                                 truths=truths)
                
                fig.savefig(os.path.join(path, "emcee_trace_{0}.png".format(bb)))
                
                # print MAP values
                idx = sampler.flatlnprobability.argmax()
                best_p = sampler.flatchain[idx]
                logger.info("MAP values: {0}".format(best_p))
                
                logger.debug("Making triangle plot...")

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
