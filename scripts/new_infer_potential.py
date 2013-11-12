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
import astropy.units as u
from astropy.io.misc import fnpickle, fnunpickle
from astropy.utils.console import color_print
import emcee
import matplotlib.pyplot as plt
import numpy as np
np.seterr(all="ignore")
import scipy
scipy.seterr(all="ignore")
import triangle
import yaml

try:
    from emcee.utils import MPIPool
except ImportError:
    color_print("Failed to import MPIPool from emcee! MPI functionality "
                "won't work.", "yellow")

# Project
from streams import usys
from streams.coordinates import _gc_to_hel, _hel_to_gc
from streams.inference import (ModelParameter, StreamModel,
                               LogNormalPrior, LogUniformPrior)
from streams.io.sgr import mass_selector
from streams.observation.gaia import RRLyraeErrorModel
import streams.potential as sp
from streams.simulation import config

global pool
pool = None

# Create logger
logger = logging.getLogger(__name__)

def _null(*args, **kwargs):
    return 0.

def get_pool(config):
    """ Given a config structure, return an MPIPool, a Python
        multiprocessing.Pool, or None.
    """

    if config.get("mpi", False):
        # Initialize the MPI pool
        pool = MPIPool()

        # Make sure the thread we're running on is the master
        if not pool.is_master():
            pool.wait()
            sys.exit(0)
        logger.info("Running with MPI.")

    elif config.get("threads", 0) > 1:
        logger.info("Running with multiprocessing on {} cores."\
                    .format(config["threads"]))
        pool = multiprocessing.Pool(config["threads"])
    else:
        logger.info("Running serial...parallel panda is sad :(")
        pool = None

    return pool

def read_simulation(config):
    """ TODO: """

    if config["simulation"]["source"] == "sgr":
        m = config["simulation"]["satellite_mass"]
        particles_today, satellite_today, time = mass_selector(m)
    elif config["simulation"]["source"] == "lm10":
        #particles_today, satellite_today, time = mass_selector(config["mass"])
        # TODO
        pass
    else:
        raise ValueError("Invalid particle_source: {particle_source}"\
                         .format(config))

    particles = particles_today(N=config["particles"]["N"],
                                expr=config["particles"]["selection_expr"])
    satellite = satellite_today()
    t1,t2 = time()

    return t1,t2,satellite,particles

def make_path(config):

    try:
        path = config["output_path"]
    except KeyError:
        raise ValueError("You must specify 'output_path' in the config file.")

    iso_now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    path = os.path.join(path, config.get("name", iso_now))
    logger.info("Will write output to '{0}'...".format(path))

    if os.path.exists(path):
        logger.debug("...output path exists.".format(path))
        if config.get("overwrite", False):
            logger.debug("...Overwrite=True, deleting path")
            shutil.rmtree(path)
        else:
            raise IOError("Path {0} already exists!".format(path))

    logger.debug("...creating path now.")
    os.mkdir(path)
    return path

def main(config_file, job_name=None):
    """ TODO: """

    # read in configurable parameters
    with open(config_file) as f:
        config = yaml.load(f.read())

    # get a pool object given the configuration parameters
    pool = get_pool(config)

    # determine the output data path
    path = make_path(config)

    make_plots = config.get("make_plots", False)
    sampler_file = os.path.join(path, "sampler_data.pickle")

    # get the potential object specified
    Potential = getattr(sp, config["potential"]["class_name"])
    potential = Potential()
    logger.debug("Using {} potential...".format(potential))

    # Actually get simulation data
    np.random.seed(config["seed"])
    t1,t2,satellite,_particles = read_simulation(config)

    # TODO: right now error specification in yml doesn't propagate
    if config.has_key("errors"):
        factor = config["errors"].get("global_factor", 1.)
        error_model = RRLyraeErrorModel(units=usys,
                                        factor=factor)
        obs_data, obs_error = _particles.observe(error_model)

    # now create the model and start adding model parameters
    model = StreamModel(potential, satellite, _particles,
                        obs_data, obs_error)

    # first add the potential parameters
    potential_params = config["potential"].get("parameters", dict())
    for name,meta in potential_params.items():
        p = getattr(potential, name)

        if meta.has_key("range"):
            # TODO: fix when astropy fixed...
            #lo,hi = map(u.Quantity, meta["range"])
            lo_hi = []
            for ii in range(2):
                try:
                    val,unit = meta["range"][ii].split()
                except AttributeError:
                    val = meta["range"][ii]
                    unit = u.dimensionless_unscaled

                lo_hi.append(u.Quantity(float(val), unit))
            lo, hi = lo_hi

            prior = LogUniformPrior(lo.decompose(usys).value,
                                    hi.decompose(usys).value)
        else:
            prior = LogPrior()

        model.parameters.append(ModelParameter(target=p,
                                               attr="_value",
                                               ln_prior=prior))

    # now add particle parameters
    particle_params = config["particles"].get("parameters", [])

    try:
        Nparticles = config["particles"]["N"]
    except KeyError:
        raise ValueError("Must specify number of partices in config file!")

    # time unbound / escape time (tub)
    if "tub" in particle_params:
        lo = [t2] * len(_particles)
        hi = [t1] * len(_particles)
        prior = LogUniformPrior(lo, hi)
        model.parameters.append(ModelParameter(target=_particles,
                                               attr="tub",
                                               ln_prior=prior))

    # here I monte carlo transform the error distribution from observed
    #   to cartesian, then take np.cov and use that for the gaussian prior
    O = np.array([np.random.normal(obs_data, obs_error) \
                    for ii in range(1000)])
    X = _hel_to_gc(O)

    obs_error_gc = []
    for ii in range(Nparticles):
        obs_error_gc.append(np.cov(X[:,ii].T))

    obs_error_gc = np.array(obs_error_gc)
    obs_data_gc = _hel_to_gc(obs_data)

    # TODO: plot observed data

    # true positions of particles (flat_X)
    if "_X" in particle_params:
        prior = LogNormalPrior(obs_data_gc, cov=obs_error_gc)
        p = ModelParameter(target=_particles, attr="_X", ln_prior=prior)
        p.ln_prior = _null # THIS IS A HACK
        model.parameters.append(p)

    # check to see if the satellite position is to be inferred
    # TODO: infer the damn satellite position!

    # read in the number of walkers to use
    Nwalkers = config.get("walkers", "auto")
    if str(Nwalkers).lower() == "auto":
        Nwalkers = model.ndim*2 + 2

    p0 = model.sample(size=Nwalkers)

    if not os.path.exists(sampler_file):
        sampler = emcee.EnsembleSampler(Nwalkers, model.ndim, model,
                                        args=(t1, t2, -1.),
                                        pool=pool)

        Nburn_in = config.get("burn_in", 0)
        Nsteps = config["steps"]
        if Nburn_in > 0:
            pos, xx, yy = sampler.run_mcmc(p0, Nburn_in)
            sampler.reset()
        else:
            pos = p0

        pos, prob, state = sampler.run_mcmc(pos, Nsteps)

        # write the sampler to a pickle file
        sampler.lnprobfn = None
        sampler.pool = None
        fnpickle(sampler, sampler_file)
    else:
        sampler = fnunpickle(sampler_file)

    pool.close()

    if make_plots:

        # Make a corner plot for the potential parameters
        Npp = len(potential_params) # number of potential parameters
        pparams = model.parameters[:Npp]

        # First, just samples from the priors:
        fig = triangle.corner(p0[:,:Npp],
                    truths=[p.target._truth for p in pparams],
                    extents=[(p._ln_prior.a,p._ln_prior.b) for p in pparams],
                    labels=[p.target.latex for p in pparams])
        fig.savefig(os.path.join(path, "potential_corner_prior.png"))

        # Now the actual chains, extents from the priors
        fig = triangle.corner(sampler.flatchain[:,:Npp],
                    truths=[p.target._truth for p in pparams],
                    extents=[(p._ln_prior.a,p._ln_prior.b) for p in pparams],
                    labels=[p.target.latex for p in pparams])
        fig.savefig(os.path.join(path, "potential_corner.png"))

        # ---------
        # Now make 7x7 corner plots for each particle
        # TODO: need these plots to fail if not specified in config...
        for ii in range(Nparticles):
            tub = sampler.flatchain[:,Npp+ii]

            start = Npp+Nparticles + 6*ii
            stop = start + 6
            XX = sampler.flatchain[:,start:stop]

            fig = triangle.corner(np.hstack((tub[:,np.newaxis], XX)),
                                  labels=['tub','x','y','z','vx','vy','vz'])
            fig.suptitle("Particle {0}".format(ii))
            fig.savefig(os.path.join(path, "particle_{0}_corner.png"\
                                     .format(ii)))
            plt.clf()

        # fig = plt.figure(figsize=(6,4))
        # ax = fig.add_subplot(111)
        # for jj in range(Npp) + [10]:
        #     ax.cla()
        #     for ii in range(Nwalkers):
        #         ax.plot(sampler.chain[ii,:,jj], drawstyle='step')

        #     fig.savefig(os.path.join(path, "{0}.png".format(jj)))

    sys.exit(0)

if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser(description="")
    parser.add_argument("-v", "--verbose", action="store_true",
                        dest="verbose", default=False,
                        help="Be chatty! (default = False)")
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
