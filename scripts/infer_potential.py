#!/vega/astro/users/amp2217/yt-x86_64/bin/python
# coding: utf-8

""" Script for using the Rewinder to infer the Galactic host potential """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys
import gc
import shutil
import copy
import logging
from datetime import datetime
import multiprocessing

# Third-party
import astropy.units as u
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
from streams.coordinates.frame import heliocentric, galactocentric
from streams.inference import (ModelParameter, StreamModel,
                               LogNormalPrior, LogUniformPrior)
import streams.io as s_io
from streams.observation.gaia import gaia_spitzer_errors
import streams.potential as s_potential
from streams.util import _parse_quantity

global pool
pool = None

# Create logger
logger = logging.getLogger(__name__)

def make_path(config):

    try:
        path = config["output_path"]
    except KeyError:
        raise ValueError("You must specify 'output_path' in the config file.")

    iso_now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    path = os.path.join(path, config.get("name", iso_now))

    if os.path.exists(path) and config.get("overwrite", False):
        shutil.rmtree(path)

    if not os.path.exists(path):
        os.mkdir(path)

    return path

def main(config_file, job_name=None):
    """ TODO: """

    # read in configurable parameters
    with open(config_file) as f:
        config = yaml.load(f.read())

    # This needs to go here so I don't read in the particle file N times!!
    # get a pool object given the configuration parameters
    if config.get("mpi", False):
        # Initialize the MPI pool
        pool = MPIPool()

        # Make sure the thread we're running on is the master
        if not pool.is_master():
            pool.wait()
            sys.exit(0)
        logger.debug("Running with MPI.")

    elif config.get("threads", 0) > 1:
        logger.debug("Running with multiprocessing on {} cores."\
                    .format(config["threads"]))
        pool = multiprocessing.Pool(config["threads"])

    else:
        logger.debug("Running serial.")
        pool = None

    # determine the output data path
    path = make_path(config)
    chain_file = os.path.join(path, "chain.npy")
    flatchain_file = os.path.join(path, "flatchain.npy")
    lnprob_file = os.path.join(path, "lnprobability.npy")

    make_plots = config.get("make_plots", False)
    if make_plots:
        logger.info("Will make plots and save to '{0}'...".format(path))
    else:
        logger.info("OK fine, I won't make plots...")

    ##########################################################################
    # Potential
    #
    # get the potential object specified from the potential subpackage
    Potential = getattr(s_potential, config["potential"]["class_name"])
    potential = Potential()
    logger.debug("Using potential '{}'..."\
                 .format(config["potential"]["class_name"]))

    ##########################################################################
    # Simulation data
    #
    # read the simulation data from the specified class
    np.random.seed(config["seed"])
    Simulation = getattr(s_io, config["data"]["class_name"])
    simulation = Simulation(**config["data"].get("kwargs", dict()))

    # read particles from the simulation class
    particles = simulation.particles(N=config["particles"]["N"],
                                expr=config["particles"]["selection_expr"])
    particles = particles.to_frame(heliocentric)

    logger.debug("Read in {} particles with expr='{}'"\
                 .format(particles.nparticles, particles.expr))

    # read the satellite position
    satellite = simulation.satellite()
    satellite = satellite.to_frame(heliocentric)
    logger.debug("Read in present position of satellite {}..."\
                 .format(satellite))

    gc.collect()

    ##########################################################################
    # Setting up the Model
    #
    model_parameters = []

    # Potential parameters
    potential_params = config["model_parameters"].get("potential", dict())
    for name,kwargs in potential_params.items():
        model_p = potential.model_parameter(name, **kwargs)
        model_parameters.append(model_p)

    # Particle parameters
    if config["model_parameters"].has_key("particles"):
        # Observational errors
        # first get the Gaia + Spitzer errors as default
        particle_errors = gaia_spitzer_errors(particles)

        particle_config = config["model_parameters"]["particles"]
        errors = particle_config["_X"].get("errors",dict())

        # now get errors specified by user in the yml configuration
        factor = errors.get("factor", 1.)
        for k,v in errors.items():
            if not particle_errors.has_key(k):
                logger.debug("Skipping error key {} because not found "
                             "in particle_errors...".format(k))
                continue

            err = _parse_quantity(v)
            logger.debug("Dimension {}, error {}...".format(k, err))

            if err.unit == u.dimensionless_unscaled:
                # fractional error
                particle_errors[k] = err.value * particles[k]
            else:
                particle_errors[k] = np.ones_like(particle_errors[k].value) * err

        o_particles = particles.observe(particle_errors)
        # now has o_particles.errors["D"] etc.

        sigmas = np.array([o_particles.errors[n].decompose(usys).value \
                    for n in o_particles.frame.coord_names]).T
        covs = [np.diag(s**2) for s in sigmas]

        prior = LogNormalPrior(np.array(o_particles._X),
                               cov=np.array(covs))
        p = ModelParameter(target=o_particles,
                           attr="_X",
                           ln_prior=prior)
        model_parameters.append(p)

        # time unbound / escape time (tub)
        if "tub" in particle_config.keys():
            lo = [simulation.t2] * particles.nparticles
            hi = [simulation.t1] * particles.nparticles
            prior = LogUniformPrior(lo, hi)
            model_parameters.append(ModelParameter(target=particles,
                                                   attr="tub",
                                                   ln_prior=prior))
    else:
        o_particles = particles

    # Satellite parameters
    if config["model_parameters"].has_key("satellite"):
        satellite_config = config["model_parameters"]["satellite"]
        errors = satellite_config["_X"].get("errors",dict())

        # first get the Gaia + Spitzer errors as default
        satellite_errors = gaia_spitzer_errors(satellite)

        # now get errors specified by user in the yml configuration
        factor = errors.get("factor", 1.)
        for k,v in errors.items():
            if not satellite_errors.has_key(k):
                logger.debug("Skipping error key {} because not found "
                             "in satellite_errors...".format(k))
                continue

            err = _parse_quantity(v)
            logger.debug("Dimension {}, error {}...".format(k, err))

            if err.unit == u.dimensionless_unscaled:
                # fractional error
                satellite_errors[k] = err.value * satellite[k]
            else:
                satellite_errors[k] = np.ones_like(satellite_errors[k].value) * err

        # satellite has different errors from individual stars...
        # from: http://iopscience.iop.org/1538-4357/618/1/L25/pdf/18807.web.pdf

        o_satellite = satellite.observe(satellite_errors)
        sigmas = np.array([o_satellite.errors[n].decompose(usys).value \
                        for n in o_satellite.frame.coord_names])
        covs = [np.diag(s**2) for s in sigmas[np.newaxis]]

        prior = LogNormalPrior(np.array(o_satellite._X),
                               cov=np.array(covs))
        p = ModelParameter(target=o_satellite,
                           attr="_X",
                           ln_prior=prior)
        model_parameters.append(p)

    else:
        o_satellite = satellite

    # now create the model
    model = StreamModel(potential, simulation, o_satellite, o_particles,
                        parameters=model_parameters)
    logger.info("Model has {} parameters".format(model.ndim))

    ##########################################################################
    # Emcee!
    #

    # read in the number of walkers to use
    Nwalkers = config.get("walkers", "auto")
    if str(Nwalkers).lower() == "auto":
        Nwalkers = model.ndim*2 + 2
    logger.debug("{} walkers".format(Nwalkers))

    # sample starting points for the walkers from the prior
    p0 = model.sample(size=Nwalkers)

    if not os.path.exists(chain_file):
        logger.debug("Cache files don't exist...")
        logger.debug("Initializing sampler...")
        sampler = emcee.EnsembleSampler(Nwalkers, model.ndim, model,
                                        pool=pool)

        Nburn_in = config.get("burn_in", 0)
        Nsteps = config["steps"]
        if Nburn_in > 0:
            logger.info("Burning in sampler for {} steps...".format(Nburn_in))
            pos, xx, yy = sampler.run_mcmc(p0, Nburn_in)
            sampler.reset()
        else:
            pos = p0

        logger.info("Running sampler for {} steps...".format(Nsteps))
        pos, prob, state = sampler.run_mcmc(pos, Nsteps)

        # write the sampler data to numpy save files
        logger.info("Writing sampler data to files in {}".format(path))
        np.save(chain_file, sampler.chain)
        np.save(flatchain_file, sampler.flatchain)
        np.save(lnprob_file, sampler.lnprobability)

        chain = np.array(sampler.chain)
        flatchain = np.array(sampler.flatchain)
        lnprobability = np.array(sampler.lnprobability)

        del sampler

    else:
        logger.debug("Cache files exist, reading in save files...")
        chain = np.load(chain_file)
        flatchain = np.load(flatchain_file)
        lnprobability = np.load(lnprob_file)

    try:
        pool.close()
    except AttributeError:
        pass

    if make_plots:

        # plot observed data / true particles
        fig = particles.plot(plot_kwargs=dict(markersize=4, color='k'),
                             hist_kwargs=dict(color='k'))
        fig = o_particles.plot(fig=fig,
                               plot_kwargs=dict(markersize=4, color='r'),
                               hist_kwargs=dict(color='r'))
        fig.savefig(os.path.join(path,"particles_hc.png"))

        extents = [(-75,60)]*3 + [(-200,200)]*3
        fig = particles.to_frame(galactocentric)\
                       .plot(plot_kwargs=dict(markersize=4, color='k'),
                             hist_kwargs=dict(color='k'),
                             extents=extents)
        fig = o_particles.to_frame(galactocentric)\
                         .plot(fig=fig,
                               plot_kwargs=dict(markersize=4, color='r'),
                               hist_kwargs=dict(color='r'),
                               extents=extents)
        fig.savefig(os.path.join(path,"particles_gc.png"))

        # Make a corner plot for the potential parameters
        Npp = len(potential_params) # number of potential parameters
        pparams = model.parameters[:Npp]

        # First, just samples from the priors:
        fig = triangle.corner(p0[:,:Npp],
                    truths=[p.target._truth for p in pparams],
                    extents=[(p._ln_prior.a,p._ln_prior.b) for p in pparams],
                    labels=[p.target.latex for p in pparams],
                    plot_kwargs=dict(color='k'),
                    hist_kwargs=dict(color='k'))
        fig.savefig(os.path.join(path, "potential_corner_prior.png"))

        # Now the actual chains, extents from the priors
        fig = triangle.corner(flatchain[:,:Npp],
                    truths=[p.target._truth for p in pparams],
                    extents=[(p._ln_prior.a,p._ln_prior.b) for p in pparams],
                    labels=[p.target.latex for p in pparams],
                    plot_kwargs=dict(color='k'),
                    hist_kwargs=dict(color='k'))
        fig.savefig(os.path.join(path, "potential_corner.png"))

        sys.exit(0)

        # ---------
        # Now make 7x7 corner plots for each particle
        Nparticles = o_particles.nparticles
        for ii in range(Nparticles):
            tub = flatchain[:,Npp+ii]

            start = Npp + Nparticles + 6*ii
            stop = start + 6
            OO = flatchain[:,start:stop]

            p = Particle(OO.T, units=usys, names=o_particles.names)
            p = p.to_units(o_particles._repr_units)

            truths = particles._repr_X[ii]
            extents = [(t2,t1)]+[(truth-0.2*abs(truth),truth+0.2*abs(truth)) \
                        for truth in truths]

            fig = triangle.corner(np.hstack((tub[:,np.newaxis], p._repr_X)),
                                  labels=['tub','l','b','D',\
                                          r'$\mu_l$', r'$\mu_l$','$v_r$'],
                                  truths=truths,
                                  extents=extents)
            fig.suptitle("Particle {0}".format(ii))
            fig.savefig(os.path.join(path, "particle_{0}_corner.png"\
                                     .format(ii)))
            del fig

        # ---------
        # Now make 6x6 corner plot for satellite
        start = Npp + Nparticles + 6*Nparticles
        stop = start + 6

        OO = flatchain[:,start:stop]
        truths = np.squeeze(satellite._X)

        # so = np.squeeze(sat_obs_data)
        # se = np.squeeze(sat_obs_error)
        # extents = zip(so - 3*se, \
        #               so + 3*se)

        # First, just samples from the priors:
        fig1 = triangle.corner(p0[:,start:stop],
                    truths=truths,
                    labels=['l','b','D',\
                            r'$\mu_l$', r'$\mu_l$','$v_r$'],)
        fig1.savefig(os.path.join(path, "satellite_corner_prior.png"))

        fig = triangle.corner(OO,
                              labels=['l','b','D',\
                                      r'$\mu_l$', r'$\mu_l$','$v_r$'],
                              truths=truths)
        for ii,ax in enumerate(np.array(fig1.axes)):
            fig.axes[ii].set_xlim(ax.get_xlim())
            fig.axes[ii].set_ylim(ax.get_ylim())
        fig.suptitle("Satellite")
        fig.savefig(os.path.join(path, "satellite_corner.png"))

        sys.exit(0)
        return

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
