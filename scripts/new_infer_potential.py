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
from streams.io import *
from streams.observation.gaia import gaia_spitzer_errors
import streams.potential as sp
from streams.util import make_path

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
        simulation = SgrSimulation(mass=m)

    elif config["simulation"]["source"] == "lm10":
        simulation = LM10Simulation()

    elif config["simulation"]["source"] == "orphan":
        simulation = OrphanSimulation()

    else:
        raise ValueError("Invalid particle_source: {particle_source}"\
                         .format(config))

    return simulation

def _parse_quantity(q):
    try:
        val,unit = q.split()
    except AttributeError:
        val = q
        unit = u.dimensionless_unscaled

    return u.Quantity(float(val), unit)

def main(config_file, job_name=None):
    """ TODO: """

    # read in configurable parameters
    with open(config_file) as f:
        config = yaml.load(f.read())

    # TODO: write a separate script just for plotting...
    # get a pool object given the configuration parameters
    pool = get_pool(config)

    # determine the output data path
    path = make_path(config)
    sampler_file = os.path.join(path, "sampler_data.pickle")

    make_plots = config.get("make_plots", False)
    if make_plots:
        logger.info("Will write output to '{0}'...".format(path))
    else:
        logger.info("OK fine, I won't write anything to disk...")

    # get the potential object specified
    Potential = getattr(sp, config["potential"]["class_name"])
    potential = Potential()
    logger.debug("Using {} potential...".format(potential))

    # get simulation data
    np.random.seed(config["seed"])
    expr = config["particles"]["selection_expr"]

    simulation = read_simulation(config)
    particles = simulation.particles(N=config["particles"]["N"],
                                     expr=expr)
    particles = particles.to_frame('heliocentric')
    particle_errors = gaia_spitzer_errors(particles)

    satellite = simulation.satellite()
    satellite = satellite.to_frame('heliocentric')
    # Note: now particles and satellite are in heliocentric coordinates!

    # get errors specified by user in config
    factor = config["errors"].get("global_factor", 1.)
    for k,v in config["errors"].items():
        if k == "global_factor": continue

        assert particle_errors.has_key(k)
        err = _parse_quantity(v)

        if err.unit == u.dimensionless_unscaled:
            # fractional error
            particle_errors[k] = err.value * particles[k]
        else:
            particle_errors[k] = np.ones_like(particle_errors[k].value) * err

    o_particles = particles.observe(particle_errors)
    logger.debug("Particles: {}".format(o_particles))
    # now has o_particles.errors["D"] etc.

    # satellite has different errors from individual stars...
    # from: http://iopscience.iop.org/1538-4357/618/1/L25/pdf/18807.web.pdf
    satellite_errors = dict(l=10*u.milliarcsecond,
                            b=10*u.milliarcsecond,
                            D=1.2*u.kpc,
                            mul=1.2*u.mas/u.yr,
                            mub=1.2*u.mas/u.yr,
                            vr=5*u.km/u.s)
    o_satellite = satellite.observe(satellite_errors)
    logger.debug("Satellite: {}".format(o_satellite))

    model_parameters = []
    ##########################################################################
    # Potential parameters
    #
    potential_params = config["potential"].get("parameters", dict())
    for name,meta in potential_params.items():
        p = getattr(potential, name)

        if meta.has_key("range"):
            # TODO: fix when astropy fixed...
            #lo,hi = map(u.Quantity, meta["range"])
            lo = _parse_quantity(meta["range"][0])
            hi = _parse_quantity(meta["range"][1])
            prior = LogUniformPrior(lo.decompose(usys).value,
                                    hi.decompose(usys).value)
        else:
            prior = LogPrior()

        model_parameters.append(ModelParameter(target=p,
                                               attr="_value",
                                               ln_prior=prior))

    ##########################################################################
    # Particle parameters
    #
    particle_params = config["particles"].get("parameters", [])

    try:
        Nparticles = config["particles"]["N"]
    except KeyError:
        raise ValueError("Must specify number of partices in config file!")

    # time unbound / escape time (tub)
    if "tub" in particle_params:
        lo = [simulation.t2] * particles.nparticles
        hi = [simulation.t1] * particles.nparticles
        prior = LogUniformPrior(lo, hi)
        model_parameters.append(ModelParameter(target=particles,
                                               attr="tub",
                                               ln_prior=prior))

    if "_X" in particle_params:
        sigmas =[o_particles.errors[n].decompose(usys)\
                    for n in o_particles.names]
        prior = LogNormalPrior(o_particles._X,
                               sigmas=sigmas)
        p = ModelParameter(target=o_particles,
                           attr="_X",
                           ln_prior=prior)
        p.ln_prior = _null # THIS IS A HACK?
        model_parameters.append(p)

    return

    ##########################################################################
    # Satellite parameters
    #
    satellite_config = config.get("satellite", dict())
    satellite_params = satellite_config.get("parameters", [])

    # here I monte carlo transform the error distribution from observed
    #   to cartesian, then take np.cov and use that to sample new satellite
    #   positions
    O = np.array([np.random.normal(sat_obs_data, sat_obs_error) \
                    for ii in range(1000)])
    X = _hel_to_gc(O)

    sat_obs_error_gc = np.array([np.cov(X[:,0].T)])
    sat_obs_data_gc = _hel_to_gc(sat_obs_data)

    # true position of the satellite
    if "_X" in satellite_params:
        satellite._X = _hel_to_gc(sat_obs_data)
        satellite.obs_data = sat_obs_data
        satellite.obs_error = sat_obs_error

        prior = LogNormalPrior(sat_obs_data_gc, cov=sat_obs_error_gc)
        p = ModelParameter(target=satellite, attr="_X", ln_prior=prior)
        p.ln_prior = _null # THIS IS A HACK?
        model_parameters.append(p)

    # now create the model
    model = StreamModel(potential, satellite, particles,
                        parameters=model_parameters)

    # read in the number of walkers to use
    Nwalkers = config.get("walkers", "auto")
    if str(Nwalkers).lower() == "auto":
        Nwalkers = model.ndim*2 + 2

    # sample starting points for the walkers from the prior
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

        pool.close()

    else:
        sampler = fnunpickle(sampler_file)

    if make_plots:

        # TODO: plot observed data / true particles

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
        true_obs_data = _gc_to_hel(_particles._X)
        for ii in range(Nparticles):
            tub = sampler.flatchain[:,Npp+ii]

            start = Npp+Nparticles + 6*ii
            stop = start + 6
            XX = sampler.flatchain[:,start:stop]
            OO = _gc_to_hel(XX)
            truths = np.append(_particles.tub[ii], true_obs_data[ii])

            extents = zip(obs_data[ii] - 3*obs_error[ii], \
                          obs_data[ii] + 3*obs_error[ii])
            extents = [(t2,t1)] + extents
            fig = triangle.corner(np.hstack((tub[:,np.newaxis], OO)),
                                  labels=['tub','l','b','D',\
                                          r'$\mu_l$', r'$\mu_l$','$v_r$'],
                                  extents=extents,
                                  truths=truths)
            fig.suptitle("Particle {0}".format(ii))
            fig.savefig(os.path.join(path, "particle_{0}_corner.png"\
                                     .format(ii)))
            plt.clf()

        # ---------
        # Now make 6x6 corner plot for satellite
        true_obs_data = _gc_to_hel(_satellite._X)
        start = Npp + Nparticles + 6*Nparticles
        stop = start + 6

        XX = sampler.flatchain[:,start:stop]
        OO = _gc_to_hel(XX)
        truths = np.squeeze(true_obs_data)

        so = np.squeeze(sat_obs_data)
        se = np.squeeze(sat_obs_error)
        extents = zip(so - 3*se, \
                      so + 3*se)

        # First, just samples from the priors:
        fig = triangle.corner(p0[:,start:stop],
                    truths=truths,
                    extents=extents,
                    labels=['l','b','D',\
                            r'$\mu_l$', r'$\mu_l$','$v_r$'],)
        fig.savefig(os.path.join(path, "satellite_corner_prior.png"))

        fig = triangle.corner(OO,
                              labels=['l','b','D',\
                                      r'$\mu_l$', r'$\mu_l$','$v_r$'],
                              extents=extents,
                              truths=truths)
        fig.suptitle("Satellite")
        fig.savefig(os.path.join(path, "satellite_corner.png"))

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
