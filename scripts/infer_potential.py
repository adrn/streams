#!/vega/astro/users/amp2217/yt-x86_64/bin/python
# coding: utf-8

""" Script for using the Rewinder to infer the Galactic host potential """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys
import gc
import copy
import logging
import multiprocessing

# Third-party
import astropy.units as u
from astropy.utils.console import color_print
import emcee
import h5py
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
from streams.dynamics import Particle, Orbit
from streams.inference import (ModelParameter, StreamModel,
                               LogNormalPrior, LogUniformPrior)
import streams.io as s_io
from streams.observation.gaia import gaia_spitzer_errors
import streams.potential as s_potential
from streams.util import _parse_quantity, make_path

global pool
pool = None

# Create logger
logger = logging.getLogger(__name__)

def get_parallel_pool(mpi=False, threads=None):
    """ Get a pool object to pass to emcee for parallel processing.
        If mpi is False and threads is None, pool is None.

        Parameters
        ----------
        mpi : bool
            Use MPI or not. If specified, ignores the threads kwarg.
        threads : int (optional)
            If mpi is False and threads is specified, use a Python
            multiprocessing pool with the specified number of threads.
    """
    # This needs to go here so I don't read in the particle file N times!!
    # get a pool object given the configuration parameters
    if mpi:
        # Initialize the MPI pool
        pool = MPIPool()

        # Make sure the thread we're running on is the master
        if not pool.is_master():
            pool.wait()
            sys.exit(0)
        logger.debug("Running with MPI...")

    elif threads > 1:
        logger.debug("Running with multiprocessing on {} cores..."\
                    .format(threads))
        pool = multiprocessing.Pool(threads)

    else:
        logger.debug("Running serial...")
        pool = None

    return pool

def main(config_file, mpi=False, threads=None, overwrite=False):
    """ TODO: """

    pool = get_parallel_pool(mpi=mpi, threads=threads)

    # read in configurable parameters
    with open(config_file) as f:
        config = yaml.load(f.read())

    # determine the output data path
    path = make_path(config["output_path"],
                     name=config["name"],
                     overwrite=overwrite)
    output_file = os.path.join(path, "inference.hdf5")

    # get the potential object specified from the potential subpackage
    Potential = getattr(s_potential, config["potential"]["class_name"])
    potential = Potential()
    logger.debug("Using potential '{}'..."\
                 .format(config["potential"]["class_name"]))

    # read data from an input HDF5 file
    # TODO: make several input files
    #   - perfect_data.hdf5 (no errors on particles or satellite)
    #   - perfectSatellite_observedParticles.hdf5 (no errors on satellite, gaia+spitzer on particles)
    #   - observedSatellite_observedParticles.hdf5 (errors on everything)
    #   - noSatellite_observedParticles.hdf5 (no satellite data all 0's, observed particles)
    # TODO: knows if hdf5 has 'errors' to return ObservedParticle, else Particle
    s_io.read_hdf5(input_file) # contains stars/satellite info

    ##########################################################################
    # Setting up the Model
    #
    model_parameters = []

    # Potential parameters
    potential_params = config["model_parameters"].get("potential", dict())
    for name,kwargs in potential_params.items():
        # TODO: not so great
        model_p = potential.model_parameter(name, **kwargs)
        logger.debug("Prior on {}: U({},{})".format(name, model_p._ln_prior.a,
                                                          model_p._ln_prior.b))
        model_parameters.append(model_p)

    if particles is an ObservedParticle instance:

        # prior on the time the particle came unbound
        # TODO: get t1,t2 from config file
        # TODO: all observed simulation particles must have tub attribute?
        lo = [t2] * particles.nparticles
        hi = [t1] * particles.nparticles
        prior = LogUniformPrior(lo, hi)
        model_parameters.append(ModelParameter(target=particles,
                                               attr="tub",
                                               ln_prior=prior))

        # TODO: replace below with particles._error_X
        sigmas = np.array([o_particles.errors[n].decompose(usys).value \
                    for n in o_particles.frame.coord_names]).T
        covs = [np.diag(s**2) for s in sigmas]

        prior = LogNormalPrior(np.array(o_particles._X),
                               cov=np.array(covs))
        p = ModelParameter(target=o_particles,
                           attr="_X",
                           ln_prior=prior)
        model_parameters.append(p)

    # Satellite parameters
    if satellite is an ObservedParticle instance:
        # TODO: replace below with satellite._error_X
        sigmas = np.array([o_satellite.errors[n].decompose(usys).value \
                        for n in o_satellite.frame.coord_names])
        covs = [np.diag(s**2) for s in sigmas.T]

        prior = LogNormalPrior(np.array(o_satellite._X),
                               cov=np.array(covs))
        p = ModelParameter(target=o_satellite,
                           attr="_X",
                           ln_prior=prior)
        model_parameters.append(p)

    # now create the model
    # TODO: fix stream model args
    model = StreamModel(potential, satellite, particles,
                        parameters=model_parameters)
    logger.info("Model has {} parameters".format(model.ndim))

    # Emcee!
    # read in the number of walkers to use
    Nwalkers = config.get("walkers", "auto")
    if str(Nwalkers).lower() == "auto":
        Nwalkers = model.ndim*4
    logger.debug("{} walkers".format(Nwalkers))

    # sample starting points for the walkers from the prior
    p0 = model.sample(size=Nwalkers)

    if not os.path.exists(output_file):
        logger.debug("Output file '{}' doesn't exist...".format(output_file))
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
        logger.info("Writing sampler data to '{}'...".format(output_file))

        with h5py.File(output_file, "w") as f:
            #grp = f.create_group("sampler")
            f["chain"] = sampler.chain
            f["flatchain"] = sampler.flatchain
            f["lnprobability"] = sampler.lnprobability
            f["p0"] = p0

    try:
        pool.close()
    except AttributeError:
        pass

    return output_file

def plot_whatever():
    if config.get("make_plots", False):
        logger.info("Generating plots and writing to {}...".format(path))

        # plot observed data / true particles
        extents = [(-180,180), (-90,90), (0.,75.), (-10.,10.), (-10.,10), (-200,200)]
        fig = particles.plot(plot_kwargs=dict(markersize=4, color='k'),
                             hist_kwargs=dict(color='k'),
                             extents=extents)
        fig = o_particles.plot(fig=fig,
                               plot_kwargs=dict(markersize=4, color='r'),
                               hist_kwargs=dict(color='r'),
                               extents=extents)
        fig.savefig(os.path.join(path,"particles_hc.png"))

        extents = [(-75,60)]*3 + [(-300,300)]*3
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

        ix = 0
        if config["model_parameters"].has_key("potential"):
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

            # now make trace plots
            for ii in range(Npp):
                p = pparams[ii]
                fig,ax = plt.subplots(1,1,figsize=(10,6))
                for jj in range(chain.shape[0]):
                    ax.plot(chain[jj,:,ii], drawstyle="steps", color='k', alpha=0.1)

                ax.axhline(p.target._truth, linewidth=4., alpha=0.5,
                           linestyle="--", color="#2B8CBE")
                ax.set_ylim(p._ln_prior.a,p._ln_prior.b)
                fig.suptitle(p.target.latex)
                fig.savefig(os.path.join(path, "{}_trace.png".format(ii)))
                del fig

            ix += Npp

        if config["model_parameters"].has_key("particles"):

            # ---------
            # Now make 7x7 corner plots for each particle
            Nparticles = o_particles.nparticles
            for ii in range(Nparticles):
                # TODO: what if tub not there?
                tub = flatchain[:,Npp+ii]

                start = ix + Nparticles + 6*ii
                stop = start + 6
                OO = flatchain[:,start:stop]

                p = Particle(OO.T, units=o_particles._internal_units,
                             frame=heliocentric)
                p = p.to_units(o_particles._repr_units)

                X_truths = particles._repr_X[ii].tolist()
                X_extents = [(truth-0.2*abs(truth),truth+0.2*abs(truth)) for truth in X_truths]
                extents = [(simulation.t2,simulation.t1)] + X_extents
                truths = [o_particles.tub[ii]] + X_truths

                fig = triangle.corner(np.hstack((tub[:,np.newaxis], p._repr_X)),
                                      labels=['tub','l','b','D',\
                                              r'$\mu_l$', r'$\mu_l$','$v_r$'],
                                      truths=truths,
                                      extents=extents)
                fig.suptitle("Particle {0}".format(ii))
                fig.savefig(os.path.join(path, "particle{0}_corner.png"\
                                         .format(ii)))
                del fig

                # now make trace plots
                fig,axes = plt.subplots(7,1,figsize=(10,14))
                for kk in range(7):
                    for jj in range(chain.shape[0]):
                        if kk == 0:
                            axes[kk].plot(chain[jj,:,Npp+ii],
                                          drawstyle="steps", color='k',
                                          alpha=0.1)
                        else:
                            q = (chain[jj,:,start+kk-1]*o_particles._internal_units[kk-1])
                            axes[kk].plot(q.to(o_particles._repr_units[kk-1]).value,
                                          drawstyle="steps", color='k',
                                          alpha=0.1)

                    axes[kk].axhline(truths[kk], linewidth=4., alpha=0.5,
                               linestyle="--", color="#2B8CBE")
                    axes[kk].set_ylim(extents[kk])

                fig.suptitle("Particle {}".format(ii))
                fig.savefig(os.path.join(path, "particle{}_trace.png".format(ii)))
                del fig

        if config["model_parameters"].has_key("satellite"):
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

    parser.add_argument("-f", "--file", dest="file", required=True,
                        help="Path to the configuration file to run with.")
    parser.add_argument("-o", "--overwrite", dest="overwrite", default=False, action="store_true",
                        help="Overwrite any existing data.")

    # threading
    parser.add_argument("--mpi", dest="mpi", default=False, action="store_true",
                        help="Run with MPI.")
    parser.add_argument("--threads", dest="threads", default=None, type=int,
                        help="Number of multiprocessing threads to run on.")

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
        main(args.file, mpi=args.mpi, threads=args.threads,
             overwrite=args.overwrite)
    except:
        raise
        sys.exit(1)

    sys.exit(0)
