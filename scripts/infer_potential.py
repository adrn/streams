# coding: utf-8

""" Script for using the Rewinder to infer the Galactic host potential """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys
import logging
import shutil

# Third-party
import astropy.units as u
from emcee.utils import sample_ball
import matplotlib.pyplot as plt
import numpy as np
import time
import yaml

# Project
from streams import usys
from streams.dynamics import ObservedParticle, Particle
import streams.io as io
import streams.inference as si
import streams.potential as sp
from streams.util import get_pool, _parse_quantity, OrderedDictYAMLLoader

global pool
pool = None

# Create logger
logger = logging.getLogger(__name__)

def make_path(path, overwrite=False):
    """ """
    if os.path.exists(path) and overwrite:
        shutil.rmtree(path)

    if not os.path.exists(path):
        os.makedirs(path)

def main(config_file, mpi, threads, overwrite):
    """ TODO: """

    # get a pool object given the configuration parameters
    # -- This needs to go here so I don't read in the particle file for each thread. --
    pool = get_pool(mpi=mpi, threads=threads)

    # get path to streams project
    streams_path = None
    if os.environ.has_key('STREAMSPATH'):
        streams_path = os.environ["STREAMSPATH"]

    # read in configurable parameters
    with open(config_file) as f:
        config = yaml.load(f.read(), OrderedDictYAMLLoader)

    if config.has_key('streams_path'):
        streams_path = config['streams_path']

    if streams_path is None:
        raise IOError("Streams path not specified.")
    elif not os.path.exists(streams_path):
        raise IOError("Specified streams path '{}' doesn't exist!".format(streams_path))
    logger.debug("Path to streams project: {}".format(streams_path))

    # set other paths relative to top-level streams project
    data_path = os.path.join(streams_path, "data/observed_particles/")
    output_path = os.path.join(streams_path, "plots/infer_potential/", config['name'])
    if config.has_key('output_path'):
        output_path = os.path.join(config['output_path'], config['name'])
    make_path(output_path, overwrite)
    logger.debug("Writing data to:\n\t{}".format(output_path))
    output_file = os.path.join(output_path, "inference.hdf5")

    # load satellite and particle data
    data_file = os.path.join(data_path, config['data_file'])
    logger.debug("Reading particle/satellite data from:\n\t{}".format(data_file))
    d = io.read_hdf5(data_file, nparticles=config.get('nparticles', None))

    true_particles = d['true_particles']
    true_satellite = d['true_satellite']
    nparticles = true_particles.nparticles
    logger.info("Running with {} particles.".format(nparticles))

    # integration stuff
    t1 = float(d["t1"])
    t2 = float(d["t2"])
    dt = config.get("dt", -1.)

    # get the potential object specified from the potential subpackage
    Potential = getattr(sp, config["potential"]["class_name"])
    potential = Potential()
    logger.info("Using potential '{}'...".format(config["potential"]["class_name"]))

    # Define the empty model to add parameters to
    model = si.StreamModel(potential, lnpargs=(t1,t2,dt),
                           true_satellite=true_satellite,
                           true_particles=true_particles)

    # Potential parameters
    potential_params = config["potential"].get("parameters", dict())
    for ii,(name,kwargs) in enumerate(potential_params.items()):
        a,b = kwargs["a"], kwargs["b"]
        p = getattr(potential, name)
        logger.debug("Prior on {}: Uniform({}, {})".format(name, a, b))

        prior = si.LogUniformPrior(_parse_quantity(a).decompose(usys).value,
                                   _parse_quantity(b).decompose(usys).value)
        p = si.ModelParameter(name, prior=prior, truth=potential.parameters[name]._truth)
        model.add_parameter('potential', p)

    # Particle parameters
    try:
        particle_parameters = config['particles']['parameters']
    except KeyError:
        particle_parameters = None

    if particle_parameters is not None:
        particles = d['particles']

        logger.debug("Particle properties added as parameters:")
        if config['particles']['parameters'].has_key('_X'):
            priors = [si.LogNormalPrior(particles._X[ii],particles._error_X[ii])
                        for ii in range(nparticles)]
            X = si.ModelParameter('_X', value=particles._X, prior=priors,
                                  truth=true_particles._X)
            model.add_parameter('particles', X)
            logger.debug("\t\t_X - particle 6D positions today")

        if config['particles']['parameters'].has_key('tub'):
            priors = [si.LogUniformPrior(t2, t1) for ii in range(nparticles)]
            tub = si.ModelParameter('tub', value=np.zeros(nparticles), prior=priors,
                                    truth=true_particles.tub)
            model.add_parameter('particles', tub)
            logger.debug("\t\ttub - unbinding time of particles")

    # Satellite parameters
    try:
        satellite_parameters = config['satellite']['parameters']
    except KeyError:
        satellite_parameters = None

    if satellite_parameters is not None:
        satellite = d['satellite']

        logger.debug("Satellite properties added as parameters:")
        if satellite_parameters.has_key('_X'):
            priors = [si.LogNormalPrior(satellite._X[0],satellite._error_X[0])]
            s_X = si.ModelParameter('_X', value=satellite._X, prior=priors,
                                    truth=true_satellite._X)
            model.add_parameter('satellite', s_X)
            logger.debug("\t\t_X - satellite 6D position today")

    # make sure true posterior value is higher than any randomly sampled value
    logger.debug("Checking posterior values...")
    true_ln_p = model.ln_posterior(model.truths, t1, t2, dt)
    logger.debug("\t\t At truth: {}".format(true_ln_p))
    p0 = model.sample_priors()
    ln_p = model.ln_posterior(p0, t1, t2, dt)
    logger.debug("\t\t At random sample: {}".format(ln_p))
    assert true_ln_p > ln_p

    logger.info("Model has {} parameters".format(model.nparameters))

    # for ii,truth in enumerate(model.truths):
    #     p = model.truths.copy()
    #     vals = truth*np.linspace(0.95, 1.05, 50)
    #     Ls = []
    #     for val in vals:
    #         p[ii] = val
    #         Ls.append(model(p))

    #     plt.clf()
    #     plt.plot(vals, Ls)
    #     plt.axvline(truth)
    #     plt.savefig(os.path.join(output_path, "test_{}.png".format(ii)))


    if os.path.exists(output_file) and overwrite:
        logger.info("Writing over output file '{}'".format(output_file))
        os.remove(output_file)

    if not os.path.exists(output_file):
        logger.info("Output file '{}' doesn't exist, running inference...".format(output_file))

        # emcee parameters
        # read in the number of walkers to use
        nwalkers = config["walkers"]
        nsteps = config["steps"]
        nburn = config.get("burn_in", 0)
        niter = config.get("iterate", 1)

        # sample starting positions
        p0 = model.sample_priors(size=nwalkers)
        ### HACK TO INITIALIZE WALKERS NEAR true tub!
        # for ii in range(c["nparticles"]):
        #     jj = ii + len(c["potential_params"])
        #     p0[:,jj] = np.random.normal(true_tub[ii], 1000., size=c["nwalkers"])

        # get the sampler
        sampler = si.StreamModelSampler(model, nwalkers, pool=pool)

        if nburn > 0:
            logger.info("Burning in sampler for {} steps...".format(nburn))
            pos, xx, yy = sampler.run_mcmc(p0, nburn)
            sampler.reset()
        else:
            pos = p0

        logger.info("Running sampler for {} iterations of {} steps..."\
                    .format(niter, nsteps//niter))

        time0 = time.time()
        for ii in range(niter):
            pos, prob, state = sampler.run_mcmc(pos, nsteps//niter)
            best_pos = sampler.flatchain[sampler.flatlnprobability.argmax()]
            std = np.std(sampler.flatchain, axis=0)
            logger.debug("Walker positions: {}".format(best_pos[:5]))
            logger.debug("Walker std dev: {}".format(std[:5]))
            pos = sample_ball(best_pos, std, size=nwalkers)

        t = time.time() - time0
        logger.debug("Spent {} seconds on sampling...".format(t))

        # write the sampler data to numpy save files
        logger.info("Writing sampler data to '{}'...".format(output_file))
        with h5py.File(output_file, "w") as f:
            f["chain"] = sampler.chain
            f["flatchain"] = sampler.flatchain
            f["lnprobability"] = sampler.lnprobability
            f["p0"] = sampler.p0
    else:
        logger.info("Output file '{}' already exists, not running sampler...".format(output_file))

    return


    if pool is not None:
        pool.close()

    if make_plots:
        with h5py.File(output_file, "r") as f:
            chain = f["chain"].value
            flatchain = f["flatchain"].value
            p0 = f["p0"].value

        logger.info("Making plots and writing to {}...".format(path))

        # plot observed data / true particles
        extents = [(-180,180), (-90,90), (0.,75.), (-10.,10.), (-10.,10), (-300,300)]
        logger.debug("Plotting particles in heliocentric coordinates")
        fig = particles.plot(plot_kwargs=dict(markersize=4, color='k'),
                             hist_kwargs=dict(color='k'),
                             extents=extents)
        fig.savefig(os.path.join(path,"particles_hc.png"))

        extents = [(-85,85)]*3 + [(-300,300)]*3
        logger.debug("Plotting particles in galactocentric coordinates")
        fig = particles.to_frame(galactocentric)\
                       .plot(plot_kwargs=dict(markersize=4, color='k'),
                             hist_kwargs=dict(color='k'),
                             extents=extents)
        fig.savefig(os.path.join(path,"particles_gc.png"))

        # plot the potential parameters
        Npp = len(potential_params)
        if Npp > 0:
            # Make a corner plot for the potential parameters
            pparams = model.parameters[:Npp]

            # First, just samples from the priors:
            logger.debug("Plotting prior over potential parameters")
            fig = triangle.corner(p0[:,:Npp],
                        truths=[p.target._truth for p in pparams],
                        extents=[(p._ln_prior.a,p._ln_prior.b) for p in pparams],
                        labels=[p.target.latex for p in pparams],
                        plot_kwargs=dict(color='k'),
                        hist_kwargs=dict(color='k'))
            fig.savefig(os.path.join(path, "potential_prior.png"))

            # Now the actual chains, extents from the priors
            logger.debug("Plotting posterior over potential parameters")
            fig = triangle.corner(flatchain[:,:Npp],
                        truths=[p.target._truth for p in pparams],
                        extents=[(p._ln_prior.a,p._ln_prior.b) for p in pparams],
                        labels=[p.target.latex for p in pparams],
                        plot_kwargs=dict(color='k'),
                        hist_kwargs=dict(color='k'))
            fig.savefig(os.path.join(path, "potential_posterior.png"))

            # now make trace plots
            fig,axes = plt.subplots(Npp,1,figsize=(8,12),sharex=True)
            for ii in range(Npp):
                ax = axes[ii]
                p = pparams[ii]
                for jj in range(chain.shape[0]):
                    ax.plot(chain[jj,:,ii], drawstyle="steps", color='k', alpha=0.1)

                ax.axhline(p.target._truth, linewidth=4., alpha=0.5,
                           linestyle="--", color="#2B8CBE")
                ax.set_ylim(p._ln_prior.a,p._ln_prior.b)
                ax.set_ylabel(p.target.latex)
            fig.savefig(os.path.join(path, "potential_trace.png"))
            del fig
        stop = Npp

        # if particles are in the model / have been observed
        if isinstance(particles, ObservedParticle):
            true_particles = d["true_particles"]
            for ii in range(particles.nparticles):
                tub = flatchain[:,Npp+ii]

                start = Npp + particles.nparticles + 6*ii
                stop = start + 6
                _X = flatchain[:,start:stop]

                p = Particle(_X.T, units=particles._internal_units,
                             frame=heliocentric)
                p = p.to_units(particles._repr_units)

                prior_p = Particle(p0[:,start:stop].T,
                                   units=particles._internal_units,
                                   frame=heliocentric)
                prior_p = prior_p.to_units(particles._repr_units)

                # plot the posterior
                truths = [true_particles.tub[ii]] + true_particles._repr_X[ii].tolist()
                X_extents = [(truth-0.2*abs(truth),truth+0.2*abs(truth)) for truth in truths[1:]]
                truth_extents = [(t2,t1)] + X_extents

                this_p0 = np.hstack((p0[:,Npp+ii][:,np.newaxis], prior_p._repr_X))
                prior_extents = zip(np.min(this_p0, axis=0), np.max(this_p0, axis=0))

                extents = []
                for jj in range(7):
                    lo = min(truth_extents[jj][0], prior_extents[jj][0])
                    hi = max(truth_extents[jj][1], prior_extents[jj][1])
                    extents.append((lo,hi))

                labels = ['$t_{ub}$ [Myr]','l [deg]','b [deg]','D [kpc]',\
                          r'$\mu_l$ [mas/yr]', r'$\mu_l$ [mas/yr]','$v_r$ [km/s]']

                logger.debug("Plotting particle {} prior".format(ii))
                fig = triangle.corner(this_p0,
                                      labels=labels,
                                      truths=truths,
                                      extents=extents)
                fig.suptitle("Particle {0}".format(ii))
                fig.savefig(os.path.join(path, "particle{0}_prior.png"\
                                         .format(ii)))
                del fig

                fig = triangle.corner(np.hstack((tub[:,np.newaxis], p._repr_X)),
                                      labels=labels,
                                      truths=truths,
                                      extents=extents)
                fig.suptitle("Particle {0}".format(ii))
                fig.savefig(os.path.join(path, "particle{0}_posterior.png"\
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
                            q = (chain[jj,:,start+kk-1]*particles._internal_units[kk-1])
                            axes[kk].plot(q.to(particles._repr_units[kk-1]).value,
                                          drawstyle="steps", color='k',
                                          alpha=0.1)

                    axes[kk].axhline(truths[kk], linewidth=4., alpha=0.5,
                               linestyle="--", color="#2B8CBE")
                    axes[kk].set_ylim(extents[kk])

                fig.suptitle("Particle {}".format(ii))
                fig.savefig(os.path.join(path, "particle{}_trace.png".format(ii)))
                del fig
                gc.collect()

        # if satellite position is in the model / have been observed
        if isinstance(satellite, ObservedParticle):
            true_satellite = d["true_satellite"]

            start = stop
            stop = start + 6
            _X = flatchain[:,start:stop]

            s = Particle(_X.T, units=satellite._internal_units,
                         frame=heliocentric)
            s = s.to_units(satellite._repr_units)

            prior_s = Particle(p0[:,start:stop].T,
                               units=satellite._internal_units,
                               frame=heliocentric)
            prior_s = prior_s.to_units(satellite._repr_units)

            # plot the posterior
            truths = np.squeeze(true_satellite._repr_X).tolist()
            truth_extents = [(truth-0.2*abs(truth),truth+0.2*abs(truth)) for truth in truths]
            prior_extents = zip(np.min(prior_s._repr_X, axis=0), np.max(prior_s._repr_X, axis=0))

            extents = []
            for jj in range(6):
                lo = min(truth_extents[jj][0], prior_extents[jj][0])
                hi = max(truth_extents[jj][1], prior_extents[jj][1])
                extents.append((lo,hi))

            labels = ['l [deg]','b [deg]','D [kpc]',
                      r'$\mu_l$ [mas/yr]', r'$\mu_l$ [mas/yr]','$v_r$ [km/s]']

            logger.debug("Plotting satellite prior")
            fig = triangle.corner(prior_s._repr_X,
                                  labels=labels,
                                  truths=truths,
                                  extents=extents)
            fig.suptitle("Satellite")
            fig.savefig(os.path.join(path, "satellite_prior."))
            del fig

            logger.debug("Plotting satellite posterior")
            fig = triangle.corner(s._repr_X,
                                  labels=labels,
                                  truths=truths,
                                  extents=extents)
            fig.suptitle("Particle {0}".format(ii))
            fig.savefig(os.path.join(path, "satellite_posterior.png"))
            del fig

            # now make trace plots
            fig,axes = plt.subplots(6,1,figsize=(10,14))
            for kk in range(6):
                for jj in range(chain.shape[0]):
                    q = (chain[jj,:,start+kk]*satellite._internal_units[kk])
                    axes[kk].plot(q.to(satellite._repr_units[kk]).value,
                                  drawstyle="steps", color='k',
                                  alpha=0.1)

                axes[kk].axhline(truths[kk], linewidth=4., alpha=0.5,
                           linestyle="--", color="#2B8CBE")
                axes[kk].set_ylim(extents[kk])

            fig.suptitle("Satellite")
            fig.savefig(os.path.join(path, "satellite_trace.png"))
            del fig

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
        pool.close() if hasattr(pool, 'close') else None
        raise
        sys.exit(1)

    sys.exit(0)
