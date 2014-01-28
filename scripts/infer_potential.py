# coding: utf-8

""" Script for using the Rewinder to infer the Galactic host potential """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys
import logging
import shutil
import time

# Third-party
import astropy.units as u
from emcee.utils import sample_ball
import h5py
import matplotlib.pyplot as plt
import numpy as np
import triangle
import yaml

# Project
from streams import usys
from streams.coordinates.frame import galactocentric
from streams.dynamics import ObservedParticle, Particle
import streams.io as io
from streams.io.sgr import SgrSimulation
import streams.inference as si
import streams.potential as sp
from streams.util import get_pool, _parse_quantity, OrderedDictYAMLLoader

global pool
pool = None

# Create logger
logger = logging.getLogger(__name__)

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

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    logger.debug("Writing data to:\n\t{}".format(output_path))
    output_file = os.path.join(output_path, "inference.hdf5")

    # load satellite and particle data
    data_file = os.path.join(data_path, config['data_file'])
    logger.debug("Reading particle/satellite data from:\n\t{}".format(data_file))
    d = io.read_hdf5(data_file,
                     nparticles=config.get('nparticles', None),
                     particle_idx=config.get('particle_idx', None))

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
    model = si.StreamModel(potential, lnpargs=[t1,t2,dt,1.], # 1. is the inverse temperature
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

    ################################################
    ################################################
    # TEST
    # test_path = os.path.join(output_path, "test")
    # if not os.path.exists(test_path):
    #     os.mkdir(test_path)

    # for ii,truth in enumerate(model.truths):
    #     p = model.truths.copy()
    #     vals = truth*np.linspace(0.7, 1.3, 101)
    #     Ls = []
    #     for val in vals:
    #         p[ii] = val
    #         Ls.append(model(p))

    #     plt.clf()
    #     plt.plot(vals, Ls)
    #     plt.axvline(truth)
    #     plt.savefig(os.path.join(test_path, "test_{}.png".format(ii)))
    # return
    ################################################
    ################################################

    if os.path.exists(output_file) and overwrite:
        logger.info("Writing over output file '{}'".format(output_file))
        os.remove(output_file)

    # emcee parameters
    # read in the number of walkers to use
    nwalkers = config["walkers"]
    nsteps = config["steps"]
    nsteps_final = config.get("steps_final", 0)
    nburn = config.get("burn_in", 0)
    niter = config.get("iterate", 1)

    if not os.path.exists(output_file):
        logger.info("Output file '{}' doesn't exist, running inference...".format(output_file))

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

        # TODO: add annealing schedule to class
        #   - specify start temp, end temp, have it continuously change each step

        logger.info("Running {} walkers for {} iterations of {} steps..."\
                    .format(nwalkers, niter, nsteps//niter))

        time0 = time.time()
        for ii in range(niter):
            pos, prob, state = sampler.run_mcmc(pos, nsteps//niter)

            # if any of the samplers have less than 5% acceptance,
            #  start them from new positions sampled from the best position
            acc_frac_test = sampler.acceptance_fraction < 0.05
            if np.any(acc_frac_test):
                nbad = np.sum(acc_frac_test)
                med_pos = np.median(sampler.flatchain, axis=0)
                std = np.std(sampler.flatchain, axis=0)
                new_pos = sample_ball(med_pos, std, size=nwalkers)

                for jj in range(nwalkers):
                    if acc_frac_test[jj]:
                        pos[jj] = new_pos[jj]


        if nsteps_final > 0:
            sampler.reset()
            pos, prob, state = sampler.run_mcmc(pos, nsteps_final)

        t = time.time() - time0
        logger.debug("Spent {} seconds on sampling...".format(t))

        # write the sampler data to numpy save files
        logger.info("Writing sampler data to '{}'...".format(output_file))
        with h5py.File(output_file, "w") as f:
            f["chain"] = sampler.chain
            f["flatchain"] = sampler.flatchain
            f["lnprobability"] = sampler.lnprobability
            f["p0"] = p0
            f["acceptance_fraction"] = sampler.acceptance_fraction
            try:
                f["acor"] = sampler.acor
            except:
                logger.warn("Failed to compute autocorrelation time.")
                f["acor"] = []
    else:
        logger.info("Output file '{}' already exists, not running sampler...".format(output_file))

    pool.close() if hasattr(pool, 'close') else None

    #############################################################
    # Plotting
    #
    plot_config = config.get("plot", dict())
    plot_ext = plot_config.get("ext", "png")

    with h5py.File(output_file, "r") as f:
        chain = f["chain"].value
        flatchain = f["flatchain"].value
        acceptance_fraction = f["acceptance_fraction"].value
        p0 = f["p0"].value
        acor = f["acor"].value

    # thin chain
    if len(acor) > 0:
        t_med = np.median(acor)
        thin_chain = chain[:,::int(t_med)]
        thin_flatchain = np.vstack(thin_chain)
        logger.info("Median autocorrelation time: {}".format(t_med))
    else:
        logger.warn("FAILED TO THIN CHAIN")
        thin_chain = chain
        thin_flatchain = flatchain

    # plot true_particles, true_satellite over the rest of the stream
    gc_particles = true_particles.to_frame(galactocentric)
    sgr = SgrSimulation("2.5e8")
    all_gc_particles = sgr.particles(N=1000, expr="tub!=0").to_frame(galactocentric)

    fig,axes = plt.subplots(1,2,figsize=(16,8))
    axes[0].plot(all_gc_particles["x"].value, all_gc_particles["z"].value,
                 markersize=10., marker='.', linestyle='none', alpha=0.25)
    axes[0].plot(gc_particles["x"].value, gc_particles["z"].value,
                 markersize=10., marker='o', linestyle='none', alpha=0.75)
    axes[1].plot(all_gc_particles["vx"].to(u.km/u.s).value,
                 all_gc_particles["vz"].to(u.km/u.s).value,
                 markersize=10., marker='.', linestyle='none', alpha=0.25)
    axes[1].plot(gc_particles["vx"].to(u.km/u.s).value,
                 gc_particles["vz"].to(u.km/u.s).value,
                 markersize=10., marker='o', linestyle='none', alpha=0.75)
    fig.savefig(os.path.join(output_path, "xyz_vxvyvz.{}".format(plot_ext)))

    if plot_config.get("mcmc_diagnostics", False):
        logger.debug("Plotting MCMC diagnostics...")

        diagnostics_path = os.path.join(output_path, "diagnostics")
        if not os.path.exists(diagnostics_path):
            os.mkdir(diagnostics_path)

        # plot histogram of autocorrelation times
        if len(acor) > 0:
            fig,ax = plt.subplots(1,1,figsize=(8,8))
            ax.hist(acor, bins=12) #model.nparameters//5)
            ax.set_xlabel("Autocorrelation time")
            fig.suptitle("Histogram of autocorrelation times")
            fig.savefig(os.path.join(diagnostics_path, "acor.{}".format(plot_ext)))

        # plot histogram of acceptance fractions
        fig,ax = plt.subplots(1,1,figsize=(8,8))
        ax.hist(acceptance_fraction, bins=nwalkers//5)
        ax.set_xlabel("Acceptance fraction")
        fig.suptitle("Histogram of acceptance fractions for all walkers")
        fig.savefig(os.path.join(diagnostics_path, "acc_frac.{}".format(plot_ext)))

        # plot individual walkers
        # plt.figure(figsize=(12,6))
        # for k in range(model.nparameters):
        #     plt.clf()
        #     for ii in range(nwalkers):
        #         plt.plot(chain[ii,:,k], alpha=0.4, drawstyle='steps', color='k')

        #     plt.axhline(model.truths[k], color='r', lw=2., linestyle='-', alpha=0.5)
        #     plt.savefig(os.path.join(diagnostics_path, "param_{}.{}".format(k, plot_ext)))

        plt.close('all')

    if plot_config.get("posterior", False):
        logger.debug("Plotting posterior distributions...")

        flatchain_dict = model.label_flatchain(thin_flatchain)
        p0 = model.sample_priors(size=10000) # HACK HACK HACK
        p0_dict = model.label_flatchain(np.vstack(p0))
        potential_group = model.parameters.get('potential', None)
        particles_group = model.parameters.get('particles', None)
        satellite_group = model.parameters.get('satellite', None)

        if potential_group:
            this_flatchain = np.zeros((len(thin_flatchain),len(potential_group)))
            this_p0 = np.zeros((len(p0),len(potential_group)))
            for ii,param_name in enumerate(potential_group.keys()):
                this_flatchain[:,ii] = np.squeeze(flatchain_dict['potential'][param_name])
                this_p0[:,ii] = np.squeeze(p0_dict['potential'][param_name])

            fig = triangle.corner(this_p0,
                        plot_kwargs=dict(color='g',alpha=0.1),
                        hist_kwargs=dict(color='g',alpha=0.75,normed=True),
                        plot_contours=False)
            # fig = triangle.corner(this_flatchain,
            #             fig=fig,
            #             truths=[p.truth for p in model.parameters['potential'].values()],
            #             truth_color="r",
            #             plot_kwargs=dict(color='k',alpha=0.0001),
            #             hist_kwargs=dict(color='k',alpha=0.0001,normed=True))
            fig = triangle.corner(this_flatchain,
                        fig=fig,
                        truths=[p.truth for p in model.parameters['potential'].values()],
                        labels=potential_group.keys(),
                        plot_kwargs=dict(color='k',alpha=1.),
                        hist_kwargs=dict(color='k',alpha=0.75,normed=True))
            fig.savefig(os.path.join(output_path, "potential.{}".format(plot_ext)))

        if particles_group:
            for jj in range(nparticles):
                this_flatchain = None
                this_p0 = None
                this_truths = []
                for ii,pname in enumerate(particles_group.keys()):
                    p = flatchain_dict['particles'][pname][:,jj]
                    _p0 = p0_dict['particles'][pname][:,jj]
                    if p.ndim == 1:
                        p = p[:,np.newaxis]
                        _p0 = _p0[:,np.newaxis]

                    if this_flatchain is None:
                        this_flatchain = p
                        this_p0 = _p0
                    else:
                        this_flatchain = np.hstack((this_flatchain, p))
                        this_p0 = np.hstack((this_p0, _p0))

                    truth = model.parameters['particles'][pname].truth[jj]
                    this_truths += list(np.atleast_1d(truth))

                fig = triangle.corner(this_p0,
                            plot_kwargs=dict(color='g',alpha=1.),
                            hist_kwargs=dict(color='g',alpha=0.75,normed=True),
                            plot_contours=False)

                # labels=potential_group.keys(),
                fig = triangle.corner(this_flatchain,
                            fig=fig,
                            truths=this_truths,
                            plot_kwargs=dict(color='k',alpha=1.),
                            hist_kwargs=dict(color='k',alpha=0.75,normed=True))
                fig.savefig(os.path.join(output_path, "particle{}.{}".format(jj,plot_ext)))

            # HACK this is the Hogg suck-it-up plot
            # fig = triangle.corner(thin_flatchain,
            #                       truths=model.truths,
            #                       plot_kwargs=dict(color='k',alpha=1.),
            #                       hist_kwargs=dict(color='k',alpha=0.75,normed=True))
            # fig.savefig(os.path.join(output_path, "suck-it-up.{}".format(plot_ext)))

        # TODO:
        if satellite_group:
            raise NotImplementedError()

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

    pool.close() if hasattr(pool, 'close') else None
    sys.exit(0)
