# coding: utf-8

""" Script for using the Rewinder to infer the Galactic host potential """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys
import glob
import logging
import random
import shutil
import time

# Third-party
import astropy.units as u
from emcee.utils import sample_ball
from emcee import autocorr
import h5py
import matplotlib.pyplot as plt
import numpy as np
import triangle

# Project
from streams.coordinates.frame import galactocentric
import streams.io as io
from streams.io.sgr import SgrSimulation
import streams.inference as si
import streams.potential as sp
from streams.util import get_pool, _label_map, _unit_transform

global pool
pool = None

# Create logger
logger = logging.getLogger(__name__)

def fix_whack_walkers(pos, acc_frac, flatlnprob, flatchain, threshold=None):
    if threshold is None:
        threshold = 0.02

    if np.any(acc_frac < threshold):
        ix = acc_frac < threshold

        # resample positions for bad walkers
        best_pos = flatchain[flatlnprob.argmax()]

        # compute walker variance using median absolute deviation
        std = np.median(np.absolute(flatchain - np.median(flatchain, axis=0)[np.newaxis]), axis=0)
        std /= 10.
        pos[ix] = np.random.normal(best_pos, std,
                                   size=(sum(ix),pos.shape[1]))

    return pos

def main(config_file, mpi=False, threads=None, overwrite=False, continue_sampler=False):
    """ TODO: """

    # get a pool object given the configuration parameters
    # -- This needs to go here so I don't read in the particle file for each thread. --
    pool = get_pool(mpi=mpi, threads=threads)

    # read configuration from a YAML file
    config = io.read_config(config_file)
    np.random.seed(config["seed"])
    random.seed(config["seed"])

    if not os.path.exists(config['streams_path']):
        raise IOError("Specified streams path '{}' doesn't exist!".format(config['streams_path']))
    logger.debug("Path to streams project: {}".format(config['streams_path']))

    # the path to write things to
    output_path = config["output_path"]
    logger.debug("Will write data to:\n\t{}".format(output_path))
    cache_output_path = os.path.join(output_path, "cache")

    # get a StreamModel from a config dict
    model = si.StreamModel.from_config(config)
    logger.info("Model has {} parameters".format(model.nparameters))

    if os.path.exists(cache_output_path) and overwrite:
        logger.info("Writing over output path '{}'".format(cache_output_path))
        logger.debug("Deleting files: '{}'".format(os.listdir(cache_output_path)))
        shutil.rmtree(cache_output_path)

    # emcee parameters
    # read in the number of walkers to use
    nwalkers = config["walkers"]
    nsteps = config["steps"]
    output_every = config.get("output_every", None)
    nburn = config.get("burn_in", 0)
    start_truth = config.get("start_truth", False)
    a = config.get("a", 2.) # emcee tuning param

    if not os.path.exists(cache_output_path) and not continue_sampler:
        logger.info("Output path '{}' doesn't exist, running inference..."\
                    .format(cache_output_path))
        os.mkdir(cache_output_path)

        # sample starting positions
        p0 = model.sample_priors(size=nwalkers,
                                 start_truth=start_truth)
        logger.debug("Priors sampled...")

        if nburn > 0:
            sampler = si.StreamModelSampler(model, nwalkers, pool=pool, a=a)

            time0 = time.time()
            logger.info("Burning in sampler for {} steps...".format(nburn))
            pos, xx, yy = sampler.run_mcmc(p0, nburn)

            pos = fix_whack_walkers(pos, sampler.acceptance_fraction,
                                    sampler.flatlnprobability, sampler.flatchain,
                                    threshold=config.get("acceptance_threshold", None))

            t = time.time() - time0
            logger.debug("Spent {} seconds on burn-in...".format(t))

        else:
            pos = p0

        if nsteps > 0:
            sampler = si.StreamModelSampler(model, nwalkers, pool=pool, a=a)
            sampler.run_inference(pos, nsteps, path=cache_output_path, first_step=0,
                                  output_every=output_every,
                                  output_file_fmt="inference_{:06d}.hdf5")

    elif os.path.exists(cache_output_path) and not continue_sampler:
        logger.info("Output path '{}' already exists, not running sampler..."\
                    .format(cache_output_path))

    elif os.path.exists(cache_output_path) and continue_sampler:
        if len(os.listdir(cache_output_path)) == 0:
            logger.error("No files in path: {}".format(cache_output_path))
            sys.exit(1)

        continue_files = glob.glob(os.path.join(cache_output_path, "inference_*.hdf5"))
        continue_file = config.get("continue_file", sorted(continue_files)[-1])
        continue_file = os.path.join(cache_output_path, continue_file)
        if not os.path.exists(continue_file):
            logger.error("File {} doesn't exist!".format(continue_file))
            sys.exit(1)

        with h5py.File(continue_file, "r") as f:
            old_chain = f["chain"].value
            old_flatchain = np.vstack(old_chain)
            old_lnprobability = f["lnprobability"].value
            old_flatlnprobability = np.vstack(old_lnprobability)
            old_acc_frac = f["acceptance_fraction"].value
            last_step = f["last_step"].value

        pos = old_chain[:,-1]
        pos = fix_whack_walkers(pos, old_acc_frac,
                                old_flatlnprobability,
                                old_flatchain,
                                threshold=config.get("acceptance_threshold", None))

        sampler = si.StreamModelSampler(model, nwalkers, pool=pool, a=a)
        logger.info("Continuing sampler...running {} walkers for {} steps..."\
                .format(nwalkers, nsteps))
        sampler.run_inference(pos, nsteps, path=cache_output_path, first_step=last_step,
                              output_every=output_every,
                              output_file_fmt = "inference_{:07d}.hdf5")

    else:
        print("Unknown state.")
        sys.exit(1)

    pool.close() if hasattr(pool, 'close') else None

    #############################################################
    # Plotting
    #
    plot_config = config.get("plot", dict())
    plot_ext = plot_config.get("ext", "png")

    # glob properly orders the list
    for filename in sorted(glob.glob(os.path.join(cache_output_path,"inference_*.hdf5"))):
        logger.debug("Reading file {}...".format(filename))
        with h5py.File(filename, "r") as f:
            try:
                chain = np.hstack((chain,f["chain"].value))
            except NameError:
                chain = f["chain"].value

            acceptance_fraction = f["acceptance_fraction"].value

    try:
        acor = autocorr.integrated_time(np.mean(chain, axis=0), axis=0,
                                        window=50) # 50 comes from emcee
    except:
        acor = []

    flatchain = np.vstack(chain)

    # thin chain
    if config.get("thin_chain", True):
        if len(acor) > 0:
            t_med = np.median(acor)
            thin_chain = chain[:,::int(t_med)]
            thin_flatchain = np.vstack(thin_chain)
            logger.info("Median autocorrelation time: {}".format(t_med))
        else:
            logger.warn("FAILED TO THIN CHAIN")
            thin_chain = chain
            thin_flatchain = flatchain
    else:
        thin_chain = chain
        thin_flatchain = flatchain

    # plot true_particles, true_satellite over the rest of the stream
    gc_particles = model.true_particles.to_frame(galactocentric)
    m = model.true_satellite.mass
    # HACK
    sgr = SgrSimulation("sgr_nfw/M2.5e+0{}".format(int(np.floor(np.log10(m)))), "SNAP113")
    all_gc_particles = sgr.particles(n=1000, expr="tub!=0").to_frame(galactocentric)

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
            fig,ax = plt.subplots(1,1,figsize=(12,6))
            ax.plot(acor, marker='o', linestyle='none') #model.nparameters//5)
            ax.set_xlabel("Parameter index")
            ax.set_ylabel("Autocorrelation time")
            fig.savefig(os.path.join(diagnostics_path, "acor.{}".format(plot_ext)))

        # plot histogram of acceptance fractions
        fig,ax = plt.subplots(1,1,figsize=(8,8))
        ax.hist(acceptance_fraction, bins=nwalkers//5)
        ax.set_xlabel("Acceptance fraction")
        fig.suptitle("Histogram of acceptance fractions for all walkers")
        fig.savefig(os.path.join(diagnostics_path, "acc_frac.{}".format(plot_ext)))

        # plot individual walkers
        plt.figure(figsize=(12,6))
        for k in range(model.nparameters):
            plt.clf()
            for ii in range(nwalkers):
                plt.plot(chain[ii,:,k], alpha=0.4, drawstyle='steps', color='k')

            plt.axhline(model.truths[k], color='r', lw=2., linestyle='-', alpha=0.5)
            plt.savefig(os.path.join(diagnostics_path, "param_{}.{}".format(k, plot_ext)))

        plt.close('all')

    if plot_config.get("posterior", False):
        logger.debug("Plotting posterior distributions...")

        flatchain_dict = model.label_flatchain(thin_flatchain)
        p0 = model.sample_priors(size=1000) # HACK HACK HACK
        p0_dict = model.label_flatchain(np.vstack(p0))
        potential_group = model.parameters.get('potential', None)
        particles_group = model.parameters.get('particles', None)
        satellite_group = model.parameters.get('satellite', None)
        flatchains = dict()

        if potential_group:
            this_flatchain = np.zeros((len(thin_flatchain),len(potential_group)))
            this_p0 = np.zeros((len(p0),len(potential_group)))
            this_truths = []
            this_extents = []
            for ii,pname in enumerate(potential_group.keys()):
                f = _unit_transform[pname]
                p = model.parameters['potential'][pname]

                this_flatchain[:,ii] = f(np.squeeze(flatchain_dict['potential'][pname]))
                this_p0[:,ii] = f(np.squeeze(p0_dict['potential'][pname]))
                this_truths.append(f(p.truth))
                this_extents.append((f(p._prior.a), f(p._prior.b)))

                print(pname, np.median(this_flatchain[:,ii]), np.std(this_flatchain[:,ii]))

            fig = triangle.corner(this_p0,
                        point_kwargs=dict(color='#2b8cbe',alpha=0.1),
                        hist_kwargs=dict(color='#2b8cbe',alpha=0.75,normed=True,bins=50),
                        plot_contours=False)

            fig = triangle.corner(this_flatchain,
                        fig=fig,
                        truths=this_truths,
                        labels=[_label_map[k] for k in potential_group.keys()],
                        extents=this_extents,
                        point_kwargs=dict(color='k',alpha=1.),
                        hist_kwargs=dict(color='k',alpha=0.75,normed=True,bins=50))
            fig.savefig(os.path.join(output_path, "potential.{}".format(plot_ext)))

            flatchains['potential'] = this_flatchain

        nparticles = model.true_particles.nparticles
        if particles_group and len(particles_group) > 1:
            for jj in range(nparticles):
                this_flatchain = np.zeros((len(thin_flatchain),len(particles_group)))
                this_p0 = np.zeros((len(p0),len(particles_group)))
                this_truths = []
                this_extents = None
                for ii,pname in enumerate(particles_group.keys()):
                    f = _unit_transform[pname]
                    p = model.parameters['particles'][pname]

                    this_flatchain[:,ii] = f(np.squeeze(flatchain_dict['particles'][pname][:,jj]))
                    this_p0[:,ii] = f(np.squeeze(p0_dict['particles'][pname][:,jj]))
                    this_truths.append(f(p.truth[jj]))
                    #this_extents.append((f(p._prior.a), f(p._prior.b)))

                fig = triangle.corner(this_p0,
                            point_kwargs=dict(color='#2b8cbe',alpha=0.1),
                            hist_kwargs=dict(color='#2b8cbe',alpha=0.75,normed=True,bins=50),
                            plot_contours=False)

                fig = triangle.corner(this_flatchain,
                            fig=fig,
                            truths=this_truths,
                            labels=[_label_map[k] for k in particles_group.keys()],
                            extents=this_extents,
                            point_kwargs=dict(color='k',alpha=1.),
                            hist_kwargs=dict(color='k',alpha=0.75,normed=True,bins=50))
                fig.savefig(os.path.join(output_path, "particle{}.{}".format(jj,plot_ext)))

        # plot the posterior for the satellite parameters
        if satellite_group and len(satellite_group) > 1:
            jj = 0
            this_flatchain = np.zeros((len(thin_flatchain),len(satellite_group)))
            this_p0 = np.zeros((len(p0),len(satellite_group)))
            this_truths = []
            this_extents = None
            for ii,pname in enumerate(satellite_group.keys()):
                f = _unit_transform[pname]
                p = model.parameters['satellite'][pname]

                this_flatchain[:,ii] = f(np.squeeze(flatchain_dict['satellite'][pname][:,jj]))
                this_p0[:,ii] = f(np.squeeze(p0_dict['satellite'][pname][:,jj]))
                try:
                    this_truths.append(f(p.truth[jj]))
                except: # IndexError:
                    this_truths.append(f(p.truth))
                #this_extents.append((f(p._prior.a), f(p._prior.b)))

            fig = triangle.corner(this_p0,
                        point_kwargs=dict(color='#2b8cbe',alpha=0.1),
                        hist_kwargs=dict(color='#2b8cbe',alpha=0.75,normed=True,bins=50),
                        plot_contours=False)

            fig = triangle.corner(this_flatchain,
                        fig=fig,
                        truths=this_truths,
                        labels=[_label_map[k] for k in satellite_group.keys()],
                        extents=this_extents,
                        point_kwargs=dict(color='k',alpha=1.),
                        hist_kwargs=dict(color='k',alpha=0.75,normed=True,bins=50))
            fig.savefig(os.path.join(output_path, "satellite.{}".format(plot_ext)))

            flatchains['satellite'] = this_flatchain

        if flatchains.has_key('potential') and flatchains.has_key('satellite'):
            this_flatchain = np.hstack((flatchains['potential'],flatchains['satellite']))
            labels = [_label_map[k] for k in potential_group.keys()+satellite_group.keys()]
            fig = triangle.corner(this_flatchain,
                        labels=labels,
                        point_kwargs=dict(color='k',alpha=1.),
                        hist_kwargs=dict(color='k',alpha=0.75,normed=True,bins=50))
            fig.savefig(os.path.join(output_path, "suck-it-up.{}".format(plot_ext)))


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
    parser.add_argument("--continue", dest="continue_sampler", default=False, action="store_true",
                        help="Continue the sampler for nsteps from endpoint of previous run.")

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
             overwrite=args.overwrite,
             continue_sampler=args.continue_sampler)
    except:
        pool.close() if hasattr(pool, 'close') else None
        raise
        sys.exit(1)

    pool.close() if hasattr(pool, 'close') else None
    sys.exit(0)
