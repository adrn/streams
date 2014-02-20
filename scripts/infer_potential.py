# coding: utf-8

""" Script for using the Rewinder to infer the Galactic host potential """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
from collections import defaultdict
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

# Project
from streams.coordinates.frame import galactocentric
import streams.io as io
from streams.io.sgr import SgrSimulation
import streams.inference as si
import streams.potential as sp
from streams.util import get_pool

global pool
pool = None

# Create logger
logger = logging.getLogger(__name__)

def main(config_file, mpi=False, threads=None, overwrite=False):
    """ TODO: """

    # get a pool object given the configuration parameters
    # -- This needs to go here so I don't read in the particle file for each thread. --
    pool = get_pool(mpi=mpi, threads=threads)

    # read configuration from a YAML file
    config = io.read_config(config_file)

    if not os.path.exists(config['streams_path']):
        raise IOError("Specified streams path '{}' doesn't exist!".format(config['streams_path']))
    logger.debug("Path to streams project: {}".format(config['streams_path']))

    # the path to write things to
    output_path = config["output_path"]
    logger.debug("Will write data to:\n\t{}".format(output_path))
    output_file = os.path.join(output_path, "inference.hdf5")

    # get a StreamModel from a config dict
    model = si.StreamModel.from_config(config)
    logger.info("Model has {} parameters".format(model.nparameters))

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
        logger.debug("Priors sampled...")

        # get the sampler
        sampler = si.StreamModelSampler(model, nwalkers, pool=pool)

        if nburn > 0:
            time0 = time.time()
            logger.info("Burning in sampler for {} steps...".format(nburn))

            pos, xx, yy = sampler.run_mcmc(p0, nburn)

            best_idx = sampler.flatlnprobability.argmax()
            best_pos = sampler.flatchain[best_idx]

            std = np.std(p0, axis=0) / 5.
            pos = np.array([np.random.normal(best_pos, std) \
                            for kk in range(nwalkers)])

            sampler.reset()
            t = time.time() - time0
            logger.debug("Spent {} seconds on burn-in...".format(t))

        else:
            pos = p0

        logger.info("Running {} walkers for {} iterations of {} steps..."\
                    .format(nwalkers, niter, nsteps//niter))

        if nsteps > 0:
            time0 = time.time()
            for ii in range(niter):
                logger.debug("Iteration: {}".format(ii))
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

            t = time.time() - time0
            logger.debug("Spent {} seconds on main sampling...".format(t))

        if nsteps_final > 0:
            time0 = time.time()
            sampler.reset()
            pos, prob, state = sampler.run_mcmc(pos, nsteps_final)
            t = time.time() - time0
            logger.debug("Spent {} seconds on final sampling...".format(t))

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
        try:
            acor = f["acor"].value
        except:
            acor = []

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
    m = "{:.1e}".format(model.true_satellite.mass).replace("0","").replace("+","")
    sgr = SgrSimulation(m)
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
        _label_map = dict(q1="$q_1$", q2="$q_2$", qz="$q_z$",
                          phi=r"$\phi$ [deg]", v_halo=r"$v_h$ [km/s]", R_halo=r"$R_h$ [kpc]",
                          l="$l$ [deg]", b="$b$ [deg]", d="$d$ [kpc]",
                          mul=r"$\mu_l$ [mas/yr]", mub=r"$\mu_b$ [mas/yr]", vr=r"$v_r$ [km/s]")

        _unit_transform = defaultdict(lambda: lambda x: x)
        _unit_transform['phi'] = lambda x: (x*u.rad).to(u.degree).value
        _unit_transform["v_halo"] = lambda x: (x*u.kpc/u.Myr).to(u.km/u.s).value
        _unit_transform["R_halo"] = lambda x: (x*u.kpc).to(u.kpc).value
        _unit_transform["l"] = lambda x: (x*u.rad).to(u.degree).value
        _unit_transform["b"] = lambda x: (x*u.rad).to(u.degree).value
        _unit_transform["d"] = lambda x: (x*u.kpc).to(u.kpc).value
        _unit_transform["mul"] = lambda x: (x*u.rad/u.Myr).to(u.mas/u.yr).value
        _unit_transform["mub"] = lambda x: (x*u.rad/u.Myr).to(u.mas/u.yr).value
        _unit_transform["vr"] = lambda x: (x*u.kpc/u.Myr).to(u.km/u.s).value

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
            this_truths = []
            this_extents = []
            for ii,pname in enumerate(potential_group.keys()):
                f = _unit_transform[pname]
                p = model.parameters['potential'][pname]

                this_flatchain[:,ii] = f(np.squeeze(flatchain_dict['potential'][pname]))
                this_p0[:,ii] = f(np.squeeze(p0_dict['potential'][pname]))
                this_truths.append(f(p.truth))
                this_extents.append((f(p._prior.a), f(p._prior.b)))

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

        nparticles = model.true_particles.nparticles
        if particles_group:
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
        if satellite_group:
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
                this_truths.append(f(p.truth[jj]))
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
