#!/vega/astro/users/amp2217/yt-x86_64/bin/python
# coding: utf-8

""" Script for using the Rewinder to infer the Galactic host potential """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys
import time
import gc
import copy
import logging
import multiprocessing
from collections import defaultdict

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
from streams.dynamics import Particle, Orbit, ObservedParticle
from streams.inference import *
import streams.io as io
from streams.observation.gaia import gaia_spitzer_errors
import streams.potential as sp
from streams.util import _parse_quantity, make_path, print_options

from streams.coordinates import _hel_to_gc, _gc_to_hel
from streams.integrate import LeapfrogIntegrator

global pool
pool = None

# Create logger
logger = logging.getLogger(__name__)

hel_units = [u.radian,u.radian,u.kpc,u.radian/u.Myr,u.radian/u.Myr,u.kpc/u.Myr]

def get_pool(mpi=False, threads=None):
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

def ln_likelihood(t1, t2, dt, potential, p_hel, s_hel, tub):

    p_gc = _hel_to_gc(p_hel)
    s_gc = _hel_to_gc(s_hel)

    gc = np.vstack((s_gc,p_gc)).copy()
    acc = np.zeros_like(gc[:,:3])
    integrator = LeapfrogIntegrator(potential._acceleration_at,
                                    np.array(gc[:,:3]), np.array(gc[:,3:]),
                                    args=(gc.shape[0], acc))

    times, rs, vs = integrator.run(t1=t1, t2=t2, dt=dt)

    s_orbit = np.vstack((rs[:,0][:,np.newaxis].T, vs[:,0][:,np.newaxis].T)).T
    p_orbits = np.vstack((rs[:,1:].T, vs[:,1:].T)).T

    # These are the unbinding time indices for each particle
    t_idx = np.array([np.argmin(np.fabs(times - t)) for t in tub])

    sat_var = np.zeros((len(times),6))
    sat_var[:,:3] = potential._tidal_radius(2.5e8, s_orbit[...,:3])*1.26
    sat_var[:,3:] += 0.017198632325
    cov = sat_var**2

    Sigma = np.array([cov[jj] for jj in t_idx])
    p_x = np.array([p_orbits[jj,ii] for ii,jj in enumerate(t_idx)])
    s_x = np.array([s_orbit[jj,0] for jj in t_idx])

    log_p_x_given_phi = -0.5*np.sum(2*np.log(Sigma) + (p_x-s_x)**2/Sigma, axis=1)

    return np.sum(log_p_x_given_phi)

def ln_data_likelihood(x, D, sigma):
    log_p_D_given_x = -0.5*np.sum(2.*np.log(sigma) + (x-D)**2/sigma**2, axis=-1)
    return np.sum(log_p_D_given_x)

def ln_posterior(p, *args):
    (t1, t2, dt, priors,
     s_hel, p_hel, s_hel_errors, p_hel_errors,
     potential_params, tub, nparticles) = args

    ll = 0. # ln likelihood
    lp = 0. # ln prior
    prior_ix = 0

    Npp = len(potential_params)

    ix1 = 0
    ix2 = Npp
    kwargs = dict(zip(potential_params, p[ix1:ix2]))
    potential = sp.LawMajewski2010(**kwargs)
    ix1 = ix2
    for ii in range(Npp):
        lp += priors[prior_ix](p[ii])
        prior_ix += 1

    if tub is None:
        ix2 = ix1 + nparticles
        tub = p[ix1:ix2]
        ix1 = ix2
        lp += priors[prior_ix](tub)
        prior_ix += 1

    if p_hel_errors is not None:
        ix2 = ix1 + nparticles*6
        walker_p_hel = p[ix1:ix2].reshape(nparticles,6)
        ix1 = ix2

        ll += ln_data_likelihood(walker_p_hel, p_hel, p_hel_errors)
    else:
        walker_p_hel = p_hel

    #     for ii in range(nparticles):
    #         ln_prior += priors[prior_ix](p_hel[ii])
    #         prior_ix += 1

    if s_hel_errors is not None:
        ix2 = ix1 + 6
        walker_s_hel = p[ix1:ix2]
        ix1 = ix2

        ll += ln_data_likelihood(walker_s_hel, s_hel, s_hel_errors)
    else:
        walker_s_hel = s_hel

    #     ln_prior += priors[prior_ix](s_hel)
    #     prior_ix += 1

    lp = np.sum(lp)
    if np.isinf(lp):
        return lp

    else:
        ll += ln_likelihood(t1, t2, dt, potential, walker_p_hel, walker_s_hel, tub)
        return lp + ll

def convert_units(X, u1, u2):
    """ Convert X from units u1 to units u2. """

    new_X = np.zeros_like(X)
    for ii in range(len(X.T)):
        new_X[...,ii] = (X[...,ii]*u1[ii]).to(u2[ii]).value
    return new_X

def main(mpi=False, threads=None, overwrite=False):
    """ TODO: """

    pool = get_pool(mpi=mpi, threads=threads)

    ##################################################
    # config
    # yeti
    home = "/vega/astro/users/amp2217/"
    # #home = "/hpc/astro/users/amp2217/"
    nburn = 7500
    nsteps = 5000
    nparticles = 32
    nwalkers = 512
    potential_params = ["q1","qz","v_halo","phi"]
    infer_tub_tf = True
    infer_particles_tf = True
    infer_satellite_tf = False
    name = "super_test32"
    plot_walkers = False
    test = False
    ##################################################

    ##################################################
    # # LAPTOP TESTING
    # home = "/Users/adrian/"
    # nburn = 0
    # nsteps = 10
    # nparticles = 4
    # nwalkers = 64
    # potential_params = ["q1","qz","v_halo","phi"]
    # infer_tub_tf = True
    # infer_particles_tf = True
    # infer_satellite_tf = False
    # name = "super_test"
    # plot_walkers = False
    # test = False
    ##################################################

    if infer_particles_tf:
        data_file = "N128_ptcl_errors.hdf5"
    else:
        data_file = "N128_no_errors.hdf5"

    path = os.path.join(home, "output_data", name)
    d_path = os.path.join(home, "projects/streams/data/observed_particles/")
    d = io.read_hdf5(os.path.join(d_path, data_file))
    output_file = os.path.join(path, "inference.hdf5")

    if not os.path.exists(path):
        os.makedirs(path)

    if os.path.exists(output_file) and overwrite:
        logger.info("Writing over output file '{}'".format(output_file))
        os.remove(output_file)

    truths = []
    potential = sp.LawMajewski2010()
    Npp = len(potential_params)

    observed_satellite_hel = d["satellite"]._X
    if infer_satellite_tf:
        satellite_hel_err = d["satellite"]._error_X
    else:
        satellite_hel_err = None
    logger.debug("Read in satellite".format(observed_satellite_hel))

    observed_particles_hel = d["particles"]._X[:nparticles]
    assert len(np.ravel(observed_particles_hel)) == len(np.unique(np.ravel(observed_particles_hel)))

    if infer_particles_tf:
        particles_hel_err = d["particles"]._error_X[:nparticles]
    else:
        particles_hel_err = None

    try:
        true_tub = d["true_particles"].tub[:nparticles]
    except KeyError:
        true_tub = d["particles"].tub[:nparticles]

    if infer_tub_tf:
        tub = None
    else:
        tub = np.array(true_tub)
    nparticles = observed_particles_hel.shape[0]
    logger.debug("Read in {} particles".format(nparticles))

    t1 = float(d["t1"])
    t2 = float(d["t2"])
    dt = -1.

    logger.debug("{} walkers".format(nwalkers))

    priors = []
    # potential
    for p_name in potential_params:
        pp = potential.parameters[p_name]
        priors.append(LogUniformPrior(*pp._range))
        truths.append(pp._truth)

    # tub
    if tub is None:
        priors.append(LogUniformPrior([t2]*nparticles, [t1]*nparticles))
        truths = truths + true_tub.tolist()

    # particles
    if infer_particles_tf:
        assert particles_hel_err is not None

        for ii in range(nparticles):
            prior = LogNormalPrior(observed_particles_hel[ii], sigma=particles_hel_err[ii])
            priors.append(prior)

        if d.has_key("true_particles"):
            truths = truths + np.ravel(d["true_particles"]._X).tolist()

    # satellite
    # TODO

    # ----------------------------------------------------------------------
    # TEST
    #
    if test:
        true_particles_hel = d["true_particles"]._X[:nparticles]

        print(ln_data_likelihood(x=observed_particles_hel,
                                 D=observed_particles_hel, sigma=particles_hel_err))

        print(ln_likelihood(t1, t2, dt, potential,
                            true_particles_hel, observed_satellite_hel,
                            true_tub))

        true_p = [potential.parameters[p_name]._truth for p_name in potential_params]
        true_p = np.append(true_p, true_tub)
        true_p = np.append(true_p, np.ravel(true_particles_hel))

        args = (t1, t2, dt, priors,
                observed_satellite_hel, true_particles_hel,
                satellite_hel_err, particles_hel_err,
                potential_params, None, nparticles)

        print(ln_posterior(true_p, *args))

        # test potential
        vals = np.linspace(0.9, 1.1, 25)
        for ii,p_name in enumerate(potential_params):
            pp = potential.parameters[p_name]
            Ls = []
            for val in vals:
                #potential = sp.LawMajewski2010(**{p_name:p._truth*val})
                # ll = ln_likelihood(t1, t2, dt, potential,
                #                    true_particles_hel, observed_satellite_hel,
                #                    true_tub)
                p = true_p.copy()
                p[ii] = pp._truth*val
                ll = ln_posterior(p, *args)
                Ls.append(ll)

            plt.clf()
            plt.plot(vals*pp._truth, np.exp(Ls))
            plt.axvline(pp._truth)
            plt.ylabel("Posterior")
            plt.savefig(os.path.join(path, "potential_{}.png".format(p_name)))

            plt.clf()
            plt.plot(vals*pp._truth, Ls)
            plt.axvline(pp._truth)
            plt.ylabel("Log Posterior")
            plt.savefig(os.path.join(path, "potential_{}_log.png".format(p_name)))

        # test tub
        tubs = np.linspace(0.,6200,25)
        for ii in range(nparticles):
            Ls = []
            for tub in tubs:
                # this_tub = true_tub.copy()
                # this_tub[ii] = tub
                # ll = ln_likelihood(t1, t2, dt, potential,
                #               true_particles_hel, observed_satellite_hel,
                #               this_tub)
                p = true_p.copy()
                p[Npp+ii] = tub
                ll = ln_posterior(p, *args)
                Ls.append(ll)

            plt.clf()
            plt.plot(tubs, np.exp(Ls))
            plt.axvline(true_tub[ii])
            plt.ylabel("Posterior")
            plt.savefig(os.path.join(path, "particle{}_tub.png".format(ii)))

            plt.clf()
            plt.plot(tubs, Ls)
            plt.axvline(true_tub[ii])
            plt.ylabel("Log Posterior")
            plt.savefig(os.path.join(path, "particle{}_tub_log.png".format(ii)))

        # particle positions
        coords = ["l","b","D","mul","mub","vr"]
        for ii in range(nparticles):
            for jj in range(6):
                vals = np.linspace(observed_particles_hel[ii,jj] - 3*particles_hel_err[ii,jj],
                                   observed_particles_hel[ii,jj] + 3*particles_hel_err[ii,jj],
                                   25)
                p_hel = true_particles_hel.copy()
                Ls = []
                for v in vals:
                    p = true_p.copy()
                    p[Npp+nparticles+ii*6+jj] = v
                    ll = ln_posterior(p, *args)
                    # p_hel[ii,jj] = v
                    # l = ln_likelihood(t1, t2, dt, potential,
                    #                 p_hel, observed_satellite_hel,
                    #                 true_tub)
                    Ls.append(ll)

                plt.clf()
                plt.plot(vals, np.exp(Ls))
                plt.axvline(true_particles_hel[ii,jj], label='true')
                plt.axvline(observed_particles_hel[ii,jj], color='r', label='observed')
                plt.legend()
                plt.ylabel("Posterior")
                plt.savefig(os.path.join(path, "particle{}_coord{}.png".format(ii,coords[jj])))

                plt.clf()
                plt.plot(vals, Ls)
                plt.axvline(true_particles_hel[ii,jj], label='true')
                plt.axvline(observed_particles_hel[ii,jj], color='r', label='observed')
                plt.legend()
                plt.ylabel("Log Posterior")
                plt.savefig(os.path.join(path, "particle{}_coord{}_log.png".format(ii,coords[jj])))

        return
    # ----------------------------------------------------------------------

    args = (t1, t2, dt, priors,
            observed_satellite_hel, observed_particles_hel,
            satellite_hel_err, particles_hel_err,
            potential_params, tub, nparticles)

    if not os.path.exists(output_file):
        logger.info("Output file '{}' doesn't exist".format(output_file))
        # get initial walker positions
        for jj in range(nwalkers):
            try:
                p0[jj] = np.concatenate([np.atleast_1d(p.sample()) for p in priors])
            except NameError:
                this_p0 = np.concatenate([np.atleast_1d(p.sample()) for p in priors])
                p0 = np.zeros((nwalkers,len(this_p0)))
                p0[jj] = this_p0

        ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###
        ### HACK HACK HACK
        ### HACK TO INITIALIZE WALKERS NEAR true tub!
        for ii in range(nparticles):
            jj = ii + len(potential_params)
            p0[:,jj] = np.random.normal(true_tub[ii], 100., size=nwalkers)

        #jj = nparticles + len(potential_params)
        #for ii in range(nwalkers):
        #    p0[ii,jj:] = np.random.normal(np.ravel(d["true_particles"]._X[:nparticles]),
        #                                  np.ravel(particles_hel_err)*0.1)

        ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###
        ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###

        logger.debug("p0 shape: {}".format(p0.shape))
        sampler = emcee.EnsembleSampler(nwalkers, p0.shape[-1], ln_posterior,
                                        pool=pool, args=args)

        if nburn > 0:
            logger.info("Burning in sampler for {} steps...".format(nburn))
            pos, xx, yy = sampler.run_mcmc(p0, nburn)
            sampler.reset()
        else:
            pos = p0

        logger.info("Running sampler for {} steps...".format(nsteps))
        a = time.time()
        pos, prob, state = sampler.run_mcmc(pos, nsteps)
        t = time.time() - a
        logger.debug("Spent {} seconds on sampler...".format(t))

        with h5py.File(output_file, "w") as f:
            f["chain"] = sampler.chain
            f["flatchain"] = sampler.flatchain
            f["lnprobability"] = sampler.lnprobability
            f["p0"] = p0
            f["acceptance_fraction"] = sampler.acceptance_fraction
    else:
        logger.info("Output file '{}' already exists - reading in data".format(output_file))

    with h5py.File(output_file, "r") as f:
        chain = f["chain"].value
        flatchain = f["flatchain"].value
        p0 = f["p0"].value
        acceptance_fraction = f["acceptance_fraction"].value

    logger.debug("Acceptance fractions: {}".format(acceptance_fraction))

    if pool is not None:
        pool.close()

    # Plotting!
    # =========
    if plot_walkers:
        logger.info("Plotting individual walkers...")
        for jj in range(p0.shape[-1]):
            logger.debug("\t plotting walker {}".format(jj))
            plt.clf()
            for ii in range(nwalkers):
                plt.plot(chain[ii,:,jj], alpha=0.4, drawstyle='steps')

            plt.axhline(truths[jj], color='k', lw=4., linestyle='--')
            plt.savefig(os.path.join(path, "walker_{}.png".format(jj)))

    # Make corner plots
    # -----------------

    # potential
    ix1 = 0
    ix2 = Npp

    if Npp > 0:
        logger.info("Plotting potential posterior...")
        d_units = [u.s,u.km,u.deg]
        potential = sp.LawMajewski2010()
        corner_kwargs = defaultdict(list)
        corner_kwargs["xs"] = np.zeros_like(flatchain[:,ix1:ix2])
        for ii, p_name in enumerate(potential_params):
            pp = potential.parameters[p_name]
            corner_kwargs["xs"][:,ix1+ii] = (flatchain[:,ix1+ii]*pp._unit).decompose(d_units).value
            corner_kwargs["truths"].append(pp.truth.decompose(d_units).value)
            corner_kwargs["extents"].append([x.decompose(d_units).value for x in pp.range])
            if pp._unit is not u.dimensionless_unscaled:
                label = "{0} [{1}]".format(pp.latex, pp.truth.decompose(d_units).unit)
            else:
                label = "{0}".format(pp.latex)
            corner_kwargs["labels"].append(label)

        fig = triangle.corner(**corner_kwargs)
        fig.savefig(os.path.join(path, "potential_posterior.png"))
        plt.close('all')
        ix1 = ix2
    
    chain_nwalkers,chain_nsteps,op = chain.shape
    tub_chain = None
    if infer_tub_tf:
        ix2 = ix1 + nparticles
        tub_chain = flatchain[:,ix1:ix2].reshape(chain_nwalkers*chain_nsteps,nparticles,1)
        ix1 = ix2

    if infer_particles_tf:
        logger.info("Plotting particle posteriors...")
        ix2 = ix1 + nparticles*6
        particles_flatchain = flatchain[:,ix1:ix2].reshape(chain_nsteps*chain_nwalkers,nparticles,6)
        particles_p0 = p0[:,ix1:ix2].reshape(chain_nwalkers,nparticles,6)

        for ii in range(nparticles):
            logger.debug("\tplotting particle {}".format(ii))
            this_p0 = convert_units(particles_p0[:,ii], hel_units, heliocentric.repr_units)
            true_X = convert_units(d["true_particles"]._X[ii], hel_units, heliocentric.repr_units)
            chain_X = convert_units(particles_flatchain[:,ii], hel_units, heliocentric.repr_units)

            corner_kwargs = defaultdict(list)
            corner_kwargs["xs"] = chain_X
            corner_kwargs["truths"] = true_X.tolist()
            mu,sigma = np.mean(this_p0, axis=0),np.std(this_p0, axis=0)
            corner_kwargs["extents"] = zip(mu-3*sigma,mu+3*sigma)
            labels = ["{0} [{1}]".format(n,uu)
                      for n,uu in zip(heliocentric.coord_names,heliocentric.repr_units)]
            corner_kwargs["labels"] = labels

            if tub_chain is not None:
                corner_kwargs["xs"] = np.hstack((corner_kwargs["xs"],tub_chain[:,ii]))
                corner_kwargs["truths"].append(true_tub[ii])
                corner_kwargs["extents"].append((t2,t1))
                corner_kwargs["labels"].append("$t_{ub}$")

            fig = triangle.corner(**corner_kwargs)
            fig.savefig(os.path.join(path, "particle{}_posterior.png".format(ii)))
            plt.close('all')
            gc.collect()
        ix1 = ix2

    # TODO: satellite
    return
    if s_hel is None:
        ix2 = ix1 + 6
        s_hel = p[ix1:ix2]
        ix1 = ix2

if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser(description="")
    parser.add_argument("-v", "--verbose", action="store_true",
                        dest="verbose", default=False,
                        help="Be chatty! (default = False)")
    parser.add_argument("-q", "--quiet", action="store_true", dest="quiet",
                    default=False, help="Be quiet! (default = False)")

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
        main(mpi=args.mpi, threads=args.threads,
             overwrite=args.overwrite)
    except:
        raise
        sys.exit(1)

    sys.exit(0)
