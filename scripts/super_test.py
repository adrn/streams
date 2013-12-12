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

def ln_prior(p, *args):
    potential,t1,t2,dt,s_hel,s_hel_err,p_hel,p_hel_err = args

    p = np.array(p)
    if np.any(p > t1) or np.any(p < t2):
        return -np.inf

    return 0.

def ln_likelihood(p, *args):

    potential,t1,t2,dt,s_hel,s_hel_err,p_hel,p_hel_err = args
    p_gc = _hel_to_gc(p_hel)
    s_gc = _hel_to_gc(s_hel)
    tub = p

    # OR
    #q1,qz,v_halo,phi = p[:4]
    #potential = LawMajewski2010(q1=q1,qz=qz,v_halo=v_halo,phi=phi)
    #p_hel = p[4:nparticles*6+4]
    #tub = p[nparticles*6+4:nparticles*6+4+nparticles]

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

    log_p_x_given_phi = -0.5*np.sum(np.log(Sigma), axis=1) - \
                         0.5*np.sum((p_x-s_x)**2/Sigma, axis=1)*abs(dt)

    return np.sum(log_p_x_given_phi)

def ln_posterior(p, *args):
    return ln_prior(p, *args) + ln_likelihood(p, *args)

def main(mpi=False, threads=None, overwrite=False):
    """ TODO: """

    pool = get_pool(mpi=mpi, threads=threads)

    ##################################################
    # determine the output data path
    home = "/vega/astro/users/amp2217/"
    #home = "/hpc/astro/users/amp2217/"
    #home = "/Users/adrian/"
    data_file = "N32_no_errors.hdf5"
    nburn = 500
    nsteps = 1000
    nparticles = 4
    nwalkers = 64
    ##################################################

    path = os.path.join(home, "output_data/super_test")
    d_path = os.path.join(home, "projects/streams/data/observed_particles/")
    d = io.read_hdf5(os.path.join(d_path, data_file))
    output_file = os.path.join(path, "inference.hdf5")

    if not os.path.exists(path):
        os.makedirs(path)

    potential = sp.LawMajewski2010()

    satellite_hel = d["satellite"]._X
    #satellite_hel_err = d["satellite"]._error_X
    satellite_hel_err = None
    logger.debug("Read in satellite".format(satellite_hel))

    particles_hel = d["particles"]._X[:nparticles]
    #particles_hel_err = d["particles"]._error_X
    particles_hel_err = None
    tub = d["particles"].tub[:nparticles]
    nparticles = particles_hel.shape[0]
    logger.debug("Read in {} particles".format(nparticles))

    t1 = float(d["t1"])
    t2 = float(d["t2"])
    dt = -1.

    p = tub.tolist()
    args = (potential, t1, t2, dt,
            satellite_hel, satellite_hel_err,
            particles_hel, particles_hel_err)

    #l = ln_posterior(p, *args) # TEST evaluating

    if nwalkers is None:
        nwalkers = len(p)*2
    p0 = np.random.uniform(t2, t1, size=(nwalkers,nparticles))
    logger.debug("{} walkers".format(nwalkers))

    sampler = emcee.EnsembleSampler(nwalkers, len(p), ln_posterior,
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

    if pool is not None:
        pool.close()

    for jj in range(nparticles):
        plt.clf()
        for ii in range(nwalkers):
            plt.plot(sampler.chain[ii,:,jj], alpha=0.4, drawstyle='steps')

        plt.axhline(tub[jj], color='k', lw=4., linestyle='--')
        plt.savefig(os.path.join(path, "walker_{}.png".format(jj)))

    fig = triangle.corner(sampler.flatchain)
    fig.savefig(os.path.join(path, "corner.png"))

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
