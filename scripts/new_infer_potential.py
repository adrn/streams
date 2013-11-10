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
from streams.inference import Parameter, StreamModel, LogUniformPrior
from streams.io.sgr import mass_selector
from streams.observation.gaia import RRLyraeErrorModel
import streams.potential as sp
from streams.simulation import config

global pool
pool = None

# Create logger
logger = logging.getLogger(__name__)

def get_pool(config):
    """ Given a config structure, return an MPIPool, a Python
        multiprocessing.Pool, or None.
    """

    if config["mpi"]:
        # Initialize the MPI pool
        pool = MPIPool()

        # Make sure the thread we're running on is the master
        if not pool.is_master():
            pool.wait()
            sys.exit(0)
    elif config["threads"] > 1:
        pool = multiprocessing.Pool(config["threads"])
    else:
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

def main(config_file, job_name=None):
    """ TODO: """

    # read in configurable parameters
    with open(config_file) as f:
        config = yaml.load(f.read())

    # get a pool object given the configuration parameters
    pool = get_pool(config)

    # get the potential object specified
    Potential = getattr(sp, config["potential"]["class_name"])
    potential = Potential()

    # Actually get simulation data
    np.random.seed(config["seed"])
    t1,t2,satellite,_particles = read_simulation(config)

    # TODO: right now error specification in yml doesn't propagate
    factor = config.get("global_error_multiple", 1.)
    error_model = RRLyraeErrorModel(units=usys,
                                    factor=factor)
    obs_data, obs_error = _particles.observe(error_model)

    # now start collecting model parameters
    params = []
    ndim = 0

    # first add the potential parameters
    potential_params = config["potential"].get("parameters", [])
    for name in potential_params:
        p = getattr(potential, name)
        prior = LogUniformPrior(*p._range)
        params.append(Parameter(target=p, attr="_value", ln_prior=prior))
        ndim += 1

    # Other parameters
    # TODO: flat_X hack...
    particle_params = config["particles"].get("parameters", [])
    if "flat_X" in particle_params:
        particle_params.remove("flat_X")
        prior = LogUniformPrior(-100., 100.)
        params.append(Parameter(target=_particles,
                                attr="flat_X",
                                ln_prior=prior))
        ndim += config["particles"]["N"]*6

    # TODO: tub is only available for sgr...
    for name in particle_params:
        p = getattr(_particles, name)
        prior = LogUniformPrior(*p._range)
        params.append(Parameter(target=p,
                                attr="_value",
                                ln_prior=prior))
        ndim += config["particles"]["N"]

    model = StreamModel(potential, satellite, _particles,
                        obs_data, obs_error, parameters=params)

    Nwalkers = config.get("walkers", "auto")
    if str(Nwalkers).lower() == "auto":
        Nwalkers = ndim*2

    print(Nwalkers, ndim)
    return

    p0 = np.zeros((Nwalkers, ndim))
    for ii in range(Npotentialparams):
        p0[:,ii] = params[ii]._ln_prior.sample(Nwalkers)

    for ii in range(Nwalkers):
        _x = _hel_to_gc(np.random.normal(obs_data, obs_error))
        p0[ii,Npotentialparams:Nparticles*6+Npotentialparams] = np.ravel(_x)
        p0[ii,Nparticles*6+Npotentialparams:] = np.random.randint(6266, \
                                                                size=Nparticles)

    sampler = emcee.EnsembleSampler(Nwalkers, ndim, model,
                                    args=(t1, t2, -1.),
                                    pool=pool)

    if Nburn_in > 0:
        pos, xx, yy = sampler.run_mcmc(p0, Nburn_in)
        sampler.reset()
    else:
        pos = p0

    pos, prob, state = sampler.run_mcmc(pos, Nsteps)

    # write the sampler to a pickle file
    data_file = os.path.join(path, "sampler_data.pickle")
    sampler.lnprobfn = None
    sampler.pool = None
    fnpickle(sampler, data_file)

    pool.close()

    fig = triangle.corner(sampler.flatchain[:,:Npotentialparams],
                          truths=[p.target._truth for p in params[:Npotentialparams]])
    fig.savefig(os.path.join(path, "corner.png"))

    fig = plt.figure(figsize=(6,4))
    ax = fig.add_subplot(111)
    for jj in range(Npotentialparams) + [10]:
        ax.cla()
        for ii in range(Nwalkers):
            ax.plot(sampler.chain[ii,:,jj], drawstyle='step')

        fig.savefig(os.path.join(path, "{0}.png".format(jj)))

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
