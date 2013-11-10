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
from streams.potential.lm10 import LawMajewski2010
from streams.simulation import config

global pool
pool = None

# Create logger
logger = logging.getLogger(__name__)

def main(config_file, job_name=None):
    # read in configurable parameters
    with open(config_file) as f:
        config = yaml.load(f.read())

    print(config)
    return

    m = "2.5e8"
    Nwalkers = 256
    Nparticles = 10
    Nburn_in = 200
    Nsteps = 300
    mpi = True
    error_factor = 0.1
    path = "/hpc/astro/users/amp2217/jobs/output_data/new_likelihood"
    #path = "/Users/adrian/projects/streams/plots/new_likelihood"
    #path = "/home/adrian/projects/streams/plots/new_likelihood"
    Nthreads = 1

    ##########################

    if mpi:
        # Initialize the MPI pool
        pool = MPIPool()

        # Make sure the thread we're running on is the master
        if not pool.is_master():
            pool.wait()
            sys.exit(0)
    elif Nthreads > 1:
        pool = multiprocessing.Pool(Nthreads)
    else:
        pool = None

    # Actually get simulation data
    np.random.seed(552)
    particles_today, satellite_today, time = mass_selector(m)
    satellite = satellite_today()
    t1,t2 = time()

    _particles = particles_today(N=Nparticles, expr="tub!=0")
    error_model = RRLyraeErrorModel(units=usys, factor=error_factor)
    obs_data, obs_error = _particles.observe(error_model)

    potential = LawMajewski2010()
    satellite = satellite_today()

    params = []
    # Potential params:
    params.append(Parameter(target=potential.q1,
                             attr="_value",
                             ln_prior=LogUniformPrior(*potential.q1._range)))
    params.append(Parameter(target=potential.qz,
                             attr="_value",
                             ln_prior=LogUniformPrior(*potential.qz._range)))
    params.append(Parameter(target=potential.v_halo,
                            attr="_value",
                            ln_prior=LogUniformPrior(*potential.v_halo._range)))
    params.append(Parameter(target=potential.phi,
                            attr="_value",
                            ln_prior=LogUniformPrior(*potential.phi._range)))
    Npotentialparams = len(params)

    # Other parameters
    params.append(Parameter(target=_particles,
                            attr="flat_X"))
    params.append(Parameter(target=_particles,
                            attr="tub",
                            ln_prior=LogUniformPrior(t2, t1)))

    model = StreamModel(potential, satellite, _particles,
                        obs_data, obs_error, parameters=params)

    ndim = sum([len(pp) for pp in params]) + Npotentialparams
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
