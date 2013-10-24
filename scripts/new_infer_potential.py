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
import emcee
import matplotlib.pyplot as plt
import numpy as np
np.seterr(all="ignore")
import scipy
scipy.seterr(all="ignore")
import astropy.units as u
from astropy.io.misc import fnpickle, fnunpickle
from astropy.utils.console import color_print
import triangle

try:
    from emcee.utils import MPIPool
except ImportError:
    color_print("Failed to import MPIPool from emcee! MPI functionality "
                "won't work.", "yellow")

# Project
from streams import usys
from streams.potential.lm10 import LawMajewski2010
from streams.io.sgr import mass_selector
from streams.observation.gaia import RRLyraeErrorModel
from streams.inference import Parameter, StreamModel, LogUniformPrior
from streams.coordinates import _gc_to_hel

global pool
pool = None

######### CONFIG #########

m = "2.5e8"
Nwalkers = 128
Nparticles = 10
Nburn_in = 0
Nsteps = 500
mpi = True
error_factor = 1.
path = "/hpc/astro/users/amp2217/jobs/output_data/new_likelihood"
#path = "/Users/adrian/projects/streams/plots/new_likelihood"
#path = "/home/adrian/projects/streams/plots/new_likelihood"
Nthreads = 1

##########################

# Create logger
logger = logging.getLogger(__name__)

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
params.append(Parameter(target=_particles,
                        attr="flat_X"))

model = StreamModel(potential, satellite, _particles,
                    obs_data, obs_error, parameters=params)

Npotentialparams = 4
ndim = sum([len(pp) for pp in params]) + Npotentialparams
p0 = np.zeros((Nwalkers, ndim))
for ii in range(Npotentialparams):
    p0[:,ii] = params[ii]._ln_prior.sample(Nwalkers)

p0[:,Npotentialparams:] = _particles.flat_X * np.random.normal(1., 0.1, size=p0[:,Npotentialparams:].shape)

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

fig = triangle.corner(sampler.flatchain[:,:4],
                      truths=[1.38, 1.36, 0.125, 1.69])
fig.savefig(os.path.join(path, "corner.png"))

fig = plt.figure(figsize=(6,4))
ax = fig.add_subplot(111)
for jj in [0,1,2,3,10]:
    ax.cla()
    for ii in range(Nwalkers):
        ax.plot(sampler.chain[ii,:,jj], drawstyle='step')

    fig.savefig("{0}.png".format(jj))

sys.exit(0)