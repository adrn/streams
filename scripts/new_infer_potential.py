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

m = "2.5e7"
Nwalkers = 64
Nparticles = 3
Nburn_in = 50
Nsteps = 100
mpi = True
error_factor = 1.
#path = "/hpc/astro/users/amp2217/jobs/output_data/new_likelihood"
#path = "/Users/adrian/projects/streams/plots/new_likelihood"
path = "/home/adrian/projects/streams/plots/new_likelihood"
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
particles_today, satellite_today, time = mass_selector("2.5e7")
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

pos, xx, yy = sampler.run_mcmc(p0, 50)
sampler.reset()
pos, prob, state = sampler.run_mcmc(pos, 100)

# write the sampler to a pickle file
data_file = os.path.join(path, "sampler_data.pickle")
sampler.lnprobfn = None
sampler.pool = None
fnpickle(sampler, data_file)

sys.exit(0)

# Plot the positions of the particles in galactic XYZ coordinates
fig,axes = true_particles.plot_r("xyz",
                        subplots_kwargs=dict(figsize=(12,12)),
                        scatter_kwargs={"alpha":0.5,"c":"k"})
data_particles.plot_r("xyz", axes=axes,
                      scatter_kwargs={"alpha":1., "c":"#CA0020"})
fig.savefig(os.path.join(path, "positions_{0}.png".format(bb)))

fig,axes = true_particles.plot_v(['vx','vy','vz'],
                        subplots_kwargs=dict(figsize=(12,12)),
                        scatter_kwargs={"alpha":0.5,"c":"k"})
data_particles.plot_v(['vx','vy','vz'], axes=axes,
                 scatter_kwargs={"alpha":1., "c":"#CA0020"})
fig.savefig(os.path.join(path, "velocities_{0}.png".format(bb)))

# write the sampler to a pickle file
data_file = os.path.join(path, "sampler_data.pickle")
sampler.lnprobfn = None
sampler.pool = None
fnpickle(sampler, data_file)

logger.debug("Making emcee trace plot...")

# make sexy plots from the sampler data
truths = [_true_params[p] for p in config["model_parameters"]]
labels = [param_to_latex[p] for p in config["model_parameters"]]
fig = emcee_plot(sampler.chain,
                 labels=labels,
                 truths=truths)

fig.savefig(os.path.join(path, "emcee_trace_{0}.png".format(bb)))

# print MAP values
idx = sampler.flatlnprobability.argmax()
best_p = sampler.flatchain[idx]
logger.info("MAP values: {0}".format(best_p))

logger.debug("Making triangle plot...")

# make triangle plot with 5-sigma ranges
extents = []
truths = []
for ii,param in enumerate(config["model_parameters"]):
    mu = np.median(sampler.flatchain[:,ii])
    sigma = np.std(sampler.flatchain[:,ii])
    extents.append((mu-5*sigma,mu+5*sigma))
    truths.append(_true_params[param])

fig = triangle.corner(sampler.flatchain,
                      labels=config["model_parameters"],
                      extents=extents,
                      truths=truths,
                      quantiles=[0.16,0.5,0.84])
fig.savefig(os.path.join(path, "triangle_{0}.png".format(bb)))

# if we're running with MPI, we have to close the processor pool, otherwise
#   the script will never finish running until the end of timmmmeeeee (echo)
if mpi: pool.close()

best_p = dict()
for ii,name in enumerate(config["model_parameters"]):
    best_p[name] = [x[ii] for x in all_best_parameters]

# finally, make plots showing the bootstrap resamples
if B > 1 and config["make_plots"]:
    fnpickle(best_p, os.path.join(path, "all_best_parameters.pickle"))

    fig,axes = plt.subplots(4,1,figsize=(14,12))
    fig.subplots_adjust(left=0.075, right=0.95)
    for ii,name in enumerate(config["model_parameters"]):
        try:
            p = true_params[name].value
        except AttributeError:
            p = true_params[name]

        axes[ii].hist(best_p[name], bins=25, histtype="step", color="k", linewidth=2)
        axes[ii].axvline(p, linestyle="--", color="#EF8A62", linewidth=3)
        axes[ii].set_ylabel(name)
        axes[ii].set_ylim(0,20)

    fig.savefig(os.path.join(path,"bootstrap_1d.png"))

    # Now make 2D plots of the bootstrap results
    if config["observational_errors"]:
        subtitle = r"$\sigma_{{RV}}={0}$ km/s; $\sigma_D={1}\%D$; {2} particles"
        subtitle = subtitle.format(rv_error, d_error, Nparticles)
    else:
        subtitle = "{0} particles".format(Nparticles)
    fig = bootstrap_scatter_plot(best_p, subtitle=subtitle)
    fig.savefig(os.path.join(path,"bootstrap_2d.png"))

if config["make_plots"]:
    with open(config_file) as f:
        g = open(os.path.join(path,"config.txt"), "w")
        g.write(f.read())
        g.close()

if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser(description="")
    parser.add_argument("-v", "--verbose", action="store_true", dest="verbose",
                    default=False, help="Be chatty! (default = False)")
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
