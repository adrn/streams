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
from streams.simulation.config import read
from streams.observation.gaia import add_uncertainties_to_particles, \
                                     rr_lyrae_observational_errors
from streams.inference import StatisticalModel
from streams.inference.back_integrate import back_integrate_likelihood
from streams.plot import emcee_plot, bootstrap_scatter_plot

global pool
pool = None

######### CONFIG #########

m = "2.5e7"
Nwalkers = 64
Nparticles = 3
Nburn_in = 0
Nsteps = 100
potential_params = ["q1"]
Nparams = len(potential_params)
mpi = False
#path = "/hpc/astro/users/amp2217/jobs/output_data/new_likelihood"
#path = "/Users/adrian/projects/streams/plots/new_likelihood"
path = "/home/adrian/projects/streams/plots/new_likelihood"
Nthreads = 1

##########################

# Create logger
logger = logging.getLogger(__name__)

# TODO: this is A HACK
from streams.coordinates import gc_to_hel
def particles_to_heliocentric(particles):
    # Transform to heliocentric coordinates
    return gc_to_hel(particles.r[:,0], particles.r[:,1], particles.r[:,2],
                     particles.v[:,0], particles.v[:,1], particles.v[:,2])


from streams.io.sgr import mass_selector
particles_today, satellite_today, time = mass_selector(m)
from streams.potential.lm10 import true_params, _true_params, param_to_latex
from streams.potential.lm10 import LawMajewski2010, param_ranges
    
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
satellite = satellite_today()
t1,t2 = time()
true_particles = particles_today(N=Nparticles, expr="(tub!=0)")

data_particles = add_uncertainties_to_particles(true_particles[:])
_data = particles_to_heliocentric(data_particles)
_data_errors = rr_lyrae_observational_errors(*_data)

usys = [u.kpc, u.Myr, u.radian, u.M_sun]
data = np.zeros((Nparticles, 6))
data_errors = np.zeros((Nparticles, 6))
for ii,q in enumerate(_data):
    data[:,ii] = q.decompose(usys).value
    data_errors[:,ii] = _data_errors[ii].decompose(usys).value

p0 = np.zeros((Nwalkers, Nparams + 7*Nparticles))
for ii,p_name in enumerate(potential_params):
    # Dan F-M says emcee is better at expanding than contracting...
    p0[:,ii] = np.random.uniform(param_ranges[p_name][0], 
                               param_ranges[p_name][1],
                               size=Nwalkers)

for ii in range(Nwalkers):
    p0[ii,Nparams:Nparams+6*Nparticles] = np.random.normal(data.ravel(), 
                                                           data_errors.ravel())
    p0[ii,Nparams+6*Nparticles:] += np.random.randint(0,6000,size=Nparticles)

# Create a new path for the output
if os.path.exists(path):
    shutil.rmtree(path)
        
os.mkdir(path)

def ln_prior(p, *args):
    potential_params = args[0]
    Nparams = len(potential_params)
    for pp,p_name in zip(p[:Nparams],potential_params):
        lo,hi = param_ranges[p_name]
        if pp < lo or pp > hi:
            return -np.inf

    Nparticles,Ndim = args[2].shape
    t_idx = p[Nparams+6*Nparticles:]

    for t in t_idx:
        if t < 0 or t > abs(int(t2-t1)):
            return -np.inf

    return 0.

def ln_posterior(p, *args):
    prior = ln_prior(p, *args)
    if np.isinf(prior):
        return prior

    like = back_integrate_likelihood(p, *args)
    return prior + like

largs = (potential_params, satellite, \
         data, data_errors, LawMajewski2010, t1, t2)

sampler = emcee.EnsembleSampler(nwalkers=Nwalkers, dim=p0.shape[1], 
                                lnpostfn=ln_posterior, 
                                pool=pool,
                                args=largs,
                                live_dangerously=True)
if Nburn_in > 0:
    pos, prob, state = sampler.run_mcmc(p0, Nburn_in)
    sampler.reset()
else:
    pos = p0

pos, prob, state = sampler.run_mcmc(pos, Nsteps)

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
