#!/hpc/astro/users/amp2217/yt-x86_64/bin/python
# coding: utf-8

""" In this module, I'll test how the distribution of 'energy distance' changes as I tweak
    various galaxy potential parameters. Ultimately, I want to come up with a way to evaluate
    the 'best' potential.
"""

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys
import datetime
import multiprocessing

# Third-party
import numpy as np
np.seterr(all="ignore")
import scipy
scipy.seterr(all="ignore")
from scipy import interpolate
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
import astropy.units as u
from astropy.io.misc import fnpickle, fnunpickle
import emcee
from emcee.utils import MPIPool

# Project
from streams.potential.lm10 import param_ranges, param_to_latex
from streams.simulation import make_posterior
from streams.simulation.setup import simulation_setup

def infer_potential(ln_posterior, nwalkers=100, nsamples=100, nburn_in=100, 
                    params=[], output_path="/tmp", with_errors=True, mpi=False, 
                    nthreads=multiprocessing.cpu_count(), logger=None, 
                    description="", expr="", nparticles=100, sgr_snap=None,
                    sgr_cen=None, **kwargs):
    
    """ Given a log posterior function, infer the parameters for the halo model. """
    
    if len(params) == 0:
        raise ValueError("No parameters specified!")
    
    if sgr_cen == None or sgr_snap == None:
        raise ValueError("You must supply sgr_snap and sgr_cen keywords!")
    
    tf_yn = dict(True="yes", False="no")
    # Create list of strings to write to run_parameters file
    run_parameters = []
    run_parameters.append("description: {0}".format(description))
    run_parameters.append("particle selection expr: {0}".format(expr))
    run_parameters.append("particles: {0}".format(nparticles))
    run_parameters.append("walkers: {0}".format(nwalkers))
    run_parameters.append("burn-in: {0}".format(nburn_in))
    run_parameters.append("samples: {0}".format(nsamples))
    run_parameters.append("halo parameters: {0}".format(", ".join(params)))
    run_parameters.append("use mpi? {0}".format(tf_yn[str(mpi)]))
    run_parameters.append("with simulated observational errors? {0}".\
                            format(tf_yn[str(with_errors)]))
    
    # Create the starting points for all walkers
    p0 = []
    for ii in range(nwalkers):
        p0.append([np.random.uniform(param_ranges[p_name][0], param_ranges[p_name][1])
                    for p_name in params])
    p0 = np.array(p0)
    ndim = p0.shape[1]

    if mpi:
        # Initialize the MPI pool
        pool = MPIPool()

        # Make sure the thread we're running on is the master
        if not pool.is_master():
            pool.wait()
            sys.exit(0)

        sampler = emcee.EnsembleSampler(nwalkers, ndim, ln_posterior, pool=pool)
    else:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, ln_posterior, threads=nthreads)
    
    # Create a new path for the output
    path = os.path.join(output_path, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    
    if not os.path.exists(path):
        os.mkdir(path)
    
    # Plot the initial positions of the particles in galactic XYZ coordinates
    fig,axes = sgr_snap.plot_positions()
    fig.savefig(os.path.join(path, "particles.png"))
    
    # If a burn-in period is requested, run the sampler for nburn_in steps then
    #   reset the walkers and use the end positions as new initial conditions
    if nburn_in > 0:
        pos, prob, state = sampler.run_mcmc(p0, nburn_in)
        sampler.reset()
    else:
        pos = p0
    
    sampler.run_mcmc(pos, nsamples)
    run_parameters.append("median acceptance fraction: {0:.3f}".\
                            format(np.median(sampler.acceptance_fraction)))
    
    # if we're running with MPI, we have to close the processor pool, otherwise
    #   the script will never finish running until the end of timmmmeeeee (echo)
    if mpi: pool.close()

    data_file = os.path.join(path, "sampler_data.pickle")
    if os.path.exists(data_file):
        os.remove(data_file)
    
    sampler.lnprobfn = None
    try:
        sampler.pickle(data_file)
    except:
        run_parameters.append("*** Unable to pickle sampler file! ***")

    idx = (sampler.acceptance_fraction > 0.1) & \
            (sampler.acceptance_fraction < 0.6) # rule of thumb, bitches
    run_parameters.append("{0} walkers ({1:.1f}%) converged"\
                            .format(sum(idx), sum(idx)/nwalkers*100))

    # If chains converged, make mcmc plots
    if len(flatchain) == 0:
        logger.warning("Not making plots -- no chains converged!")
    else:
        fig = emcee_plot(sampler, params=params, converged_idx=100, 
                         acceptance_fraction_bounds=(0.1, 0.6))
        fig.savefig(os.path.join(path, "emcee_sampler.png"), format="png")
    
    # Save the run parameters
    with open(os.path.join(path, "run_parameters"), "w") as f:
        f.write("\n".join(run_parameters))
        
    return

if __name__ == "__main__":
    config_kwargs = simulation_setup()
    
    ln_posterior = make_posterior(config_kwargs["params"], 
                                  config_kwargs["sgr_snap"], 
                                  config_kwargs["sgr_cen"], 
                                  config_kwargs["dt"])
    infer_potential(ln_posterior, **config_kwargs)
    
    sys.exit(0)
