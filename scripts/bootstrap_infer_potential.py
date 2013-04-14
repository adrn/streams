#!/hpc/astro/users/amp2217/yt-x86_64/bin/python
# coding: utf-8

""" In this module, I'll:
        - Sample some stars from Sgr
        - Draw B bootstrap sub-samples from this set
        - For each b sub-sample, do the inference and get max. likelihood
            parameters
        - Plot up projections of the derived parameter distribution
"""

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys
import copy
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
from streams.plot.data import scatter_plot_matrix

tf_yn = dict(True="yes", False="no")

def max_likelihood_parameters(sampler):
    """ """
    ii = np.argmax(sampler.flatlnprobability)
    return flatchain[ii]

def bootstrap_sgr_snap(sgr_snap, num):
    """ Bootstrap (subsample) with replacement from sgr_snap """
    idx = np.random.randint(0, len(sgr_snap), num)
    
    sgr_snap_bootstrap = copy.deepcopy(sgr_snap)
    sgr_snap_bootstrap.x = sgr_snap.x[idx]
    sgr_snap_bootstrap.y = sgr_snap.y[idx]
    sgr_snap_bootstrap.z = sgr_snap.z[idx]
    sgr_snap_bootstrap.vx = sgr_snap.vx[idx]
    sgr_snap_bootstrap.vy = sgr_snap.vy[idx]
    sgr_snap_bootstrap.vz = sgr_snap.vz[idx]
    sgr_snap_bootstrap._set_xyz_vxyz()
    
    return sgr_snap_bootstrap

def test_bootstrap_sgr_snap():
    from streams.data import SgrSnapshot
    
    np.random.seed(42)
    sgr_snap = SgrSnapshot(num=100,
                           expr="tub > 0")
                           
    fig,axes = sgr_snap.plot_positions(subplots_kwargs=dict(figsize=(16,16)))
    
    np.random.seed(401)
    sgr_snap_bootstrap = bootstrap_sgr_snap(sgr_snap, 25)
    sgr_snap_bootstrap.plot_positions(axes=axes, scatter_kwargs={"c":"r", "alpha":0.5, "s":40})
    
    fig.savefig("plots/tests/test_bootstrap.png")

def main():
    config_kwargs = simulation_setup()
    
    # 100 different bootstrap resamples
    B = 100
    best_parameters = []
    for b in range(B):
        sgr_snap = bootstrap_sgr_snap(config_kwargs["sgr_snap"], int(config_kwargs["nparticles"]/10))
        
        ln_posterior = make_posterior(config_kwargs["params"], 
                                      sgr_snap, 
                                      config_kwargs["sgr_cen"], 
                                      config_kwargs["dt"])
        
        sampler = infer_potential(ln_posterior, **config_kwargs)
        best_parameters.append(max_likelihood_parameters(sampler))
    
    best_parameters = np.array(best_parameters)
    
    # Create a new path for the output
    path = os.path.join(output_path, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    
    if not os.path.exists(path):
        os.mkdir(path)
    
    fig, axes = scatter_plot_matrix(best_parameters, 
                        labels=[param_to_latex(param) for param in config_kwargs["params"]])
                        
    fig.savefig(os.path.join(path, "bootstrap.png"))
    
    
def infer_potential(ln_posterior, nwalkers=100, nsamples=100, nburn_in=100, 
                    params=[], output_path="/tmp", with_errors=True, mpi=False, 
                    nthreads=multiprocessing.cpu_count(), logger=None, 
                    description="", expr="", **kwargs):
    
    """ Given a log posterior function, infer the parameters for the halo model. """
    
    if len(params) == 0:
        raise ValueError("No parameters specified!")
    
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
    
    return sampler

if __name__ == "__main__":
    main()
    
    sys.exit(0)
