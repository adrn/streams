# coding: utf-8

""" Contains priors and likelihood functions for inferring parameters of
    the Logarithmic potential using back integration.
"""

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys

# Third-party
import numpy as np
import emcee
from emcee.utils import MPIPool
import astropy.units as u

from ..potential.lm10 import param_ranges
from ..data import SgrCen, SgrSnapshot
from .core import ln_posterior as apw_ln_posterior

__all__ = [""]

class Inferenzmaschine(object):

    def __init__(self, param_names, stars, satellite_orbit, dt=None):
        """ Contains priors, likelihood function, and posterior for inferring
            the parameters of the Galactic dark matter halo, assuming a 
            Logarithmic potential. 
        """
        
        self.params = list(param_names)
        
        self.t1 = min(satellite_orbit["t"])
        self.t2 = max(satellite_orbit["t"])
        
        if dt == None:
            self.dt = satellite_orbit.dt
        else:
            self.dt = dt
            
        self.particles = stars
        self.satellite_orbit = satellite_orbit
        
    def run_mcmc(self, ln_posterior, nwalkers, nburn_in, nsteps, mpi=False, nthreads=1):
        """ Use emcee to sample from the 'posterior' with a given number of 
            walkers, for a given number of steps.
        """
        
        # Create the starting points for all walkers
        p0 = []
        for ii in range(nwalkers):
            p0.append([np.random.uniform(param_ranges[p_name][0], param_ranges[p_name][1])
                        for p_name in self.params])
        p0 = np.array(p0)
        ndim = p0.shape[1]
        
        args = (self.params, self.particles, self.satellite_orbit, self.t1, self.t2, self.dt)
        
        if mpi:
            # Initialize the MPI pool
            pool = MPIPool()
    
            # Make sure the thread we're running on is the master
            if not pool.is_master():
                pool.wait()
                sys.exit(0)
    
            self.sampler = emcee.EnsembleSampler(nwalkers, ndim, ln_posterior, 
                                                 args=args, pool=pool)
        else:
            self.sampler = emcee.EnsembleSampler(nwalkers, ndim, ln_posterior, 
                                                 args=args, threads=nthreads)
        
        # If a burn-in period is requested, run the sampler for nburn_in steps then
        #   reset the walkers and use the end positions as new initial conditions
        if nburn_in > 0:
            pos, prob, state = self.sampler.run_mcmc(p0, nburn_in)
            self.sampler.reset()
        else:
            pos = p0
        
        self.sampler.run_mcmc(pos, nsteps)
        
        # if we're running with MPI, we have to close the processor pool, otherwise
        #   the script will never finish running until the end of timmmmeeeee (echo)
        if mpi: pool.close()
        
        return self.sampler
        
def test_inferenzmaschine(nthreads=1):
    # API: 
    
    sgr_cen = SgrCen()
    ts = np.arange(max(sgr_cen["t"]), min(sgr_cen["t"]), -sgr_cen["dt"][0]*10)*u.Myr
    sgr_cen = sgr_cen.interpolate(ts)
    
    sgr_snap = SgrSnapshot(N=100, expr="tub > 0")
    maschine = Inferenzmaschine(["qz","v_halo"], 
                                stars=sgr_snap.as_particles(), 
                                satellite_orbit=sgr_cen)
    sampler = maschine.run_mcmc(apw_ln_posterior, nwalkers=4, nburn_in=1, 
                                nsteps=4, mpi=False, nthreads=nthreads)
    
    sampler.lnprobfn = None
    sampler.pickle("/tmp/test.pickle", clobber=True)