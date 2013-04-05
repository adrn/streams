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

from ..simulation import back_integrate, generalized_variance
from ..potential import LawMajewski2010
from ..potential.lm10 import halo_params as true_halo_params
from ..potential.lm10 import param_ranges
from ..data import SgrCen, SgrSnapshot

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
        
    def ln_p_qz(self, qz):
        """ Flat prior on vertical (z) axis flattening parameter. """
        lo,hi = param_ranges["qz"]
        
        if qz <= lo or qz >= hi:
            return -np.inf
        else:
            return 0.
    
    def ln_p_q1(self, q1):
        """ Flat prior on x-axis flattening parameter. """
        lo,hi = param_ranges["q1"]
        
        if q1 <= lo or q1 >= hi:
            return -np.inf
        else:
            return 0.
    
    def ln_p_q2(self, q2):
        """ Flat prior on y-axis flattening parameter. """
        lo,hi = param_ranges["q2"]
        
        if q2 <= lo or q2 >= hi:
            return -np.inf
        else:
            return 0.
    
    def ln_p_v_halo(self, v):
        """ Flat prior on mass of the halo (v_halo). The range imposed is 
            roughly a halo mass between 10^10 and 10^12 M_sun at 200 kpc.
        """
        lo,hi = param_ranges["v_halo"]
        
        if v <= lo or v >= hi:
            return -np.inf
        else:
            return 0.
    
    def ln_p_phi(self, phi):
        """ Flat prior on orientation angle between DM halo and disk. """
        lo,hi = param_ranges["phi"]
        
        if phi < lo or phi > hi:
            return -np.inf
        else:
            return 0.
    
    def ln_p_r_halo(self, r_halo):
        """ Flat prior on halo concentration parameter. """
        lo,hi = param_ranges["r_halo"]
        
        if r_halo < lo or r_halo > hi:
            return -np.inf
        else:
            return 0.
    
    def ln_prior(self, p):
        """ Join prior over all parameters. """
        
        sum = 0.
        for ii,param in enumerate(self.params):
            f = getattr(self,"ln_p_{0}".format(param))
            sum += f(p[ii])
        
        return sum
    
    def ln_likelihood(self, p):
        """ Evaluate the likelihood function for a given set of halo 
            parameters.
        """
        
        halo_params = true_halo_params.copy()
        for ii,param in enumerate(self.params):
            halo_params[param] = p[ii]
        
        # LawMajewski2010 contains a disk, bulge, and logarithmic halo 
        mw_potential = LawMajewski2010(**halo_params)
        
        #ts,xs,vs = back_integrate(mw_potential, self.particles, 
        #                          self.t1, self.t2, self.dt)
        ts,xs,vs = back_integrate(mw_potential, copy.copy(self.particles), 
                                  self.t1, self.t2, self.dt)
        
        return -generalized_variance(mw_potential, xs, vs, copy.copy(self.satellite_orbit))
    
    def ln_posterior(self, p):
        return self.ln_prior(p) + self.ln_likelihood(p)
    
    def run_mcmc(self, nwalkers, nburn_in, nsteps, mpi=False, nthreads=1):
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
        
        if mpi:
            # Initialize the MPI pool
            pool = MPIPool()
    
            # Make sure the thread we're running on is the master
            if not pool.is_master():
                pool.wait()
                sys.exit(0)
    
            self.sampler = emcee.EnsembleSampler(nwalkers, ndim, 
                                            self.ln_posterior, pool=pool)
        else:
            self.sampler = emcee.EnsembleSampler(nwalkers, ndim, 
                                            self.ln_posterior, threads=nthreads)
        
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
    sampler = maschine.run_mcmc(nwalkers=4, nburn_in=1, 
                                nsteps=4, mpi=False, nthreads=nthreads)
    
    sampler.lnprobfn = None
    sampler.pickle("/tmp/test.pickle", clobber=True)