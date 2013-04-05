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

from . import back_integrate, generalized_variance
from ..potential import LawMajewski2010
from ..potential.lm10 import halo_params as true_halo_params
from ..potential.lm10 import param_ranges

__all__ = [""]

class Inferenzmaschine(object):

    def __init__(self, param_names, stars, satellite_orbit, dt=None):
        """ Contains priors, likelihood function, and posterior for inferring
            the parameters of the Galactic dark matter halo, assuming a 
            Logarithmic potential. 
        """
        
        self.params = list(param_names)
        
        self.t1 = max(satellite_orbit["t"])
        self.t2 = min(satellite_orbit["t"])
        
        if dt == None:
            self.dt = satellite_orbit["dt"][0]
        else: 
            self.dt = dt
            
        self.particles = stars
        
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
        for ii,param in enuerate(self.params):
            halo_params[param] = p[ii]
        
        # LawMajewski2010 contains a disk, bulge, and logarithmic halo 
        mw_potential = LawMajewski2010(**halo_params)
        
        ts,xs,vs = back_integrate(mw_potential, self.particles, 
                                  self.t1, self.t2, self.dt)
        return -generalized_variance(mw_potential, xs, vs, self.satellite_orbit)
    
    def ln_posterior(self, p):
        return self.ln_prior(p) + self.ln_likelihood(p)
    
def test_inferenzmaschine():
    # API: 
    sgr_cen = SgrCen()
    sgr_snap = SgrSnapshot(N=100, expr=expr)
    maschine = Inferenzmaschine(["qz","v_halo"], 
                                stars=sgr_snap.as_particles(), 
                                satellite_orbit=sgr_cen)
    sampler = maschine.run_mcmc(walkers=128, burn_in=100, steps=200, mpi=True)