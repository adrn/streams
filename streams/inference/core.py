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

from ..simulation import generalized_variance, TestParticleOrbit
from ..potential import LawMajewski2010
from ..potential.lm10 import halo_params as true_halo_params
from ..potential.lm10 import param_ranges, param_units
from ..data import SgrCen, SgrSnapshot

__all__ = ["ln_posterior", "ln_posterior_lm10", "ln_likelihood", "ln_likelihood_lm10"]

def ln_p_qz(qz):
    """ Flat prior on vertical (z) axis flattening parameter. """
    lo,hi = param_ranges["qz"]
    
    if qz <= lo or qz >= hi:
        return -np.inf
    else:
        return 0.

def ln_p_q1(q1):
    """ Flat prior on x-axis flattening parameter. """
    lo,hi = param_ranges["q1"]
    
    if q1 <= lo or q1 >= hi:
        return -np.inf
    else:
        return 0.

def ln_p_q2(q2):
    """ Flat prior on y-axis flattening parameter. """
    lo,hi = param_ranges["q2"]
    
    if q2 <= lo or q2 >= hi:
        return -np.inf
    else:
        return 0.

def ln_p_v_halo(v):
    """ Flat prior on mass of the halo (v_halo). The range imposed is 
        roughly a halo mass between 10^10 and 10^12 M_sun at 200 kpc.
    """
    lo,hi = param_ranges["v_halo"]
    
    if v <= lo or v >= hi:
        return -np.inf
    else:
        return 0.

def ln_p_phi(phi):
    """ Flat prior on orientation angle between DM halo and disk. """
    lo,hi = param_ranges["phi"]
    
    if phi < lo or phi > hi:
        return -np.inf
    else:
        return 0.

def ln_p_r_halo(r_halo):
    """ Flat prior on halo concentration parameter. """
    lo,hi = param_ranges["r_halo"]
    
    if r_halo < lo or r_halo > hi:
        return -np.inf
    else:
        return 0.

def ln_prior(p, param_names):
    """ Joint prior over all parameters. """
    
    sum = 0.
    for ii,param in enumerate(param_names):
        f = globals()["ln_p_{0}".format(param)]
        sum += f(p[ii])
    
    return sum

# Note: if this doesn't work, I can always pass param_names in to each prior
#   and if it isn't in there, return 0...

def ln_likelihood(p, param_names, particles, satellite_ic, t):
    """ Evaluate the likelihood function for a given set of halo 
        parameters.
    """
    
    halo_params = true_halo_params.copy()
    for ii,param in enumerate(param_names):
        halo_params[param] = p[ii]*param_units[param]
    
    # LawMajewski2010 contains a disk, bulge, and logarithmic halo 
    potential = LawMajewski2010(**halo_params)
    
    satellite_orbit = satellite_ic.integrate(potential, t)
    particle_orbits = particles.integrate(potential, t)
    return -generalized_variance(potential, particle_orbits, satellite_orbit)

def ln_posterior(p, *args):
    param_names, particles, satellite_ic, t = args
    return ln_prior(p, param_names) + ln_likelihood(p, *args)

def ln_prior_lm10(p):
    """ Joint prior over all parameters. """
    return ln_p_q1(p[0]) + ln_p_qz(p[1]) + ln_p_phi(p[2]) + ln_p_v_halo(p[3])

def ln_likelihood_lm10(p, particles, satellite_ic, t):
    """ Evaluate the likelihood function for a given set of halo 
        parameters.
    """
    q1,qz,phi,v_halo = p
    
    sat_and_particles = particles.add_particle(satellite_ic)
    
    #satellite_orbit = satellite_ic._lm10_integrate(t,q1,qz,phi,v_halo)
    #particle_orbits = particles._lm10_integrate(t,q1,qz,phi,v_halo)
    orbits = sat_and_particles._lm10_integrate(t,q1,
                                               qz,phi,
                                               (v_halo*u.km/u.s).to(u.kpc/u.Myr).value)
    
    p_t = orbits.t
    p_r = orbits.r[:,:-1,:]
    p_v = orbits.v[:,:-1,:]
    particle_orbits = TestParticleOrbit(p_t, p_r, p_v)
    
    s_t = orbits.t
    s_r = orbits.r[:,-1,:][:,np.newaxis,:]
    s_v = orbits.v[:,-1,:][:,np.newaxis,:]
    satellite_orbit = TestParticleOrbit(s_t, s_r, s_v) 
    
    # LawMajewski2010 contains a disk, bulge, and logarithmic halo 
    halo_params = true_halo_params.copy()
    halo_params["q1"] = q1
    halo_params["qz"] = qz
    halo_params["v_halo"] = v_halo*u.km/u.s
    halo_params["phi"] = phi*u.radian
    potential = LawMajewski2010(**halo_params)
    
    return -generalized_variance(potential, particle_orbits, satellite_orbit)

def ln_posterior_lm10(p, *args):
    return ln_prior_lm10(p) + ln_likelihood_lm10(p, *args)