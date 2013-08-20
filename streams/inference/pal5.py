# coding: utf-8

""" Contains priors and likelihood functions for inferring parameters of
    the Logarithmic potential using back integration.
"""

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys
import math

# Third-party
import numpy as np
import astropy.units as u

from ..inference import generalized_variance
from ..potential.pal5 import Palomar5, true_params
from ..dynamics import OrbitCollection
from ..integrate.satellite_particles import SatelliteParticleIntegrator

__all__ = ["ln_posterior", "ln_likelihood"]

# Parameter ranges to initialize the walkers over
# v_halo range comes from 5E11 < M < 5E12, current range of MW mass @ 200 kpc
param_ranges = dict(v_halo=((100.*u.km/u.s).to(u.kpc/u.Myr).value,
                            (330.*u.km/u.s).to(u.kpc/u.Myr).value),
                    q1=(1.,2.),
                    q2=(0.5,1.5),
                    qz=(1.0,2.),
                    phi=(np.pi/4, 3*np.pi/4),
                    r_halo=(8,20)) # kpc

def objective(potential, satellite_orbit, particle_orbits):
    """ This is a new objective function, motivated by the fact that what 
        I was doing before doesn't really make sense...
    """
    
    # get numbers for any relevant loops below
    Ntimesteps, Nparticles, Ndim = particle_orbits._r.shape
    
    r_tide = potential._tidal_radius(m=satellite_orbit._m,
                                     r=satellite_orbit._r)
    v_esc = potential._escape_velocity(m=satellite_orbit._m,
                                       r_tide=r_tide)
    r_tide = r_tide[:,:,np.newaxis]
    v_esc = v_esc[:,:,np.newaxis]
    
    # compute relative, normalized coordinates and then phase-space distance
    R = particle_orbits._r - satellite_orbit._r
    V = particle_orbits._v - satellite_orbit._v
    Q = R / r_tide
    P = V / v_esc
    D_ps = np.sqrt(np.sum(Q**2, axis=-1) + np.sum(P**2, axis=-1))
    
    # velocity dispersion from measuring the dispersion of the still-bound
    #   particles from LM10
    v_disp = 0.0010 # kpc/Myr
    
    # Find the index of the time of the minimum D_ps for each particle
    min_time_idx = D_ps.argmin(axis=0)
    cov = np.zeros((6,6))
    b = np.vstack((R.T, V.T)).T
    for ii in range(Nparticles):
        idx = min_time_idx[ii]
        r_disp = np.squeeze(r_tide[idx])
        c = b[idx,ii] / np.array([r_disp]*3+[v_disp]*3)
        cov += np.outer(c, c.T)
    cov /= Nparticles
    
    sign,logdet = np.linalg.slogdet(cov)
    return logdet

def ln_p_qz(qz):
    """ Flat prior on vertical (z) axis flattening parameter. """
    lo,hi = param_ranges["qz"]
    
    if qz <= lo or qz >= hi:
        return -np.inf
    else:
        return 0.

def ln_p_m(m):
    """ Flat prior on mass of the halo
    """
    lo,hi = param_ranges["m"]
    
    if v <= lo or v >= hi:
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

def ln_likelihood(p, param_names, particles, satellite, t1, t2, resolution):
    """ Evaluate the likelihood function for a given set of halo 
        parameters.
    """
    halo_params = dict(zip(param_names, p))
    
    # LawMajewski2010 contains a disk, bulge, and logarithmic halo 
    potential = Palomar5(**halo_params)
    
    integrator = SatelliteParticleIntegrator(potential, satellite, particles, lm10=False)
    
    # not adaptive: s_orbit,p_orbits = integrator.run(t1=t1, t2=t2, dt=-1.)
    s_orbit,p_orbits = integrator.run(t1=t1, t2=t2, dt=-1.)
    
    return -objective(potential, s_orbit, p_orbits)

def ln_posterior(p, *args):
    param_names, particles, satellite, t1, t2, resolution = args
    return ln_prior(p, param_names) + ln_likelihood(p, *args)