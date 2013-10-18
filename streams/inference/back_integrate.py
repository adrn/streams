# coding: utf-8

""" Contains likelihood function specific to back-integration and 
    the Rewinder 
"""

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys

# Third-party
import numpy as np

from ..inference import generalized_variance
from ..integrate.satellite_particles import satellite_particles_integrate

__all__ = ["back_integrate_likelihood", "variance_likelihood"]

def back_integrate_likelihood(p, potential_params, satellite, 
                              data_particles, data_errors, 
                              Potential, t1, t2):
    """ This is a simplified version of the likelihood laid out by D. Hogg in 
        Bread and Butter (https://github.com/davidwhogg/BreadAndButter/). The
        stars are assumed to come from a Gaussian progenitor, described by 
        just two scales -- the tidal radius and velocity dispersion.
    """

    # First need to pull apart the parameters p -- first few are the 
    #   potential parameters, then the true position of the stars, then
    #   the time the stars came unbound from their progenitor.
    Nparticles = len(data_particles)
    Nparams = len(potential_param_names)

    # Use the specified Potential class and parameters 
    potential_params = dict(zip(potential_param_names, p[:Nparams]))
    potential = Potential(**potential_params)
    
    # These are the true positions/velocities of the particles, which we 
    #   add as parameters in the model
    x = np.array(p[Nparams:Nparams+(Nparticles*6)]).reshape(Nparticles,6)
    log_p_D_given_x = -0.5 * (np.prod(data_errors**2) + np.sum((x-data_particles._x)**2 / data_errors**2, axis=-1))

    # These are the unbinding times for each particle
    t_idx = p[Nparams+(N*6):]
    
    # A Particle object for the true positions of the particles -- not great...
    particles = Particle(x[:,:3]*u.kpc, x[:,3:]*u.kpc/u.Myr, 0.*u.M_sun)
    
    acc = np.zeros((Nparticles+1,3))
    s,p = satellite_particles_integrate(satellite, particles, potential, 
                                        potential_args=(Nparticles+1, acc), 
                                        time_spec=dict(t1=t1, t2=t2, dt=-1.))
    
    Ntimesteps  = p._x.shape[0]
    
    sat_var = np.zeros((Ntimesteps,6))
    sat_var[:,:3] = potential._tidal_radius(satellite._m, s._r) * 1.26
    sat_var[:,3:] += v_disp # kpc/Myr
    cov = sat_var**2
    
    l = np.zeros((N,))
    for ii in range(N):
        pref = (0.004031441804149937 * (np.prod(cov[t_idx[ii]])**-0.5))
        yy = -0.5*np.sum((p._x[t_idx[ii],ii] - s._x[t_idx[ii],0])**2 / cov[t_idx[ii]])
        l[ii] = np.log(pref) + yy + log_p_D_given_x[ii]
    
    return np.sum(l)

def variance_likelihood(p, param_names, satellite, particles, 
                        Potential, t1, t2):
    """ Evaluate the TODO """

    model_params = dict(zip(param_names, p))
    potential = Potential(**model_params)

    Nparticles = len(particles)
    acc = np.zeros((Nparticles+1,3)) # placeholder
    s_orbit,p_orbits = satellite_particles_integrate(satellite, particles,
                                                     potential,
                                                     potential_args=(Nparticles+1, acc), \
                                                     time_spec=dict(t1=t1, t2=t2, dt=-1.))
    
    return -generalized_variance(lm10, s_orbit, p_orbits)