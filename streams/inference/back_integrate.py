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
import astropy.units as u

from ..dynamics import Particle
from ..inference import generalized_variance
from ..integrate.satellite_particles import satellite_particles_integrate

__all__ = ["back_integrate_likelihood", "variance_likelihood"]

def _gc_to_hel(gc):
    """ Convert galactocentric to heliocentric, assuming galactic
        units: kpc, Myr, M_sun, radian

        gc should be Nparticles, Ndim
    """
    Rsun = 8. # kpc
    Vcirc = 0.224996676312 # kpc / Myr
    
    x,y,z,vx,vy,vz = gc.T

    # transform to heliocentric cartesian
    x += Rsun
    vy -= Vcirc
    
    # transform from cartesian to spherical
    d = np.sqrt(x**2 + y**2 + z**2)
    l = np.arctan2(y, x)
    b = np.pi/2. - np.arccos(z/d)
    
    # transform cartesian velocity to spherical
    d_xy = np.sqrt(x**2 + y**2)
    vr = (x*vx + y*vy + z*vz) / d # velocity 

    mu_l = -(vx*y - x*vy) / d_xy**2 # rad/Myr
    mu_b = -(z*(x*vx + y*vy) - d_xy**2*vz) / (d**2 * d_xy) # rad/Myr
    
    hel = np.zeros_like(gc).T
    for ii,col in enumerate([l,b,d,mu_l,mu_b,vr]):
        hel[ii] = col

    return hel.T

def _hel_to_gc(hel):
    """ Convert heliocentric to galactocentric, assuming galactic
        units: kpc, Myr, M_sun, radian

        gc should be Nparticles, Ndim
    """
    Rsun = 8. # kpc
    Vcirc = 0.224996676312 # kpc / Myr

    l,b,d,mul,mub,vr = hel.T

    # transform from spherical to cartesian
    x = d*np.cos(b)*np.cos(l)
    y = d*np.cos(b)*np.sin(l)
    z = d*np.sin(b)
    
    # transform spherical velocity to cartesian
    omega_l = -mul
    omega_b = -mub
    
    vx = x/d*vr + y*omega_l + z*np.cos(l)*omega_b
    vy = y/d*vr - x*omega_l + z*np.sin(l)*omega_b
    vz = z/d*vr - d*np.cos(b)*omega_b
    
    x -= Rsun
    vy += Vcirc
    
    gc = np.zeros_like(hel).T
    for ii,col in enumerate([x,y,z,vx,vy,vz]):
        gc[ii] = col

    return gc.T

def back_integrate_likelihood(p, potential_params, satellite, 
                              data, data_errors, 
                              Potential, t1, t2):
    """ This is a simplified version of the likelihood laid out by D. Hogg in 
        Bread and Butter (https://github.com/davidwhogg/BreadAndButter/). The
        stars are assumed to come from a Gaussian progenitor, described by 
        just two scales -- the tidal radius and velocity dispersion.
    """

    # First need to pull apart the parameters p -- first few are the 
    #   potential parameters, then the true position of the stars, then
    #   the time the stars came unbound from their progenitor.
    Nparticles,Ndim = data.shape
    Nparams = len(potential_params)
    dt = -1. 

    # Use the specified Potential class and parameters 
    potential_params = dict(zip(potential_params, p[:Nparams]))
    potential = Potential(**potential_params)
    
    # These are the true positions/velocities of the particles, which we 
    #   add as parameters in the model
    x = np.array(p[Nparams:Nparams+(Nparticles*6)]).reshape(Nparticles,6)
    hel = _gc_to_hel(x)

    # These are the unbinding times for each particle
    t_idx = [int(pp) for pp in p[Nparams+(Nparticles*6):]]
    
    # A Particle object for the true positions of the particles -- not great...
    particles = Particle(x[:,:3]*u.kpc, x[:,3:]*u.kpc/u.Myr, 0.*u.M_sun)
    
    acc = np.zeros((Nparticles+1,3))
    s,p = satellite_particles_integrate(satellite, particles, potential, 
                                        potential_args=(Nparticles+1, acc), 
                                        time_spec=dict(t1=t1, t2=t2, dt=dt))
    
    Ntimesteps  = p._x.shape[0]
    
    sat_var = np.zeros((Ntimesteps,6))
    sat_var[:,:3] = potential._tidal_radius(satellite._m, s._r) * 1.26
    sat_var[:,3:] += 0.0083972030362941957 #v_disp # kpc/Myr for 2.5E7
    cov = sat_var**2

    Sigma = np.array([cov[jj] for ii,jj in enumerate(t_idx)])
    p_x = np.array([p._x[jj,ii] for ii,jj in enumerate(t_idx)])
    s_x = np.array([s._x[jj,0] for ii,jj in enumerate(t_idx)])
    log_p_x_given_phi = -0.5*np.sum(-2.*np.log(Sigma) + 
                        (p_x-s_x)**2/Sigma, axis=1) * abs(dt)

    log_p_D_given_x = -0.5*np.sum(-2.*np.log(data_errors) + \
                                  (hel-data)**2/data_errors**2, axis=1)

    return np.sum(log_p_D_given_x + log_p_x_given_phi)

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