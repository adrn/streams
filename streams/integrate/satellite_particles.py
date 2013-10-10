# coding: utf-8

""" Special integrator for adaptively integrating a Satellite particle 
    and test particles in a potential.
"""

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys

# Third-party
import numpy as np
import astropy.units as u

from ..dynamics import Orbit
from .leapfrog import LeapfrogIntegrator

__all__ = ["satellite_particles_integrate"]

def satellite_particles_integrate(satellite, particles, potential, \
                                  potential_args=(), time_spec=dict()):
    """ TODO: """   

    # Stack positions and velocities from satellite + particle
    r_0 = np.vstack((satellite._r, particles._r))
    v_0 = np.vstack((satellite._v, particles._v))

    integrator = LeapfrogIntegrator(potential._acceleration_at, r_0, v_0,
                                    args=potential_args)
    t,r,v = integrator.run(**time_spec)
        
    sat_r = np.array(r[:,0][:,np.newaxis,:])
    sat_v = np.array(v[:,0][:,np.newaxis,:])
    
    usys = (u.kpc, u.Myr, u.M_sun)
    satellite_orbit = Orbit(t=t*u.Myr, 
                            r=sat_r*u.kpc,
                            v=sat_v*u.kpc/u.Myr,
                            m=satellite.m,
                            units=usys)

    nparticles = r.shape[1]-1
    particle_orbits = Orbit(t=t*u.Myr, 
                            r=r[:,1:]*u.kpc,
                            v=v[:,1:]*u.kpc/u.Myr,
                            m=np.zeros(nparticles)*u.M_sun,
                            units=usys)
    
    return satellite_orbit, particle_orbits

# lm10 args: (len(r_0), np.zeros((len(r_0), 3)))