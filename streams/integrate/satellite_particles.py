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

from ..misc.units import UnitSystem
from ..nbody import ParticleCollection, OrbitCollection
from .leapfrog import LeapfrogIntegrator

__all__ = ["SatelliteParticleIntegrator"]

class SatelliteParticleIntegrator(LeapfrogIntegrator):
    
    def __init__(self, potential, satellite, particles):
        """ TODO """
        
        # Stack positions and velocities from satellite + particle
        r_0 = np.vstack((satellite._r, particles._r))
        v_0 = np.vstack((satellite._v, particles._v))
        
        self.satellite_mass = satellite.m.value
        
        super(SatelliteParticleIntegrator, self).__init__(potential._acceleration_at,
                                                          r_0, v_0)
    
    def _adaptive_run(self, timestep_func, timestep_args=(), resolution=1., **time_spec):
        """ """
        
        if not time_spec.has_key("t1") or not time_spec.has_key("t2"):
            raise ValueError("You must specify t1 and t2 for adaptive "
                             "timestep integration.")
        
        t1 = time_spec['t1']
        t2 = time_spec['t2']
        
        dt_i = dt_im1 = timestep_func(self.r_im1, self.v_im1, *timestep_args) / resolution
        self._prime(dt_i)
        
        if t2 < t1 and dt_i < 0.:
            f = -1.
        elif t2 > t1 and dt_i > 0:
            f = 1.
        else:
            raise ValueError("dt must be positive or negative.")
            
        times = [t1]
        
        Ntimesteps = int(5000.*resolution)
        rs = np.zeros((Ntimesteps,) + self.r_im1.shape, dtype=float)
        vs = np.zeros((Ntimesteps,) + self.v_im1.shape, dtype=float)
        rs[0] = self.r_im1
        vs[0] = self.v_im1
        
        ii = 0
        while times[-1] > t2:
            dt = 0.5*(dt_im1 + dt_i)

            r_i, v_i = self.step(dt)
            rs[ii] = r_i
            vs[ii] = v_i
            
            dt_i = timestep_func(r_i, v_i, *timestep_args) / resolution
            times.append(times[-1] + dt)
            dt_im1 = dt_i
            ii += 1
        
        return np.array(times)[:ii], rs[:ii], vs[:ii]
    
    def run(self, timestep_func=None, timestep_args=(), resolution=5., **time_spec):
        """ """
        
        if timestep_func is None:
            t,r,v = super(SatelliteParticleIntegrator, self)\
                        .run(**time_spec)
        else:
            t,r,v = self._adaptive_run(time_spec, timestep_func, timestep_args, resolution=resolution)
        
        usys = UnitSystem(u.kpc, u.Myr, u.M_sun)
        
        sat_r = r[:,0][:,np.newaxis,:]
        sat_v = v[:,0][:,np.newaxis,:]
        satellite_orbit = OrbitCollection(t=t*u.Myr, 
                                          r=sat_r*u.kpc,
                                          v=sat_v*u.kpc/u.Myr,
                                          m=self.satellite_mass*u.M_sun,
                                          unit_system=usys)
    
        nparticles = r.shape[1]-1
        particle_orbits = OrbitCollection(t=t*u.Myr, 
                                          r=r[:,1:]*u.kpc,
                                          v=v[:,1:]*u.kpc/u.Myr,
                                          m=np.ones(nparticles)*u.M_sun,
                                          unit_system=usys)
        
        return satellite_orbit, particle_orbits