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

from ..nbody import Particle, ParticleCollection
from .leapfrog import LeapfrogIntegrator

__all__ = ["SatelliteParticleIntegrator"]

class SatelliteParticleIntegrator(LeapfrogIntegrator):
    
    def __init__(self, potential, satellite, particles):
        """ TODO """
        
        # Stack positions and velocities from satellite + particle
        r_0 = np.vstack((satellite._r, particles._r))
        v_0 = np.vstack((satellite._v, particles._v))
        
        super(SatelliteParticleIntegrator, self).__init__(potential._acceleration_at,
                                                          r_0, v_0)
    
    def _adaptive_run(self, time_spec, timestep_func, timestep_args=()):
        """ """
        
        if not time_spec.has_key("t1") or not time_spec.has_key("t2"):
            raise ValueError("You must specify t1 and t2 for adaptive "
                             "timestep integration.")
        
        t1 = time_spec['t1']
        t2 = time_spec['t2']
        
        dt_i = dt_im1 = timestep_func(self.r_im1, self.v_im1, *timestep_args)
        self._prime(dt_i)
        
        if t2 < t1 and dt_i < 0.:
            f = -1.
        elif t2 > t1 and dt_i > 0:
            f = 1.
        else:
            raise ValueError("dt must be positive or negative.")
            
        times = [t1]
        rs = self.r_im1[np.newaxis]
        vs = self.v_im1[np.newaxis]
        while f*times[-1] > f*t2:
            dt = 0.5*(dt_im1 + dt_i)
            r_i, v_i = self.step(dt)
            rs = np.vstack((rs,r_i[np.newaxis]))
            vs = np.vstack((vs,v_i[np.newaxis]))
            
            dt_i = timestep_func(r_i, v_i, *timestep_args)
            times.append(times[-1] + dt)
            dt_im1 = dt_i
        
        return np.array(times), rs, vs
    
    def run(self, time_spec=dict(), timestep_func=None, timestep_args=()):
        """ """
        
        if timestep_func is None:
            t,r,v = super(SatelliteParticleIntegrator, self)\
                        .run(time_spec=time_spec)
        else:
            t,r,v = self._adaptive_run(time_spec, timestep_func, timestep_args)
        