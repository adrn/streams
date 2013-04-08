# coding: utf-8

""" """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys

# Third-party
import numpy as np
import astropy.units as u
from astropy.utils.misc import isiterable

from ..potential import Potential
from ..integrate import leapfrog

__all__ = ["Particle", "TestParticleSimulation"]

class TestParticleSimulation(object):

    def __init__(self, potential):
        """ This is a handy class that will handle integrating a group of particles through a potential. """

        if not isinstance(potential, Potential):
            raise TypeError("potential must be a streams.Potential object.")

        self.potential = potential
        #self.particles = dict()
        self.particles = list()
        self._particle_map = dict()

        self._particle_pos_array = None
        self._particle_vel_array = None
    
    def add_particles(self, ps):
        """ Add a set of test-particles to the simulation """
        for p in ps:
            self.add_particle(p)
        
    def add_particle(self, p):
        """ Add a test-particle to the simulation (to be integrated through the potential) """

        if not isinstance(p, Particle):
            raise TypeError("add_particle() only accepts Particle objects.")

        self.particles.append(p)

        if self._particle_pos_array == None:
            self._particle_pos_array = p.position.reshape((1,p.position.shape[0]))
            self._particle_vel_array = p.velocity.reshape((1,p.velocity.shape[0]))
        else:
            self._particle_pos_array = np.append(self._particle_pos_array, p.position.reshape((1,p.position.shape[0])), axis=0)
            self._particle_vel_array = np.append(self._particle_vel_array, p.velocity.reshape((1,p.velocity.shape[0])), axis=0)

        #self._particle_map[hash(p)] = len(self._particle_pos_array)-1

    def run(self, t1, t2, dt=None, integrator=leapfrog):
        """ Integrate the particle positions from t1 to t2 using the specified integrator """

        if dt == None:
            dt = (t2 - t1) / 100.
        ts, xs, vs = integrator(self.potential.acceleration_at, self._particle_pos_array, self._particle_vel_array, t1, t2, dt)

        return ts, xs, vs

    def particles_at(self, t):
        """ Get a list of Particle objects at the given time step """
        raise NotImplementedError()

class Particle(object):
    
    def __init__(self, r, v, m):
        """ Represents a single or array of massive particles with positions 
            and velocities. Useful for integration because internally stores 
            data as arrays.
            
            Parameters
            ----------
            r : astropy.units.Quantity
                A Quantity object with position(s) aligned with the
                specified velocities.
            v : astropy.units.Quantity
                A Quantity object with velocities aligned with the
                specified position(s).
            m : astropy.units.Quantity
                A Quantity object with an array of masses aligned with the
                specified positions/velocities.
        """
                
        if not isinstance(r, u.Quantity) or \
           not isinstance(v, u.Quantity) or \
           not isinstance(m, u.Quantity):
            raise TypeError("Position, Velocity, and Mass must be Astropy "
                            "Quantity objects. You specified {0},{1},{2}."
                            .format(type(r),type(v),type(m)))
        
        if not isiterable(r.value):
            r = u.Quantity([r.value], unit=r.unit)
        
        if not isiterable(v.value):
            v = u.Quantity([v.value], unit=v.unit)
        
        if not isiterable(m.value):
            m = u.Quantity([m.value], unit=m.unit)
        
        if r.value.shape != v.value.shape:
            raise ValueError("Shape mismatch: position and velocity "
                             "shapes must match. {0}, {1}"
                             .format(r.value.shape,v.value.shape))
        
        if len(r.value.shape) > 1:
            if r.value.shape[0] != m.value.shape[0]:
                raise ValueError("Shape mismatch: mass array must be aligned "
                                 "with positions along axis=0. {0} vs {1}"
                                 .format(m.value.shape, r.value.shape))
                             
        self.r = r
        self.v = v
        self.m = m
    
    def __getitem__(self, val):
        """ Implement indexing / slicing """
        
        if isinstance(val, int) or isinstance(val, slice):
            return Particle(r=self.r[val], v=self.v[val], m=self.m[val])
        else:
            raise TypeError("Indices must be integer, not {0}".format(type(val)))
    
    def __len__(self):
        return len(self.r)
    
class Orbit(object):
    
    def __init__(self, t):
        """ Represents the orbit of a Particle """