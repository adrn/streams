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

    def __init__(self, r, v, m=1.0*u.M_sun):
        """ Represents a massive particle at some position with some velocity.
            
            Parameters
            ----------
            r : astropy.units.Quantity
                A Quantity object with a vector position.
            v : astropy.units.Quantity
                A Quantity object with a vector velocity.
            m : astropy.units.Quantity
                A Quantity object for the mass of the particle.
        """

        # Set position as array
        if not isinstance(r, u.Quantity) or not isinstance(v, u.Quantity):
            raise TypeError("Position and Velocity arrays must be Astropy "
                            "Quantity objects. You specified {0},{1}."
                            .format(type(r),type(v)))
        
        if not isinstance(m, u.Quantity):
            raise TypeError("Mass must be an Astropy Quantity object. You "
                            "specified {0}.".format(type(m)))
        
        if not isiterable(r.value):
            r = u.Quantity([r.value], unit=r.unit)
        
        if not isiterable(v.value):
            v = u.Quantity([v.value], unit=v.unit)
            
        if len(r.value.shape) > 1 or len(v.value.shape) > 1:
            raise ValueError("Particle objects cannot contain arrays. Use a "
                             "ParticleCollection instead.")
        
        if r.value.shape != v.value.shape:
            raise ValueError("Shape mismatch: position vectory doesn't "
                             "match velocity vector. {0} vs. {1}"
                             .format(_r.value.shape,_v.value.shape))
        
        self.position = self.r = r.copy()
        self.velocity = self.v = v.copy()
        self.m = self.mass = m.copy()

    def __repr__(self):
        return "<Particle m={0} at x=[{1}], v=[{2}]".format(str(self.mass), ",".join(map(str, self.position)), ",".join(map(str, self.velocity)))

    def __key(self):
        return (tuple(self.position), tuple(self.velocity), self.mass)

    def __hash__(self):
        return hash(self.__key())

class ParticleCollection(object):
    
    def __init__(self, particles=None, r=None, v=None, m=None):
        """ A collection of Particle objects. Useful for doing integration
            because internally stores data as arrays and is thus faster than
            iterating over lists of Particles.
            
            Parameters
            ----------
            particles : list, ndarray
                A list/array of Particle objects.
            r : astropy.units.Quantity
                A Quantity object with an array of positions aligned with the
                specified velocities (if Particles is not specified).
            v : astropy.units.Quantity
                A Quantity object with an array of velocities aligned with the
                specified positions (if Particles is not specified).
            m : astropy.units.Quantity (optional)
                A Quantity object with an array of masses aligned with the
                specified positions/velocities. Will default to 1 solar mass.
        """
        
        if particles != None:
            ndim = 0
            _r, _v, _m = [], [], []
            r_unit, v_unit, m_unit = None,None,None
            for p in particles:
                # Make sure all objects are Particle's
                if not isinstance(p,Particle):
                    raise TypeError("particles must be a list or array of "
                                    "particle objects! You passed a {0}."
                                    .format(type(p)))
                
                # store the dimensionality to check that all positions and
                #   velocities match
                if ndim == 0:
                    ndim = len(p.position)
                
                # store the units of the position/velocity for the final array
                if r_unit == None:
                    r_unit = p.position.unit
                    v_unit = p.velocity.unit
                    m_unit = p.mass.unit
                
                if len(p.position) != ndim or len(p.velocity) != ndim:
                    raise ValueError("Particle dimensionality ({0}) must match "
                                     "{1} for all particles in this collection."
                                     .format(len(p.position),ndim))
                
                _r.append(p.position.value)
                _v.append(p.velocity.value)
                _m.append(p.mass.value)
            
            _r = np.array(_r)*r_unit
            _v = np.array(_v)*v_unit
            _m = np.array(_m)*m_unit
        
        else:
            if r == None or v == None:
                raise ValueError("You must specify either a list/array of "
                                 "Particle objects or both an array of "
                                 "positions and an array of velocities.")
            
            elif not isinstance(r, u.Quantity) or not isinstance(v, u.Quantity):
                raise TypeError("Position and Velocity arrays must be Astropy "
                                "Quantity objects. You specified {0},{1}."
                                .format(type(r),type(v)))
            
            if m != None and not isinstance(m, u.Quantity):
                raise TypeError("Mass array must be an Astropy Quantity object."
                                " You specified {0}.".format(type(m)))
            
            if r.value.shape != v.value.shape:
                raise ValueError("Shape mismatch: position vector doesn't "
                                 "match velocity vector. {0} vs. {1}"
                                 .format(r.value.shape,v.value.shape))
            
            if m == None:
                m = np.ones_like(r.value)*u.M_sun
            else:
                if r.value.shape != m.value.shape:
                    raise ValueError("Shape mismatch: mass vector doesn't "
                                     "match other shapes. {0} vs. {1}"
                                     .format(m.value.shape,r.value.shape))
            
            _r = r
            _v = v
            _m = m
            
        self.r = _r
        self.v = _v
        self.m = _m
    
    def __getitem__(self, val):
        """ Implement indexing / slicing """
        
        if isinstance(val, int):
            return Particle(r=self.r[val], v=self.v[val], m=self.m[val])
        elif isinstance(val, slice):
            return ParticleCollection(r=self.r[val], v=self.v[val], m=self.m[val])
        else:
            raise TypeError("Indices must be integer, not {0}".format(type(val)))
        
